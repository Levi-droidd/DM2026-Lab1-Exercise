import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools as make_patterns_tools


def _setup_session():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })
    return session


def test_mine_patterns_zero_survived_filtering():
    session = _setup_session()
    tools = make_patterns_tools(session)
    mine_patterns_tool = next(t for t in tools if t.name == "mine_patterns_tool")

    # Known answer #17: variance filtering keeps zero terms for both categories
    res_var = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "variance",
        "algorithm": "fpgrowth",
        "min_sup": 1,
    })
    assert res_var["n_patterns"] == 0
    assert res_var["patterns"] == []
    assert "no terms survived" in res_var.get("note", "")

    # Term frequency filtering also keeps zero terms for both categories
    res_tf = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "term_frequency",
        "algorithm": "fpgrowth",
        "min_sup": 1,
    })
    assert res_tf["n_patterns"] == 0
    assert res_tf["patterns"] == []
    assert "no terms survived" in res_tf.get("note", "")


def test_mine_patterns_fpgrowth_known_answer_17():
    session = _setup_session()
    tools = make_patterns_tools(session)
    mine_patterns_tool = next(t for t in tools if t.name == "mine_patterns_tool")

    # Known answer #17: catA keeps only alpha (support 3 at min_sup=1 to 3, 0 at min_sup=4)
    res_catA = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1,
    })
    assert res_catA["n_patterns"] == 1
    assert res_catA["patterns"][0]["pattern"] == ["alpha"]
    assert res_catA["patterns"][0]["support"] == 3

    # At min_sup=4, catA should yield 0 patterns
    res_catA_4 = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 4,
    })
    assert res_catA_4["n_patterns"] == 0
    assert res_catA_4["patterns"] == []

    # catB keeps only gamma (support 4 at min_sup=1 to 4)
    res_catB = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 4,
    })
    assert res_catB["n_patterns"] == 1
    assert res_catB["patterns"][0]["pattern"] == ["gamma"]
    assert res_catB["patterns"][0]["support"] == 4


def test_mine_patterns_topk_and_maxfpgrowth_known_answer_20():
    session = _setup_session()
    tools = make_patterns_tools(session)
    mine_patterns_tool = next(t for t in tools if t.name == "mine_patterns_tool")

    # Known answer #20: topk at k=1 finds {alpha} (sup 3) for catA, {gamma} (sup 4) for catB
    res_topk_a = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 1,
    })
    assert res_topk_a["n_patterns"] == 1
    assert res_topk_a["patterns"][0]["pattern"] == ["alpha"]
    assert res_topk_a["patterns"][0]["support"] == 3

    res_topk_b = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 1,
    })
    assert res_topk_b["n_patterns"] == 1
    assert res_topk_b["patterns"][0]["pattern"] == ["gamma"]
    assert res_topk_b["patterns"][0]["support"] == 4

    # Known answer #20: maxfpgrowth returns 0 patterns for both categories without error
    res_max_a = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "maxfpgrowth",
        "min_sup": 1,
    })
    assert res_max_a["n_patterns"] == 0
    assert res_max_a["patterns"] == []

    res_max_b = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "maxfpgrowth",
        "min_sup": 1,
    })
    assert res_max_b["n_patterns"] == 0
    assert res_max_b["patterns"] == []


def test_mine_patterns_parameter_validation_errors():
    session = _setup_session()
    tools = make_patterns_tools(session)
    mine_patterns_tool = next(t for t in tools if t.name == "mine_patterns_tool")

    # Missing min_sup for fpgrowth
    res_err1 = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
    })
    assert res_err1.get("status") == "error"
    assert "min_sup is required" in res_err1.get("message", "")

    # Missing k for topk
    res_err2 = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "topk",
    })
    assert res_err2.get("status") == "error"
    assert "k is required" in res_err2.get("message", "")
