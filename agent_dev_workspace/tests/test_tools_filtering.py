import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_filtering import make_tools as make_filtering_tools

def test_variance_filter_tool():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Build DTM using tools_dtm.py
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Instantiate variance_filter_tool from tools_filtering.py
    filtering_tools = make_filtering_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")

    # Invoke variance_filter_tool with threshold=0.15 (Known answer #5)
    res = variance_filter_tool.invoke({"threshold": 0.15})

    # Assert known answer #5 expectations
    assert res["kept_count"] == 4
    assert res["removed_count"] == 1
    assert "always" in res["removed_features"]
    assert "alpha" in res["kept_features"]
    assert "beta" in res["kept_features"]
    assert "delta" in res["kept_features"]
    assert "gamma" in res["kept_features"]

    # Verify full_report in results_store and check variances
    result_id = res["result_id"]
    assert result_id in session.results_store
    full_report = session.results_store[result_id]["full_report"]

    variance_map = dict(zip(full_report["term"], full_report["variance"]))
    assert variance_map["alpha"] == pytest.approx(1.1875, abs=1e-3)
    assert variance_map["always"] == pytest.approx(0.109375, abs=1e-3)
    assert variance_map["beta"] == pytest.approx(0.234375, abs=1e-3)
    assert variance_map["delta"] == pytest.approx(0.25, abs=1e-3)
    assert variance_map["gamma"] == pytest.approx(1.109375, abs=1e-3)


def test_pearson_filter_tool():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Build DTM and set labels using tools_dtm.py
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Instantiate pearson_filter_tool from tools_filtering.py
    filtering_tools = make_filtering_tools(session)
    pearson_filter_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")

    # Invoke pearson_filter_tool with target_class="catB" (Known answer #6)
    res = pearson_filter_tool.invoke({"target_class": "catB", "threshold": 0.0})

    # Verify full_report in results_store and check pearson_r values
    result_id = res["result_id"]
    assert result_id in session.results_store
    full_report = session.results_store[result_id]["full_report"]

    pearson_map = dict(zip(full_report["term"], full_report["pearson_r"]))
    assert pearson_map["alpha"] == pytest.approx(-0.6882, abs=1e-3)
    assert pearson_map["always"] == pytest.approx(0.3780, abs=1e-3)
    assert pearson_map["beta"] == pytest.approx(-0.7746, abs=1e-3)
    assert pearson_map["delta"] == pytest.approx(1.0, abs=1e-3)
    assert pearson_map["gamma"] == pytest.approx(0.8307, abs=1e-3)


def test_spearman_filter_tool():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Build DTM and set labels using tools_dtm.py
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Instantiate spearman_filter_tool from tools_filtering.py
    filtering_tools = make_filtering_tools(session)
    spearman_filter_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")

    # Invoke spearman_filter_tool with target_class="catB" (Known answer #6)
    res = spearman_filter_tool.invoke({"target_class": "catB", "threshold": 0.0})

    # Verify full_report in results_store and check spearman_r values
    result_id = res["result_id"]
    assert result_id in session.results_store
    full_report = session.results_store[result_id]["full_report"]

    spearman_map = dict(zip(full_report["term"], full_report["spearman_r"]))
    assert spearman_map["alpha"] == pytest.approx(-0.7500, abs=1e-3)
    assert spearman_map["always"] == pytest.approx(0.3780, abs=1e-3)
    assert spearman_map["beta"] == pytest.approx(-0.7746, abs=1e-3)
    assert spearman_map["delta"] == pytest.approx(1.0, abs=1e-3)
    assert spearman_map["gamma"] == pytest.approx(0.9363, abs=1e-3)
