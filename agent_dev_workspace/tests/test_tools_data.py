import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools

def test_check_missing_tool():
    # 1. Initialize shared session and load premade tools
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # 2. Load dataset into session
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 3. Instantiate student tools from tools_data.py
    data_tools = make_tools(session)
    check_missing_tool = next(t for t in data_tools if t.name == "check_missing_tool")

    # 4. Invoke check_missing_tool
    res = check_missing_tool.invoke({})

    # 5. Assert known answer #2 expectations
    assert res["missing_count"] == 1
    assert res["missing_indices"] == [7]
    assert res["missing_indices_truncated"] is False

    # 6. Verify result is stored in session.results_store
    result_id = res["result_id"]
    assert result_id in session.results_store
    record = session.results_store[result_id]["record"]
    assert record.summary["missing_count"] == 1


def test_check_duplicates_tool():
    # 1. Fresh session to avoid mutation leakage across tests
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_tools(session)
    check_duplicates_tool = next(t for t in data_tools if t.name == "check_duplicates_tool")

    # 2. Test drop=False (Check duplicates without dropping)
    res_no_drop = check_duplicates_tool.invoke({"drop": False})
    assert res_no_drop["duplicate_count"] == 1
    assert res_no_drop["dropped"] is False
    assert res_no_drop["rows_remaining"] == 8

    result_id_no_drop = res_no_drop["result_id"]
    assert result_id_no_drop in session.results_store
    record_no_drop = session.results_store[result_id_no_drop]["record"]
    assert record_no_drop.summary["duplicate_count"] == 1

    # 3. Test drop=True (Drop duplicates using keep=False)
    res_drop = check_duplicates_tool.invoke({"drop": True})
    assert res_drop["dropped"] is True
    assert res_drop["rows_remaining"] == 6

    result_id_drop = res_drop["result_id"]
    assert result_id_drop in session.results_store
    record_drop = session.results_store[result_id_drop]["record"]
    assert record_drop.summary["rows_remaining"] == 6
