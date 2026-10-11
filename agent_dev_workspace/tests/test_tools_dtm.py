import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools


def test_build_dtm_tool_fixture():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load the fixture data
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Get build_dtm_tool from tools_dtm.py
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")

    # Invoke build_dtm_tool
    res = build_dtm_tool.invoke({})

    # Check Known Answer #4 from TEST_FIXTURE.md
    assert res["n_terms"] == 5
    assert session.feature_names == ["alpha", "always", "beta", "delta", "gamma"]
    assert res["non_zero"] == 21
    assert res["total_elements"] == 40
    assert np.isclose(res["sparsity_pct"], 47.5)

    # Check that count_vectorizer artifact is saved in session.artifacts
    assert "count_vectorizer" in session.artifacts
    assert session.artifacts["count_vectorizer"] is not None


def test_term_frequency_tool_fixture():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load fixture data
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    term_frequency_tool = next(t for t in dtm_tools if t.name == "term_frequency_tool")

    # Build DTM first
    build_dtm_tool.invoke({})

    # Run term_frequency_tool
    res = term_frequency_tool.invoke({"top_n": 10})

    # Convert returned list of dicts to a map for exact term checking
    freq_map = {item["term"]: item["frequency"] for item in res["term_frequencies"]}

    # Assert exact frequencies from Known Answer #14 in TEST_FIXTURE.md
    expected_frequencies = {
        "alpha": 6,
        "always": 7,
        "beta": 3,
        "delta": 4,
        "gamma": 7,
    }

    assert freq_map == expected_frequencies


def test_dtm_heatmap_tool_fixture():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load fixture data
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    dtm_heatmap_tool = next(t for t in dtm_tools if t.name == "dtm_heatmap_tool")

    # Build DTM first
    build_dtm_tool.invoke({})

    # 1. Test full matrix slice defaults (20x20 on 8x5 matrix)
    res_full = dtm_heatmap_tool.invoke({})

    expected_full_matrix = [
        [3, 1, 1, 0, 0],
        [2, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 1, 0, 1, 3],
        [0, 1, 0, 1, 2],
        [0, 1, 0, 1, 1],
        [0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0],
    ]

    expected_doc_labels = [f"Doc {i}" for i in range(8)]

    assert res_full["n_documents_sliced"] == 8
    assert res_full["n_terms_sliced"] == 5
    assert res_full["terms"] == ["alpha", "always", "beta", "delta", "gamma"]
    assert res_full["documents"] == expected_doc_labels
    assert res_full["matrix_slice"] == expected_full_matrix
    assert session.pending_figure is not None

    # 2. Test corner slice (n_documents=3, n_terms=2)
    res_corner = dtm_heatmap_tool.invoke({"n_documents": 3, "n_terms": 2})
    assert res_corner["n_documents_sliced"] == 3
    assert res_corner["n_terms_sliced"] == 2
    assert res_corner["terms"] == ["alpha", "always"]
    assert res_corner["documents"] == ["Doc 0", "Doc 1", "Doc 2"]
    assert res_corner["matrix_slice"] == [
        [3, 1],
        [2, 1],
        [1, 1],
    ]
