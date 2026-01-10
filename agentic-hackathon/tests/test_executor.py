from src.tools.rag_tool import rag_lookup


def test_rag_lookup_missing_index(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_INDEX_PATH", str(tmp_path / "missing"))
    output = rag_lookup("test query")
    assert "RAG index missing" in output