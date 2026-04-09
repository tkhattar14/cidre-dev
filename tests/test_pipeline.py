from pathlib import Path
from unittest.mock import MagicMock, patch
from cidre.indexer.pipeline import IndexingPipeline
from cidre.db import init_db, get_item_by_path


def test_pipeline_indexes_markdown(tmp_path):
    db_path = tmp_path / "cidre.db"
    conn = init_db(db_path, embedding_dimensions=768)

    md_file = tmp_path / "docs" / "notes.md"
    md_file.parent.mkdir()
    md_file.write_text("# Meeting Notes\nDiscussed project timeline and deadlines.")

    mock_llm = MagicMock()
    mock_llm.generate.return_value = (
        "Description: Meeting notes about project timeline\n"
        "Categories: work, notes, meeting\n"
        "Summary: Team meeting notes discussing deadlines"
    )

    mock_embedder = MagicMock()
    mock_embedder.dimensions = 768
    mock_embedder.embed.return_value = [[0.1] * 768]

    pipeline = IndexingPipeline(conn=conn, llm=mock_llm, embedder=mock_embedder)
    pipeline.index_file(md_file, file_type="markdown")

    item = get_item_by_path(conn, str(md_file))
    assert item is not None
    assert "meeting" in item.ai_description.lower()
    assert "work" in item.categories
    conn.close()


def test_pipeline_skips_unchanged_file(tmp_path):
    db_path = tmp_path / "cidre.db"
    conn = init_db(db_path, embedding_dimensions=768)

    md_file = tmp_path / "notes.md"
    md_file.write_text("# Notes")

    mock_llm = MagicMock()
    mock_llm.generate.return_value = (
        "Description: Simple notes\nCategories: notes\nSummary: Notes file"
    )
    mock_embedder = MagicMock()
    mock_embedder.dimensions = 768
    mock_embedder.embed.return_value = [[0.1] * 768]

    pipeline = IndexingPipeline(conn=conn, llm=mock_llm, embedder=mock_embedder)
    pipeline.index_file(md_file, file_type="markdown")
    pipeline.index_file(md_file, file_type="markdown")

    # LLM should only be called once — second call skipped due to same hash
    assert mock_llm.generate.call_count == 1
    conn.close()
