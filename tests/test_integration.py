"""End-to-end integration test using mocked Ollama responses."""
from pathlib import Path
from unittest.mock import MagicMock

from cidre.config import CidreConfig, save_config
from cidre.db import init_db, get_index_stats, get_item_by_path
from cidre.indexer.scanner import scan_directory
from cidre.indexer.pipeline import IndexingPipeline
from cidre.search.engine import SearchEngine


def test_full_index_and_search_flow(tmp_path):
    # Setup config
    cidre_home = tmp_path / ".cidre"
    cidre_home.mkdir()
    config = CidreConfig(embedding_dimensions=32)
    save_config(config, cidre_home / "config.toml")

    # Create test files
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "meeting.md").write_text("# Q1 Planning\nBudget review and timeline.")
    (docs / "recipe.md").write_text("# Pasta Recipe\nBoil water, add pasta, cook 10 mins.")
    (docs / "code.py").write_text("print('hello')")  # should be excluded

    # Init DB
    conn = init_db(cidre_home / "cidre.db", embedding_dimensions=32)

    # Mock LLM
    mock_llm = MagicMock()

    def fake_generate(prompt):
        if "Q1 Planning" in prompt or "Budget" in prompt:
            return "Description: Quarterly planning meeting notes\nCategories: work, meeting, planning\nSummary: Q1 budget and timeline review"
        return "Description: Italian pasta cooking recipe\nCategories: food, recipe, cooking\nSummary: Simple pasta recipe with instructions"

    mock_llm.generate.side_effect = fake_generate

    # Mock embedder
    mock_embedder = MagicMock()
    mock_embedder.dimensions = 32

    def fake_embed(texts):
        results = []
        for text in texts:
            if "planning" in text.lower() or "meeting" in text.lower():
                results.append([0.9, 0.1] + [0.0] * 30)
            elif "pasta" in text.lower() or "recipe" in text.lower():
                results.append([0.0, 0.0] + [0.9, 0.1] + [0.0] * 28)
            else:
                results.append([0.5] * 32)
        return results

    mock_embedder.embed.side_effect = fake_embed

    # Scan and index
    scanned = scan_directory(docs, exclude_patterns=["*.py"])
    assert len(scanned) == 2  # only .md files

    pipeline = IndexingPipeline(conn=conn, llm=mock_llm, embedder=mock_embedder)
    count = pipeline.index_batch([(s.path, s.file_type) for s in scanned])
    assert count == 2

    # Verify stats
    stats = get_index_stats(conn)
    assert stats["total"] == 2
    assert stats["markdown"] == 2

    # Search for meeting-related content
    engine = SearchEngine(conn=conn, embedder=mock_embedder)
    results = engine.search("quarterly planning meeting")
    assert len(results) >= 1
    assert "meeting" in results[0].ai_description.lower() or "planning" in results[0].ai_description.lower()

    # Search for recipe
    results = engine.search("pasta cooking recipe")
    assert len(results) >= 1
    assert "pasta" in results[0].ai_description.lower()

    # Verify code.py was excluded
    assert get_item_by_path(conn, str(docs / "code.py")) is None

    conn.close()
