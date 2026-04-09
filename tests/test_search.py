from datetime import datetime
from cidre.search.query import parse_query, QueryIntent
from cidre.search.engine import SearchEngine
from cidre.db import init_db, insert_item, ItemRow
from unittest.mock import MagicMock


def test_parse_simple_query():
    intent = parse_query("sunset photos")
    assert intent.text == "sunset photos"
    assert intent.file_type is None
    assert intent.date_start is None


def test_parse_query_with_photo_hint():
    intent = parse_query("photos of mountains")
    assert intent.text == "mountains"
    assert intent.file_type == "photo"


def test_parse_query_with_date():
    intent = parse_query("invoices from 2025")
    assert intent.text == "invoices"
    assert intent.date_start is not None
    assert intent.date_start.year == 2025
    assert intent.date_end.year == 2025


def test_parse_query_last_year():
    intent = parse_query("receipts from last year")
    now = datetime.now()
    assert intent.date_start.year == now.year - 1


def test_search_engine_returns_ranked_results(tmp_path):
    db_path = tmp_path / "cidre.db"
    conn = init_db(db_path, embedding_dimensions=4)

    insert_item(conn, ItemRow(
        file_path="/test/sunset.jpg",
        file_hash="h1",
        file_type="photo",
        file_size=1000,
        modified_at="2024-06-15T00:00:00",
        ai_description="A golden sunset over the ocean",
        categories=["landscape", "travel"],
        summary="Sunset photo",
        embedding=[0.9, 0.1, 0.0, 0.0],
        source="filesystem",
    ))
    insert_item(conn, ItemRow(
        file_path="/test/invoice.pdf",
        file_hash="h2",
        file_type="document",
        file_size=2000,
        modified_at="2025-01-10T00:00:00",
        ai_description="Health insurance invoice from Niva Bupa",
        categories=["receipt", "insurance"],
        summary="Insurance invoice",
        embedding=[0.0, 0.0, 0.9, 0.1],
        source="filesystem",
    ))

    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [[0.85, 0.15, 0.0, 0.0]]

    engine = SearchEngine(conn=conn, embedder=mock_embedder)
    results = engine.search("sunset photos")

    assert len(results) >= 1
    assert results[0].file_path == "/test/sunset.jpg"
    conn.close()
