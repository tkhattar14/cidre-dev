import sqlite3
from pathlib import Path
from cidre.db import init_db, insert_item, search_by_metadata, get_item_by_path, ItemRow


def test_init_db_creates_tables(tmp_path):
    db_path = tmp_path / "cidre.db"
    conn = init_db(db_path, embedding_dimensions=768)
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor.fetchall()}
    assert "items" in tables
    conn.close()


def test_insert_and_retrieve_item(tmp_path):
    db_path = tmp_path / "cidre.db"
    conn = init_db(db_path, embedding_dimensions=768)
    embedding = [0.1] * 768

    insert_item(conn, ItemRow(
        file_path="/Users/test/photo.jpg",
        file_hash="abc123",
        file_type="photo",
        file_size=2400000,
        modified_at="2024-06-15T10:30:00",
        ai_description="A sunset over the ocean",
        categories=["travel", "landscape"],
        summary="Sunset photo from beach vacation",
        embedding=embedding,
        source="filesystem",
    ))

    item = get_item_by_path(conn, "/Users/test/photo.jpg")
    assert item is not None
    assert item.ai_description == "A sunset over the ocean"
    assert item.categories == ["travel", "landscape"]
    assert item.file_hash == "abc123"
    conn.close()


def test_search_by_metadata_type_filter(tmp_path):
    db_path = tmp_path / "cidre.db"
    conn = init_db(db_path, embedding_dimensions=768)
    embedding = [0.1] * 768

    insert_item(conn, ItemRow(
        file_path="/test/doc.pdf",
        file_hash="hash1",
        file_type="document",
        file_size=100000,
        modified_at="2025-01-10T00:00:00",
        ai_description="Health insurance invoice",
        categories=["receipt", "insurance"],
        summary="Niva Bupa invoice",
        embedding=embedding,
        source="filesystem",
    ))
    insert_item(conn, ItemRow(
        file_path="/test/photo.jpg",
        file_hash="hash2",
        file_type="photo",
        file_size=200000,
        modified_at="2025-01-10T00:00:00",
        ai_description="Mountain landscape",
        categories=["landscape"],
        summary="Mountain photo",
        embedding=embedding,
        source="filesystem",
    ))

    results = search_by_metadata(conn, file_type="document")
    assert len(results) == 1
    assert results[0].file_path == "/test/doc.pdf"
    conn.close()
