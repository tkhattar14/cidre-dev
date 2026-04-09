from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import sqlite_vec


@dataclass
class ItemRow:
    file_path: str
    file_hash: str
    file_type: str  # "photo", "video", "document", "markdown"
    file_size: int
    modified_at: str
    ai_description: str
    categories: list[str]
    summary: str
    embedding: list[float]
    source: str  # "filesystem" | "apple_photos"
    rowid: int | None = None


def init_db(db_path: Path, embedding_dimensions: int) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    conn.executescript(f"""
        CREATE TABLE IF NOT EXISTS items (
            rowid INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT UNIQUE NOT NULL,
            file_hash TEXT NOT NULL,
            file_type TEXT NOT NULL,
            file_size INTEGER NOT NULL,
            modified_at TEXT NOT NULL,
            ai_description TEXT NOT NULL,
            categories TEXT NOT NULL DEFAULT '[]',
            summary TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'filesystem'
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS items_vec USING vec0(
            rowid INTEGER PRIMARY KEY,
            embedding float[{embedding_dimensions}]
        );

        CREATE INDEX IF NOT EXISTS idx_items_file_type ON items(file_type);
        CREATE INDEX IF NOT EXISTS idx_items_source ON items(source);
        CREATE INDEX IF NOT EXISTS idx_items_modified_at ON items(modified_at);
    """)
    return conn


def insert_item(conn: sqlite3.Connection, item: ItemRow) -> int:
    cursor = conn.execute(
        """INSERT OR REPLACE INTO items
           (file_path, file_hash, file_type, file_size, modified_at,
            ai_description, categories, summary, source)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            item.file_path, item.file_hash, item.file_type, item.file_size,
            item.modified_at, item.ai_description, json.dumps(item.categories),
            item.summary, item.source,
        ),
    )
    rowid = cursor.lastrowid
    conn.execute(
        "INSERT OR REPLACE INTO items_vec (rowid, embedding) VALUES (?, ?)",
        (rowid, json.dumps(item.embedding)),
    )
    conn.commit()
    return rowid


def get_item_by_path(conn: sqlite3.Connection, file_path: str) -> ItemRow | None:
    row = conn.execute(
        "SELECT rowid, * FROM items WHERE file_path = ?", (file_path,)
    ).fetchone()
    if row is None:
        return None
    return _row_to_item(row)


def search_by_metadata(
    conn: sqlite3.Connection,
    file_type: str | None = None,
    source: str | None = None,
    category: str | None = None,
) -> list[ItemRow]:
    query = "SELECT rowid, * FROM items WHERE 1=1"
    params: list = []
    if file_type:
        query += " AND file_type = ?"
        params.append(file_type)
    if source:
        query += " AND source = ?"
        params.append(source)
    if category:
        query += " AND categories LIKE ?"
        params.append(f'"%{category}%"')

    rows = conn.execute(query, params).fetchall()
    return [_row_to_item(r) for r in rows]


def search_by_vector(
    conn: sqlite3.Connection,
    query_embedding: list[float],
    k: int = 20,
) -> list[tuple[int, float]]:
    """Returns list of (rowid, distance) pairs."""
    rows = conn.execute(
        "SELECT rowid, distance FROM items_vec WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
        (json.dumps(query_embedding), k),
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


def get_items_by_rowids(conn: sqlite3.Connection, rowids: list[int]) -> list[ItemRow]:
    if not rowids:
        return []
    placeholders = ",".join("?" * len(rowids))
    rows = conn.execute(
        f"SELECT rowid, * FROM items WHERE rowid IN ({placeholders})", rowids
    ).fetchall()
    return [_row_to_item(r) for r in rows]


def get_index_stats(conn: sqlite3.Connection) -> dict:
    stats = {}
    row = conn.execute("SELECT COUNT(*) FROM items").fetchone()
    stats["total"] = row[0]
    for ftype in ("photo", "video", "document", "markdown"):
        row = conn.execute("SELECT COUNT(*) FROM items WHERE file_type = ?", (ftype,)).fetchone()
        stats[ftype] = row[0]
    row = conn.execute("SELECT COUNT(*) FROM items WHERE source = 'apple_photos'").fetchone()
    stats["apple_photos"] = row[0]
    return stats


def _row_to_item(row: tuple) -> ItemRow:
    # SELECT rowid, * returns: rowid(0), rowid_pk(1), file_path(2), file_hash(3),
    # file_type(4), file_size(5), modified_at(6), ai_description(7),
    # categories(8), summary(9), source(10)
    return ItemRow(
        rowid=row[0],
        file_path=row[2],
        file_hash=row[3],
        file_type=row[4],
        file_size=row[5],
        modified_at=row[6],
        ai_description=row[7],
        categories=json.loads(row[8]),
        summary=row[9],
        source=row[10],
        embedding=[],  # not loaded from items table
    )
