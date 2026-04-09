from __future__ import annotations
import sqlite3
from dataclasses import dataclass
from cidre.db import search_by_vector, get_items_by_rowids, ItemRow
from cidre.search.query import parse_query, QueryIntent


@dataclass
class SearchResult:
    file_path: str
    file_type: str
    ai_description: str
    categories: list[str]
    summary: str
    file_size: int
    modified_at: str
    score: float
    source: str


class SearchEngine:
    def __init__(self, conn: sqlite3.Connection, embedder):
        self._conn = conn
        self._embedder = embedder

    def search(
        self,
        query: str,
        k: int = 20,
        file_type: str | None = None,
        category: str | None = None,
    ) -> list[SearchResult]:
        intent = parse_query(query)
        if file_type:
            intent.file_type = file_type
        if category:
            intent.category = category

        query_embedding = self._embedder.embed([intent.text])[0]
        vector_results = search_by_vector(self._conn, query_embedding, k=k * 2)

        if not vector_results:
            return []

        rowids = [r[0] for r in vector_results]
        distances = {r[0]: r[1] for r in vector_results}
        items = get_items_by_rowids(self._conn, rowids)

        results = []
        for item in items:
            if intent.file_type and item.file_type != intent.file_type:
                continue
            if intent.date_start and item.modified_at < intent.date_start.isoformat():
                continue
            if intent.date_end and item.modified_at > intent.date_end.isoformat():
                continue
            if intent.category and intent.category not in item.categories:
                continue

            distance = distances.get(item.rowid, 1.0)
            score = max(0.0, 1.0 - distance)

            results.append(SearchResult(
                file_path=item.file_path,
                file_type=item.file_type,
                ai_description=item.ai_description,
                categories=item.categories,
                summary=item.summary,
                file_size=item.file_size,
                modified_at=item.modified_at,
                score=round(score, 4),
                source=item.source,
            ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]
