from __future__ import annotations
import sqlite3
from pathlib import Path
import pdfplumber
from cidre.db import insert_item, get_item_by_path, ItemRow
from cidre.indexer.scanner import file_hash
from cidre.vision.describe import describe_image, describe_document, describe_markdown, describe_video


class IndexingPipeline:
    def __init__(self, conn: sqlite3.Connection, llm, embedder):
        self._conn = conn
        self._llm = llm
        self._embedder = embedder

    def index_file(self, path: Path, file_type: str, source: str = "filesystem") -> bool:
        current_hash = file_hash(path)
        existing = get_item_by_path(self._conn, str(path))
        if existing and existing.file_hash == current_hash:
            return False

        stat = path.stat()
        description = self._describe(path, file_type)
        embedding = self._embedder.embed([description["description"]])[0]

        insert_item(self._conn, ItemRow(
            file_path=str(path),
            file_hash=current_hash,
            file_type=file_type,
            file_size=stat.st_size,
            modified_at=__import__("datetime").datetime.fromtimestamp(stat.st_mtime).isoformat(),
            ai_description=description["description"],
            categories=description["categories"],
            summary=description["summary"],
            embedding=embedding,
            source=source,
        ))
        return True

    def index_batch(self, files: list[tuple[Path, str]], source: str = "filesystem") -> int:
        count = 0
        for path, file_type in files:
            try:
                if self.index_file(path, file_type, source):
                    count += 1
            except Exception as e:
                print(f"Error indexing {path}: {e}")
        return count

    def _describe(self, path: Path, file_type: str) -> dict:
        if file_type == "photo":
            return describe_image(self._llm, path)
        elif file_type == "video":
            return describe_video(self._llm, path)
        elif file_type == "document":
            text = self._extract_pdf_text(path)
            return describe_document(self._llm, text)
        elif file_type == "markdown":
            text = path.read_text(errors="replace")
            return describe_markdown(self._llm, text)
        else:
            return {"description": f"{file_type} file", "categories": [], "summary": ""}

    def _extract_pdf_text(self, path: Path) -> str:
        try:
            with pdfplumber.open(path) as pdf:
                pages = []
                for page in pdf.pages[:10]:
                    text = page.extract_text()
                    if text:
                        pages.append(text)
                return "\n\n".join(pages)
        except Exception:
            return ""
