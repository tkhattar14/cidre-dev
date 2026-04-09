from __future__ import annotations
import hashlib
import fnmatch
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

FILE_TYPES: dict[str, str] = {
    ".jpg": "photo", ".jpeg": "photo", ".png": "photo",
    ".heic": "photo", ".heif": "photo", ".webp": "photo",
    ".tiff": "photo", ".tif": "photo", ".bmp": "photo",
    ".mp4": "video", ".mov": "video", ".m4v": "video",
    ".avi": "video", ".mkv": "video",
    ".pdf": "document",
    ".md": "markdown", ".markdown": "markdown",
}


@dataclass
class ScannedFile:
    path: Path
    file_type: str
    file_hash: str
    file_size: int
    modified_at: str


def classify_file(path: Path) -> str | None:
    return FILE_TYPES.get(path.suffix.lower())


def should_exclude(path: Path, patterns: list[str]) -> bool:
    for pattern in patterns:
        if fnmatch.fnmatch(path.name, pattern):
            return True
        for part in path.parts:
            if part == pattern:
                return True
    return False


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_directory(
    directory: Path,
    exclude_patterns: list[str] | None = None,
) -> list[ScannedFile]:
    exclude_patterns = exclude_patterns or []
    results: list[ScannedFile] = []

    for path in directory.rglob("*"):
        if not path.is_file():
            continue
        if should_exclude(path.relative_to(directory), exclude_patterns):
            continue
        ftype = classify_file(path)
        if ftype is None:
            continue

        stat = path.stat()
        results.append(ScannedFile(
            path=path,
            file_type=ftype,
            file_hash=file_hash(path),
            file_size=stat.st_size,
            modified_at=datetime.fromtimestamp(stat.st_mtime).isoformat(),
        ))

    return results
