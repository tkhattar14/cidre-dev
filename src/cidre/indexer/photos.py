from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import osxphotos


@dataclass
class PhotoInfo:
    uuid: str
    original_path: str
    date: str
    title: str | None


def list_apple_photos(limit: int | None = None) -> list[PhotoInfo]:
    db = osxphotos.PhotosDB()
    photos = db.photos()

    results = []
    for photo in photos:
        if photo.ismissing:
            continue
        if photo.path is None:
            continue

        info = PhotoInfo(
            uuid=photo.uuid,
            original_path=photo.path,
            date=photo.date.isoformat() if photo.date else "",
            title=photo.title,
        )
        results.append(info)

        if limit and len(results) >= limit:
            break

    return results
