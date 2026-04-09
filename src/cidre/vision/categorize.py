from __future__ import annotations
import re

CATEGORY_PROMPT_SUFFIX = "Categories: A comma-separated list of categories."


def parse_categories(raw: str) -> list[str]:
    if not raw or raw.lower() in ("none", "n/a", "[]"):
        return []
    cleaned = raw.strip().strip("[]")
    cats = [c.strip().lower() for c in re.split(r"[,;]", cleaned)]
    return [c for c in cats if c and c not in ("none", "n/a")]
