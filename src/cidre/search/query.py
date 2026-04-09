from __future__ import annotations
import re
from dataclasses import dataclass
from datetime import datetime


@dataclass
class QueryIntent:
    text: str
    file_type: str | None = None
    date_start: datetime | None = None
    date_end: datetime | None = None
    category: str | None = None


def parse_query(raw: str) -> QueryIntent:
    text = raw.strip()
    file_type = None
    date_start = None
    date_end = None

    # Detect type hints
    photo_patterns = [r"^photos?\s+of\s+", r"^images?\s+of\s+", r"^pictures?\s+of\s+"]
    for pattern in photo_patterns:
        match = re.match(pattern, text, re.IGNORECASE)
        if match:
            file_type = "photo"
            text = text[match.end():]
            break

    doc_patterns = [r"^documents?\s+about\s+", r"^docs?\s+about\s+"]
    for pattern in doc_patterns:
        match = re.match(pattern, text, re.IGNORECASE)
        if match:
            file_type = "document"
            text = text[match.end():]
            break

    # Detect "last year" BEFORE year pattern (more specific first)
    last_year_match = re.search(r"\bfrom\s+last\s+year\b", text, re.IGNORECASE)
    if last_year_match:
        year = datetime.now().year - 1
        date_start = datetime(year, 1, 1)
        date_end = datetime(year, 12, 31, 23, 59, 59)
        text = text[:last_year_match.start()].strip()
    else:
        # Detect Q1-Q4: "from Q1 2025"
        quarter_match = re.search(r"\bfrom\s+Q([1-4])\s+(\d{4})\b", text, re.IGNORECASE)
        if quarter_match:
            q = int(quarter_match.group(1))
            year = int(quarter_match.group(2))
            start_month = (q - 1) * 3 + 1
            end_month = start_month + 2
            date_start = datetime(year, start_month, 1)
            if end_month == 12:
                date_end = datetime(year, 12, 31, 23, 59, 59)
            else:
                date_end = datetime(year, end_month + 1, 1)
            text = text[:quarter_match.start()].strip()
        else:
            # Detect "this month"
            this_month_match = re.search(r"\bfrom\s+this\s+month\b", text, re.IGNORECASE)
            if this_month_match:
                now = datetime.now()
                date_start = datetime(now.year, now.month, 1)
                date_end = now
                text = text[:this_month_match.start()].strip()
            else:
                # Detect year: "from 2025"
                year_match = re.search(r"\bfrom\s+(\d{4})\b", text, re.IGNORECASE)
                if year_match:
                    year = int(year_match.group(1))
                    date_start = datetime(year, 1, 1)
                    date_end = datetime(year, 12, 31, 23, 59, 59)
                    text = text[:year_match.start()].strip()

    return QueryIntent(
        text=text,
        file_type=file_type,
        date_start=date_start,
        date_end=date_end,
    )
