from __future__ import annotations

from datetime import date, datetime, time
from typing import Any


def timestamp_seconds(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.timestamp()
    if isinstance(value, date):
        return datetime.combine(value, time.min).timestamp()

    raw = str(value).strip()
    if not raw:
        return None

    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def ceo_record_sort_key(record: Any) -> tuple[bool, float, int]:
    timestamp = timestamp_seconds(getattr(record, "recorded_at", None))
    if timestamp is None:
        timestamp = timestamp_seconds(getattr(record, "created_at", None))
    return (timestamp is None, -(timestamp or 0), -(getattr(record, "id", 0) or 0))
