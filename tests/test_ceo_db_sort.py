import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402
from services.ceo_time_utils import ceo_record_sort_key  # noqa: E402


@pytest.fixture
def jst_timezone():
    old_tz = os.environ.get("TZ")
    os.environ["TZ"] = "Asia/Tokyo"
    if hasattr(time, "tzset"):
        time.tzset()

    yield

    if old_tz is None:
        os.environ.pop("TZ", None)
    else:
        os.environ["TZ"] = old_tz
    if hasattr(time, "tzset"):
        time.tzset()


def test_record_sort_key_orders_mixed_timestamp_formats_by_display_time(jst_timezone):
    rows = [
        SimpleNamespace(id=7, recorded_at="2026-07-06T18:10:51", created_at=None),
        SimpleNamespace(id=6, recorded_at="2026-07-06T18:09:31", created_at=None),
        SimpleNamespace(id=5, recorded_at="2026-07-06T17:46:37", created_at=None),
        SimpleNamespace(id=2, recorded_at="2026-07-06T11:20:00+09:00", created_at=None),
        SimpleNamespace(id=8, recorded_at="2026-07-06T09:11:00.000Z", created_at=None),
        SimpleNamespace(id=1, recorded_at="2026-07-06T09:00:00+09:00", created_at=None),
        SimpleNamespace(id=4, recorded_at="2026-07-06T07:48:00.000Z", created_at=None),
        SimpleNamespace(id=3, recorded_at="2026-07-05T17:40:00+09:00", created_at=None),
    ]

    sorted_ids = [row.id for row in sorted(rows, key=ceo_record_sort_key)]

    assert sorted_ids == [8, 7, 6, 5, 4, 2, 1, 3]


def test_record_sort_key_falls_back_to_created_at_and_id(jst_timezone):
    rows = [
        SimpleNamespace(id=1, recorded_at="", created_at="2026-07-06 12:00:00.000000"),
        SimpleNamespace(id=3, recorded_at=None, created_at="2026-07-06 12:00:00.000000"),
        SimpleNamespace(id=2, recorded_at="not-a-date", created_at="2026-07-06 13:00:00.000000"),
        SimpleNamespace(id=4, recorded_at="not-a-date", created_at=None),
    ]

    sorted_ids = [row.id for row in sorted(rows, key=ceo_record_sort_key)]

    assert sorted_ids == [2, 3, 1, 4]
