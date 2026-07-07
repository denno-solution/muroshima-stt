import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402
from services.ceo_time_utils import (  # noqa: E402
    ceo_record_sort_key,
    format_created_at_for_web,
    format_recorded_at_for_web,
)


@contextmanager
def process_timezone(name: str):
    old_tz = os.environ.get("TZ")
    os.environ["TZ"] = name
    if hasattr(time, "tzset"):
        time.tzset()

    try:
        yield
    finally:
        if old_tz is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = old_tz
        if hasattr(time, "tzset"):
            time.tzset()


@pytest.fixture
def jst_timezone():
    with process_timezone("Asia/Tokyo"):
        yield


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


def test_web_recorded_at_display_uses_main_recording_format(jst_timezone):
    assert format_recorded_at_for_web("2026-07-07T11:02:28") == "2026-07-07T11:02:28"
    assert format_recorded_at_for_web("2026-07-07T02:02:00.000Z") == "2026-07-07T11:02:00"
    assert format_recorded_at_for_web("2026-07-07T11:02:00+09:00") == "2026-07-07T11:02:00"


def test_web_created_at_display_keeps_main_web_values_and_localizes_aware_values(jst_timezone):
    assert (
        format_created_at_for_web("2026-07-07 11:02:30.572861")
        == "2026-07-07 11:02:30.572861"
    )
    assert (
        format_created_at_for_web("2026-07-07T02:03:13.856666+00:00")
        == "2026-07-07 11:03:13.856666"
    )
    assert format_created_at_for_web("2026-07-07T02:03:13+00:00") == "2026-07-07 11:03:13"


def test_record_sort_key_does_not_depend_on_process_timezone():
    with process_timezone("UTC"):
        rows = [
            SimpleNamespace(id=8, recorded_at="2026-07-06T09:11:00.000Z", created_at=None),
            SimpleNamespace(id=7, recorded_at="2026-07-06T18:10:51", created_at=None),
        ]

        sorted_ids = [row.id for row in sorted(rows, key=ceo_record_sort_key)]

        assert sorted_ids == [8, 7]
        assert format_recorded_at_for_web("2026-07-06T09:11:00.000Z") == "2026-07-06T18:11:00"
