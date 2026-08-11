"""タイムゾーン扱い(created_at書き込み・JSTの今日)のテスト。"""

import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402

from models import AudioTranscription, CeoTranscription, utcnow_naive  # noqa: E402
from services.rag import date_utils  # noqa: E402
from services.rag.date_utils import jst_today, parse_date_from_query  # noqa: E402
from services.rag.reconcile import JST, normalize_to_jst_date  # noqa: E402


class _FakeDatetime:
    """JSTでは翌日になっているUTC夕方の時刻に固定する。"""

    fixed = datetime(2026, 8, 10, 16, 30, tzinfo=timezone.utc)  # JST: 2026-08-11 01:30

    @classmethod
    def now(cls, tz=None):
        if tz is None:
            return cls.fixed.replace(tzinfo=None)
        return cls.fixed.astimezone(tz)


class TestUtcnowNaive:
    def test_returns_naive_utc(self):
        value = utcnow_naive()
        assert value.tzinfo is None
        delta = abs((value - datetime.now(timezone.utc).replace(tzinfo=None)).total_seconds())
        assert delta < 5.0

    def test_model_defaults_use_utcnow_naive(self):
        # bare datetime.now(サーバーローカル依存)に戻っていないことを担保する
        # (SQLAlchemyは引数なしのdefault関数をラップするため__wrapped__を見る)
        def default_fn(column):
            arg = column.default.arg
            return getattr(arg, "__wrapped__", arg)

        assert default_fn(AudioTranscription.__table__.c.created_at) is utcnow_naive
        assert default_fn(CeoTranscription.__table__.c.created_at) is utcnow_naive

    def test_written_value_reconciles_to_jst_date(self):
        # naive UTCとして書いた値は、日付判定(UTC解釈+9h)でJSTの日付になる
        value = datetime(2026, 8, 10, 16, 30)  # naive UTC
        assert normalize_to_jst_date(value) == "2026-08-11"


class TestJstToday:
    def test_jst_today_rolls_forward_at_utc_evening(self, monkeypatch):
        monkeypatch.setattr(date_utils, "datetime", _FakeDatetime)
        assert jst_today() == date(2026, 8, 11)

    def test_parse_date_fallback_uses_jst_today(self, monkeypatch):
        monkeypatch.setattr(date_utils, "datetime", _FakeDatetime)
        dr = parse_date_from_query("今日の作業")  # today指定なし
        assert (dr.start, dr.end) == (date(2026, 8, 11), date(2026, 8, 11))

    def test_jst_constant(self):
        assert JST.utcoffset(None).total_seconds() == 9 * 3600


class TestExplicitTodayStillWins:
    def test_passed_today_overrides_fallback(self, monkeypatch):
        monkeypatch.setattr(date_utils, "datetime", _FakeDatetime)
        dr = parse_date_from_query("昨日の記録", today=date(2026, 7, 31))
        assert dr.start == date(2026, 7, 30)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("2026-08-10 16:30:00.123456", "2026-08-11"),  # Web版naive UTC
        ("2026-08-10T16:30:00.906558100+00:00", "2026-08-11"),  # desktop RFC3339
    ],
)
def test_both_writer_formats_agree(raw, expected):
    assert normalize_to_jst_date(raw) == expected
