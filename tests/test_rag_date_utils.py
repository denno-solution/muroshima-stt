import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from services.rag.date_utils import parse_date_from_query  # noqa: E402

TODAY = date(2026, 7, 31)  # 金曜日


def _parse(q):
    return parse_date_from_query(q, today=TODAY)


class TestExplicitDates:
    def test_full_date_kanji(self):
        dr = _parse("2026年7月28日の作業")
        assert (dr.start, dr.end) == (date(2026, 7, 28), date(2026, 7, 28))
        assert dr.kind == "explicit"

    def test_full_date_slash(self):
        dr = _parse("2025/10/15の業務内容")
        assert dr.start == date(2025, 10, 15)

    def test_full_date_hyphen(self):
        dr = _parse("2025-10-15の業務内容を教えてください。")
        assert dr.start == date(2025, 10, 15)

    def test_year_month(self):
        dr = _parse("2026年6月の記録をまとめて")
        assert (dr.start, dr.end) == (date(2026, 6, 1), date(2026, 6, 30))

    def test_month_day_this_year(self):
        dr = _parse("7月28日の作業内容")
        assert dr.start == date(2026, 7, 28)

    def test_month_day_rolls_back_to_last_year(self):
        dr = _parse("10月15日の業務")
        assert dr.start == date(2025, 10, 15)

    def test_month_day_slash(self):
        dr = _parse("10/15の業務内容を教えてください")
        assert dr.start == date(2025, 10, 15)

    def test_month_only(self):
        dr = _parse("6月の成形記録")
        assert (dr.start, dr.end) == (date(2026, 6, 1), date(2026, 6, 30))

    def test_month_only_rolls_back(self):
        dr = _parse("12月の記録")
        assert (dr.start, dr.end) == (date(2025, 12, 1), date(2025, 12, 31))

    def test_invalid_day_falls_back_to_month(self):
        # 存在しない日(2/30)は年月の範囲にフォールバックする
        dr = _parse("2026年2月30日")
        assert (dr.start, dr.end) == (date(2026, 2, 1), date(2026, 2, 28))


class TestRelativeDates:
    def test_today(self):
        dr = _parse("今日の作業")
        assert (dr.start, dr.end) == (TODAY, TODAY)

    def test_yesterday(self):
        dr = _parse("昨日の録音内容")
        assert dr.start == date(2026, 7, 30)

    def test_day_before_yesterday(self):
        dr = _parse("一昨日の件")
        assert dr.start == date(2026, 7, 29)

    def test_last_week(self):
        dr = _parse("先週の記録")
        assert (dr.start, dr.end) == (date(2026, 7, 20), date(2026, 7, 26))

    def test_last_month(self):
        dr = _parse("先月の作業")
        assert (dr.start, dr.end) == (date(2026, 6, 1), date(2026, 6, 30))

    def test_last_year(self):
        dr = _parse("去年のトラブル")
        assert (dr.start, dr.end) == (date(2025, 1, 1), date(2025, 12, 31))

    def test_n_days_ago(self):
        dr = _parse("3日前の作業")
        assert dr.start == date(2026, 7, 28)

    def test_n_months_ago_uses_month_arithmetic(self):
        dr = _parse("3ヶ月前の作業")
        assert (dr.start, dr.end) == (date(2026, 4, 1), date(2026, 4, 30))

    def test_n_months_ago_across_year(self):
        dr = _parse("13ヶ月前の記録")
        assert (dr.start, dr.end) == (date(2025, 6, 1), date(2025, 6, 30))


class TestSpans:
    def test_kono_n_weeks(self):
        dr = _parse("この2週間の記録")
        assert (dr.start, dr.end) == (date(2026, 7, 17), TODAY)

    def test_recent_with_parenthesized_span(self):
        dr = _parse("最近（2週間）の成形の記録")
        assert (dr.start, dr.end) == (date(2026, 7, 17), TODAY)
        assert dr.kind == "explicit"

    def test_past_n_days(self):
        dr = _parse("過去10日間の作業")
        assert (dr.start, dr.end) == (date(2026, 7, 21), TODAY)


class TestRecency:
    def test_recent_keyword(self):
        dr = _parse("最近の業務記録をまとめてください")
        assert dr.kind == "recency"
        assert dr.end == TODAY
        assert (TODAY - dr.start).days == 30

    def test_no_date(self):
        assert _parse("ボイドの対策を教えて") is None
