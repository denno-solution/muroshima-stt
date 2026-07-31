"""クエリからの日付範囲抽出。

検索はSQLの WHERE recorded_date BETWEEN で行うため、ここでは
「クエリ文字列 → DateRange」の変換のみを担当する。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

# 「最近」の既定ウィンドウ(日)
RECENCY_DAYS = 30


@dataclass(frozen=True)
class DateRange:
    start: date
    end: date
    kind: str  # "explicit"(明示的な日付) | "recency"(「最近」等の曖昧表現)
    matched_text: str  # クエリ中でマッチした文字列(UI表示・キーワード除去用)


def _month_range(year: int, month: int) -> tuple[date, date]:
    start = date(year, month, 1)
    if month == 12:
        end = date(year, 12, 31)
    else:
        end = date(year, month + 1, 1) - timedelta(days=1)
    return start, end


def _shift_months(base: date, months_back: int) -> tuple[int, int]:
    """monthsBackヶ月前の(年, 月)を正確に求める。"""
    idx = base.year * 12 + (base.month - 1) - months_back
    return idx // 12, idx % 12 + 1


def parse_date_from_query(query: str, today: Optional[date] = None) -> Optional[DateRange]:
    """クエリから日付範囲を抽出する。より具体的な表現を優先する。"""
    today = today or date.today()
    q = query or ""

    # --- 完全な年月日 (2026年7月28日 / 2026/7/28 / 2026-07-28) ---
    m = re.search(r"(\d{4})[年/\-](\d{1,2})[月/\-](\d{1,2})日?", q)
    if m:
        try:
            d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            return DateRange(d, d, "explicit", m.group(0))
        except ValueError:
            pass

    # --- 年月 (2026年7月 / 2026/7) ---
    m = re.search(r"(\d{4})[年/\-](\d{1,2})月?(?![\d/\-])", q)
    if m:
        year, month = int(m.group(1)), int(m.group(2))
        if 1 <= month <= 12:
            s, e = _month_range(year, month)
            return DateRange(s, e, "explicit", m.group(0))

    # --- 月日 (7月28日 / 7/28)。未来なら前年扱い ---
    m = re.search(r"(\d{1,2})[月/\-](\d{1,2})日?", q)
    if m:
        try:
            month, day = int(m.group(1)), int(m.group(2))
            d = date(today.year, month, day)
            if d > today:
                d = date(today.year - 1, month, day)
            return DateRange(d, d, "explicit", m.group(0))
        except ValueError:
            pass

    # --- 月のみ (7月)。未来の月なら前年扱い ---
    m = re.search(r"(?<![\d/\-])(\d{1,2})月(?![\d/\-])", q)
    if m:
        month = int(m.group(1))
        if 1 <= month <= 12:
            year = today.year if month <= today.month else today.year - 1
            s, e = _month_range(year, month)
            return DateRange(s, e, "explicit", m.group(0))

    # --- 相対表現(固定語) ---
    fixed: list[tuple[str, tuple[date, date]]] = []
    if "今日" in q:
        fixed.append(("今日", (today, today)))
    if "一昨日" in q or "おととい" in q:
        d = today - timedelta(days=2)
        fixed.append(("一昨日" if "一昨日" in q else "おととい", (d, d)))
    elif "昨日" in q:
        d = today - timedelta(days=1)
        fixed.append(("昨日", (d, d)))
    if "今週" in q:
        start = today - timedelta(days=today.weekday())
        fixed.append(("今週", (start, today)))
    if "先週" in q:
        start = today - timedelta(days=today.weekday() + 7)
        fixed.append(("先週", (start, start + timedelta(days=6))))
    if "今月" in q:
        fixed.append(("今月", (today.replace(day=1), today)))
    if "先月" in q:
        y, mo = _shift_months(today, 1)
        fixed.append(("先月", _month_range(y, mo)))
    if "去年" in q or "昨年" in q:
        y = today.year - 1
        fixed.append(("去年" if "去年" in q else "昨年", (date(y, 1, 1), date(y, 12, 31))))
    elif "今年" in q:
        fixed.append(("今年", (date(today.year, 1, 1), today)))
    if fixed:
        text, (s, e) = fixed[0]
        return DateRange(s, e, "explicit", text)

    # --- N日前 / N週間前 / Nヶ月前 ---
    m = re.search(r"(\d+)\s*日前", q)
    if m:
        d = today - timedelta(days=int(m.group(1)))
        return DateRange(d, d, "explicit", m.group(0))
    m = re.search(r"(\d+)\s*週間?前", q)
    if m:
        end = today - timedelta(weeks=int(m.group(1)))
        return DateRange(end - timedelta(days=6), end, "explicit", m.group(0))
    m = re.search(r"(\d+)\s*[ヶかカケ箇]?月前", q)
    if m:
        y, mo = _shift_months(today, int(m.group(1)))
        s, e = _month_range(y, mo)
        return DateRange(s, e, "explicit", m.group(0))

    # --- 期間幅 (この2週間 / 過去10日間 / ここ3ヶ月) ---
    m = re.search(r"(?:この|ここ|過去|直近|最近)\s*[（(]?\s*(\d+)\s*(日|週間|[ヶかカケ箇]?月)\s*[)）]?", q)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        if unit.startswith("日"):
            days = n
        elif unit.startswith("週"):
            days = n * 7
        else:
            days = n * 30
        return DateRange(today - timedelta(days=days), today, "explicit", m.group(0))

    # --- 曖昧な「最近」系 ---
    m = re.search(r"最近|直近|近頃|ここのところ", q)
    if m:
        return DateRange(today - timedelta(days=RECENCY_DAYS), today, "recency", m.group(0))

    return None


def highlight_date_in_query(query: str, today: Optional[date] = None) -> str:
    """クエリ内の日付表現をStreamlit表示用にハイライトする。"""
    dr = parse_date_from_query(query, today)
    if not dr or not dr.matched_text:
        return query
    return query.replace(dr.matched_text, f":orange[{dr.matched_text}]", 1)
