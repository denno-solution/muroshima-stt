from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple


def parse_date_from_query(query: str) -> Optional[Tuple[date, date]]:
    """ユーザークエリから日付範囲を抽出する。"""
    today = date.today()
    current_year = today.year

    if "今日" in query:
        return (today, today)
    if "昨日" in query:
        y = today - timedelta(days=1)
        return (y, y)
    if "一昨日" in query or "おととい" in query:
        d = today - timedelta(days=2)
        return (d, d)
    if "今週" in query:
        start = today - timedelta(days=today.weekday())
        end = start + timedelta(days=6)
        return (start, min(end, today))
    if "先週" in query:
        start = today - timedelta(days=today.weekday() + 7)
        end = start + timedelta(days=6)
        return (start, end)
    if "今月" in query:
        start = today.replace(day=1)
        return (start, today)
    if "先月" in query:
        first = today.replace(day=1)
        last_month_end = first - timedelta(days=1)
        last_month_start = last_month_end.replace(day=1)
        return (last_month_start, last_month_end)

    days_ago = re.search(r"(\d+)\s*日前", query)
    if days_ago:
        n = int(days_ago.group(1))
        target = today - timedelta(days=n)
        return (target, target)

    weeks_ago = re.search(r"(\d+)\s*週間?前", query)
    if weeks_ago:
        n = int(weeks_ago.group(1))
        target_end = today - timedelta(weeks=n)
        target_start = target_end - timedelta(days=6)
        return (target_start, target_end)

    months_ago = re.search(r"(\d+)\s*[ヶか]?月前", query)
    if months_ago:
        n = int(months_ago.group(1))
        target = today - timedelta(days=30 * n)
        month_start = target.replace(day=1)
        if target.month == 12:
            month_end = target.replace(day=31)
        else:
            month_end = target.replace(month=target.month + 1, day=1) - timedelta(days=1)
        return (month_start, month_end)

    full_date = re.search(r"(\d{4})[年/\-](\d{1,2})[月/\-](\d{1,2})日?", query)
    if full_date:
        try:
            year = int(full_date.group(1))
            month = int(full_date.group(2))
            day = int(full_date.group(3))
            target = date(year, month, day)
            return (target, target)
        except ValueError:
            pass

    month_day = re.search(r"(\d{1,2})[月/\-](\d{1,2})日?", query)
    if month_day:
        try:
            month = int(month_day.group(1))
            day = int(month_day.group(2))
            target = date(current_year, month, day)
            if target > today:
                target = date(current_year - 1, month, day)
            return (target, target)
        except ValueError:
            pass

    return None


def highlight_date_in_query(query: str) -> str:
    """クエリ内の日付表現をStreamlit用にハイライトする。"""
    result = query

    def wrap(match: re.Match) -> str:
        return f":orange[{match.group(0)}]"

    relative_patterns = [
        r"今日",
        r"昨日",
        r"一昨日",
        r"おととい",
        r"今週",
        r"先週",
        r"今月",
        r"先月",
        r"\d+\s*日前",
        r"\d+\s*週間?前",
        r"\d+\s*[ヶか]?月前",
    ]

    for pattern in relative_patterns:
        result = re.sub(pattern, wrap, result)

    result = re.sub(r"(\d{4})[年/\-](\d{1,2})[月/\-](\d{1,2})日?", wrap, result)
    result = re.sub(r"(\d{1,2})[月/\-](\d{1,2})日?", wrap, result)

    return result


def filter_matches_by_date(matches: List[Dict], date_range: Tuple[date, date]) -> List[Dict]:
    """検索結果を指定日付でフィルタリングする。"""
    start_date, end_date = date_range
    filtered = []
    for m in matches:
        recorded_at = m.get("recorded_at")
        if not recorded_at:
            continue
        if isinstance(recorded_at, str):
            try:
                recorded_date = datetime.fromisoformat(recorded_at.replace("Z", "+00:00")).date()
            except (ValueError, TypeError):
                try:
                    recorded_date = datetime.strptime(recorded_at[:10], "%Y-%m-%d").date()
                except (ValueError, TypeError):
                    continue
        elif isinstance(recorded_at, datetime):
            recorded_date = recorded_at.date()
        elif isinstance(recorded_at, date):
            recorded_date = recorded_at
        else:
            continue

        if start_date <= recorded_date <= end_date:
            filtered.append(m)
    return filtered
