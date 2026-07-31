import os
import sys
from datetime import date
from pathlib import Path

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from services.rag_service import RAGService  # noqa: E402

TODAY = date(2026, 7, 31)
HISTORY = [
    {"role": "user", "content": "最近の業務記録をまとめてください"},
    {"role": "assistant", "content": "以下のとおりです…"},
]


def _plan(query, **kw):
    return RAGService().plan_query(query, today=TODAY, **kw)


class TestBrowseDetection:
    def test_generic_summary_goes_browse(self):
        # 従来は「の状況」バイグラムが内容語扱いされsearchに倒れていた
        plan = _plan("最近の状況についてまとめて")
        assert plan.mode == "browse"
        assert plan.aggregate

    def test_seikei_summary_goes_browse(self):
        plan = _plan("最近の成形の記録を詳し目にまとめて 表形式で確認したい")
        assert plan.mode == "browse"

    def test_date_pinpoint_goes_browse(self):
        plan = _plan("昨日の作業内容を教えてください")
        assert plan.mode == "browse"

    def test_topic_with_date_stays_search(self):
        plan = _plan("最近の成形業務で不良に関係するような記録内容を調べ 表形式でまとめてください")
        assert plan.mode == "search"
        assert plan.aggregate
        assert plan.match_query and "不良" in plan.match_query

    def test_aggregate_without_date_goes_browse(self):
        plan = _plan("これまでの記録をまとめてください")
        assert plan.mode == "browse"
        assert plan.date_range is None


class TestSynonymsInPlan:
    def test_hike_query_carries_kanji_variant(self):
        plan = _plan("ヒケの対策")
        assert "引け" in plan.retrieval_text
        assert plan.match_query and "引け" in plan.match_query

    def test_instruction_bigrams_not_in_match_query(self):
        plan = _plan("最近で「ヒケ」という記録が多いものをピックアップしてください")
        assert plan.match_query and "ヒケ" in plan.match_query
        assert "ピッ" not in plan.match_query


class TestFollowupPlan:
    def test_format_request_with_history(self):
        plan = _plan("表形式でまとめることはできますか？", chat_history=HISTORY)
        assert plan.mode == "followup"

    def test_allow_followup_false_forces_search(self):
        plan = _plan(
            "表形式でまとめることはできますか？",
            chat_history=HISTORY,
            allow_followup=False,
        )
        assert plan.mode != "followup"

    def test_no_history_is_not_followup(self):
        plan = _plan("表形式でまとめることはできますか？")
        assert plan.mode != "followup"

    def test_date_followup_goes_browse(self):
        plan = _plan("一昨日はどうだった？", chat_history=HISTORY)
        assert plan.mode == "browse"
        assert plan.date_range.start == date(2026, 7, 29)

    def test_meta_question_reuses_previous(self):
        plan = _plan(
            "参照チャンクを確認したら去年の10月の内容のようです。最近と指示した理由を述べてください",
            chat_history=HISTORY,
        )
        assert plan.mode == "followup"
        assert plan.date_range is None  # 質問文中の「10月」で誤って絞り込まない

    def test_manual_range_respected(self):
        plan = _plan(
            "作業のまとめ",
            manual_date_range=(date(2026, 7, 1), date(2026, 7, 15)),
            chat_history=HISTORY,
        )
        assert plan.mode == "browse"
        assert plan.date_range.kind == "manual"
