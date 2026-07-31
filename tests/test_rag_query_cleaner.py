import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from services.rag.query_cleaner import (  # noqa: E402
    build_match_query,
    expand_synonyms,
    fts_query_content,
    has_content_keywords,
    is_followup,
    strip_instructions,
    wants_aggregate,
)


class TestStripInstructions:
    def test_removes_format_phrases(self):
        out = strip_instructions("最近の記録を表形式でまとめてください")
        assert "表形式" not in out
        assert "まとめて" not in out
        assert "最近の記録" in out

    def test_removes_pickup(self):
        out = strip_instructions("「ヒケ」という記録が多いものをピックアップしてください")
        assert "ピックアップ" not in out
        assert "ヒケ" in out


class TestHasContentKeywords:
    def test_boundary_bigram_not_counted_as_content(self):
        # 「の状況」のような語境界をまたぐバイグラムを内容語と誤認しない
        assert not has_content_keywords("の状況についてまとめて")

    def test_generic_summary_question_has_no_content(self):
        # 実質問#4: 日付表現除去後の残り
        assert not has_content_keywords("の成形の記録を詳し目にまとめて 表形式で確認したい")

    def test_domain_word_is_content(self):
        assert has_content_keywords("不良に関係するような記録")

    def test_katakana_term_is_content(self):
        assert has_content_keywords("ヒケ")

    def test_empty(self):
        assert not has_content_keywords("")


class TestWantsAggregate:
    def test_matome(self):
        assert wants_aggregate("最近の記録をまとめてください")

    def test_pickup(self):
        assert wants_aggregate("多いものをピックアップして")

    def test_plain_question(self):
        assert not wants_aggregate("ボイドの対策は？")


class TestIsFollowup:
    def test_format_change_request(self):
        assert is_followup("表形式でまとめることはできますか？", has_history=True, has_date=False)

    def test_more_detail_request(self):
        assert is_followup("具体的な内容で教えてください", has_history=True, has_date=False)

    def test_meta_question_about_chunks(self):
        assert is_followup(
            "参照チャンクを確認したら去年の10月の内容のようです。理由を述べてください",
            has_history=True,
            has_date=False,
        )

    def test_meta_question_about_answer(self):
        assert is_followup(
            "今の内容は最初に提示された内容に紐づいていますか？",
            has_history=True,
            has_date=False,
        )

    def test_date_question_is_not_followup(self):
        # 「昨日は？」は追問でも期間ブラウズで答えるべき
        assert not is_followup("昨日はどうだった？", has_history=True, has_date=True)

    def test_content_question_is_not_followup(self):
        assert not is_followup("ボイドの対策は？", has_history=True, has_date=False)

    def test_no_history(self):
        assert not is_followup("表形式でまとめて", has_history=False, has_date=False)


class TestSynonyms:
    def test_hike_expands_to_kanji(self):
        out = expand_synonyms("ヒケの対策")
        assert "引け" in out and "ひけ" in out

    def test_already_present_not_duplicated(self):
        out = expand_synonyms("ヒケ(引け)の対策")
        assert "引け" not in out

    def test_no_match(self):
        assert expand_synonyms("金型の温度調整") == []


class TestFtsQueryContent:
    def test_keeps_content_drops_particles(self):
        q = fts_query_content("成形 不良")
        assert '"成形"' in q and '"不良"' in q

    def test_generic_only_returns_none(self):
        assert fts_query_content("の記録について") is None

    def test_build_match_query_with_synonyms(self):
        clean = strip_instructions("「ヒケ」という記録が多いものをピックアップしてください")
        q = build_match_query(clean, ["引け", "ひけ"])
        assert "ヒケ" in q
        assert "引け" in q
        assert "ピッ" not in q  # 指示語のバイグラムが混入しない

    def test_build_match_query_synonyms_only(self):
        q = build_match_query("", ["引け"])
        assert "引け" in q
