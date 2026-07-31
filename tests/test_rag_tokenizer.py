import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402

from services.rag.tokenizer import (  # noqa: E402
    fts_query_any,
    fts_query_exact,
    index_tokens,
    to_fts_text,
)


class TestIndexTokens:
    def test_cjk_bigrams(self):
        assert index_tokens("保圧") == ["保圧"]
        assert index_tokens("ジュラコン") == ["ジュ", "ュラ", "ラコ", "コン"]

    def test_latin_run_kept_whole(self):
        assert index_tokens("PA66-GF30") == ["pa66", "gf30"]

    def test_mixed_text(self):
        tokens = index_tokens("保圧を50に上げた")
        assert "保圧" in tokens
        assert "50" in tokens

    def test_nfkc_normalization(self):
        # 全角英数字・半角カナは正規化される
        assert index_tokens("ＰＡ６６") == ["pa66"]
        assert index_tokens("ﾎﾞｲﾄﾞ") == index_tokens("ボイド")

    def test_empty(self):
        assert index_tokens("") == []
        assert index_tokens("、。！") == []


class TestFtsQueryAny:
    def test_drops_hiragana_only_bigrams_when_content_exists(self):
        q = fts_query_any("ヒケという記録")
        assert '"ヒケ"' in q
        assert '"とい"' not in q  # ひらがなのみのバイグラムは除外

    def test_keeps_hiragana_for_hiragana_only_query(self):
        q = fts_query_any("せいけい")
        assert q is not None and '"せい"' in q

    def test_empty_returns_none(self):
        assert fts_query_any("。、") is None


class TestFtsQueryExact:
    def test_two_char_word(self):
        assert fts_query_exact("保圧") == '"保圧"'

    def test_phrase_of_bigrams(self):
        assert fts_query_exact("ジュラコン") == '"ジュ ュラ ラコ コン"'

    def test_single_char_prefix(self):
        assert fts_query_exact("巣") == '"巣"*'

    def test_multi_run_and(self):
        q = fts_query_exact("PA66材料")
        assert q == '"pa66" AND "材料"'


@pytest.fixture()
def fts_db(tmp_path):
    """libsqlのローカルファイルにFTS5テーブルを作り、実際のMATCH挙動を検証する。"""
    libsql = pytest.importorskip("libsql")
    conn = libsql.connect(str(tmp_path / "fts.db"))
    cur = conn.cursor()
    cur.execute("CREATE VIRTUAL TABLE fts USING fts5(tokens, tokenize='unicode61')")
    docs = {
        1: "保圧を50に上げたらショートが直った",
        2: "ジュラコンM944の成形条件を確認",
        3: "ボイドが出たのでゲートを拡大した",
        4: "今日は天気が良い",
    }
    for rowid, text in docs.items():
        cur.execute("INSERT INTO fts(rowid, tokens) VALUES (?, ?)", (rowid, to_fts_text(text)))
    conn.commit()
    yield cur


def _match_ids(cur, query):
    cur.execute("SELECT rowid FROM fts WHERE fts MATCH ? ORDER BY rowid", (query,))
    return [r[0] for r in cur.fetchall()]


class TestFtsIntegration:
    def test_exact_two_char_kanji(self, fts_db):
        assert _match_ids(fts_db, fts_query_exact("保圧")) == [1]

    def test_exact_katakana_word(self, fts_db):
        assert _match_ids(fts_db, fts_query_exact("ジュラコン")) == [2]

    def test_exact_substring_of_word(self, fts_db):
        # 部分文字列(ラコン)でも連接フレーズで一致する
        assert _match_ids(fts_db, fts_query_exact("ラコン")) == [2]

    def test_exact_latin_model_number(self, fts_db):
        assert _match_ids(fts_db, fts_query_exact("M944")) == [2]

    def test_no_false_positive(self, fts_db):
        assert _match_ids(fts_db, fts_query_exact("ヒケ")) == []

    def test_any_query_ranks_rare_term(self, fts_db):
        ids = _match_ids(fts_db, fts_query_any("ボイドの対策を教えて"))
        assert 3 in ids
