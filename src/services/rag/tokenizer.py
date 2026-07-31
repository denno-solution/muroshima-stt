"""FTS5用の日本語トークナイザ(文字バイグラム方式)。

形態素解析辞書に載らない現場用語・材料名・型番(保圧/ジュラコン/PA66-GF30等)でも
確実に索引・検索できるよう、CJK連続文字は文字バイグラム、英数字連続は1トークン
として扱う。索引時と検索時で同じ関数を使うことで一致を保証する。
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterator, List, Tuple

# CJK扱いする文字(ひらがな/カタカナ/長音/漢字/々〆ヶ)
_CJK_RE = re.compile(r"[ぁ-んァ-ヶーｦ-ﾟ一-龯々〆〤㐀-䶿]")
_LATIN_RE = re.compile(r"[a-z0-9]")
_HIRAGANA_RE = re.compile(r"^[ぁ-んー]+$")


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).lower()


def _runs(text: str) -> Iterator[Tuple[str, str]]:
    """正規化済みテキストを (種別, 連続文字列) に分割。種別: 'cjk' | 'latin'"""
    buf: List[str] = []
    kind = ""
    for ch in text:
        if _CJK_RE.match(ch):
            k = "cjk"
        elif _LATIN_RE.match(ch):
            k = "latin"
        else:
            k = ""
        if k != kind and buf:
            if kind:
                yield kind, "".join(buf)
            buf = []
        kind = k
        if k:
            buf.append(ch)
    if buf and kind:
        yield kind, "".join(buf)


def _cjk_bigrams(run: str) -> List[str]:
    if len(run) == 1:
        return [run]
    return [run[i : i + 2] for i in range(len(run) - 1)]


def index_tokens(text: str) -> List[str]:
    """索引用トークン列。FTS5には ' '.join(index_tokens(t)) を格納する。"""
    tokens: List[str] = []
    for kind, run in _runs(_normalize(text or "")):
        if kind == "latin":
            tokens.append(run)
        else:
            tokens.extend(_cjk_bigrams(run))
    return tokens


def to_fts_text(text: str) -> str:
    return " ".join(index_tokens(text))


def _quote(token: str) -> str:
    return '"' + token.replace('"', '""') + '"'


def fts_query_any(text: str) -> str | None:
    """ランキング用のOR結合クエリ。

    質問文全体からトークンを取り、BM25のidfで希少語(ヒケ/ボイド等)が上位に
    来るようにする。ひらがなのみのバイグラム(助詞・言い回し)は、他の
    トークンが存在する場合はノイズとして除外する。
    """
    tokens = index_tokens(text)
    if not tokens:
        return None
    content = [t for t in tokens if not _HIRAGANA_RE.match(t)]
    use = content if content else tokens
    # 重複除去(順序維持)
    seen: set[str] = set()
    uniq = [t for t in use if not (t in seen or seen.add(t))]
    if not uniq:
        return None
    return " OR ".join(_quote(t) for t in uniq)


def fts_query_exact(term: str) -> str | None:
    """完全一致(部分文字列)用のフレーズクエリ。

    CJK連続部はバイグラムの連接フレーズにすることで、元テキスト上の
    部分文字列一致と等価になる。複数の連続部はANDで結合。
    """
    parts: List[str] = []
    for kind, run in _runs(_normalize(term or "")):
        if kind == "latin":
            parts.append(_quote(run))
        elif len(run) == 1:
            # 1文字のCJKはバイグラム前方一致で拾う
            parts.append(_quote(run) + "*")
        else:
            parts.append('"' + " ".join(_cjk_bigrams(run)).replace('"', '""') + '"')
    if not parts:
        return None
    return " AND ".join(parts)
