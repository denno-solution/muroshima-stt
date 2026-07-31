"""質問文の解釈(検索用クリーニング・意図判定・同義語展開)。

検索計画(rag_service.plan_query)の判断材料をすべて純関数として提供する。
prod実データでの評価(2026-07)に基づく:
- 指示語(「表形式で」「ピックアップして」等)のバイグラムはBM25を汚染するため
  FTSクエリから除外する(実測でnDCG@6 0.47→0.63)
- STTの表記ゆれ(ヒケ→「引け」等)は埋め込みでも橋渡しできないため辞書で展開する
- browse判定は語境界をまたぐバイグラム(「の状況」等)で誤って検索モードに
  倒れていたため、単語レベルで内容語の有無を判定する
"""

from __future__ import annotations

import re
from typing import List, Optional

from services.rag.tokenizer import index_tokens

_HIRAGANA_ONLY = re.compile(r"^[ぁ-んー]+$")

# 「◯◯をまとめて」等で実質的な検索の手がかりにならない一般語。
# browse判定(単語レベル)とFTSクエリのノイズ除去の両方で使う。
_GENERIC_WORDS = [
    "業務", "記録", "内容", "作業", "状況", "報告", "一覧", "詳細",
    "確認", "データ", "録音", "質問", "回答", "情報", "様子", "結果",
]
_GENERIC_BIGRAMS = {t for w in _GENERIC_WORDS for t in index_tokens(w)}

# browse判定専用の追加一般語。この工場では「成形」はほぼ全録音に含まれ
# 絞り込みに寄与しないため、期間要約の判定上は一般語として扱う。
# (FTSクエリからは除外しない: 「成形 不良」等の複合では有効なため)
_BROWSE_EXTRA_GENERIC = ["成形", "成型", "整形"]

# 回答の形式・粒度に関する指示語句。検索文からは除去する。
# 長い語句を先にマッチさせるため、長さ降順で結合する。
_INSTRUCTION_PHRASES = [
    "分かりやすい", "わかりやすい", "詳し目に", "詳しめに", "詳しく", "簡潔に",
    "抽象化して", "抽象的な", "具体的な", "具体的に",
    "表形式で", "報告書のように", "報告書形式で", "一覧で", "一覧化",
    "リストアップ", "リストで", "箇条書きで",
    "ピックアップしてください", "ピックアップして", "ピックアップ",
    "まとめてください", "まとめて", "まとめると", "要約してください", "要約して",
    "教えてください", "教えてほしい", "教えて",
    "整理してください", "整理して", "洗い出してください", "洗い出して",
    "挙げてください", "挙げて", "確認したい", "知りたい", "見たい",
    "比較して", "調べてください", "調べて", "調べ",
    "してください", "してほしい", "ください", "下さい",
    "お願いします", "お願いできますか", "できますか", "ですか", "ましたか", "ますか",
    "に関係するような", "に相当しそうなもの", "という記録が多いもの",
    "について話している録音", "について",
]
_INSTRUCTION_RE = re.compile(
    "|".join(re.escape(p) for p in sorted(_INSTRUCTION_PHRASES, key=len, reverse=True))
)

# 期間内を広く読んでほしい集約系の意図(件数上限を引き上げる判断に使う)
_AGGREGATE_RE = re.compile(
    r"まとめ|一覧|リスト|ピックアップ|要約|整理|傾向|報告書|表形式|洗い出|挙げ|比較|多い"
)

# 直前のやり取りを指すメタ表現(前回の参照録音を再利用する判断に使う)
_FOLLOWUP_META_RE = re.compile(
    r"さっき|先ほど|今の(?:内容|回答)|最初の質問|前の(?:回答|質問)|チャンク|"
    r"この(?:回答|スレッド)|その回答|回答のこと|紐づ|(?:もう少し|もっと)(?!前)"
)

# STT表記ゆれの同義語グループ(prodコーパス走査で実在を確認したもののみ)。
# グループ内のいずれかが質問に含まれたら、残りを検索語に加える。
_SYNONYM_GROUPS: List[List[str]] = [
    ["ヒケ", "引け", "ひけ"],
    ["ソリ", "反り", "そり"],
    ["ヤケ", "焼け", "焦げ"],
    ["バリ", "ばり"],
    ["ショート", "充填不足"],
    ["サックバック", "サクバック", "サックバッグ", "タックバック"],
    ["ウェルド", "ウエルド"],
    ["エコログ", "エコロム", "エコロゴ", "エポログ"],
    ["リハビリテック", "リハビテック"],
    ["ホットランナー", "フットランナー"],
]


def strip_instructions(text: str) -> str:
    """回答形式の指示語句を除去する(埋め込み・FTSの前処理)。"""
    return _INSTRUCTION_RE.sub(" ", text or "")


def expand_synonyms(query: str) -> List[str]:
    """質問に含まれる用語の表記ゆれを返す(質問に既出のものは除く)。"""
    q = query or ""
    out: List[str] = []
    for group in _SYNONYM_GROUPS:
        if any(term in q for term in group):
            out.extend(t for t in group if t not in q and t not in out)
    return out


def has_content_keywords(text: str) -> bool:
    """指示語・一般語・日付表現除去後に、検索の手がかりになる内容語が残るか。

    バイグラムが語境界をまたぐ問題(「の状況」→内容語扱い)を避けるため、
    一般語は単語単位で文字列から除去してからトークン化する。
    """
    residual = strip_instructions(text)
    for w in _GENERIC_WORDS + _BROWSE_EXTRA_GENERIC:
        residual = residual.replace(w, " ")
    for token in index_tokens(residual):
        if not _HIRAGANA_ONLY.match(token):
            return True
    return False


def wants_aggregate(query: str) -> bool:
    """「まとめて」「一覧」等、期間内を広く参照してほしい意図か。"""
    return bool(_AGGREGATE_RE.search(query or ""))


def is_followup(query: str, has_history: bool, has_date: bool) -> bool:
    """直前のやり取りの続き(形式変更・深掘り・メタ質問)か。

    該当する場合は新規検索ではなく前回の参照録音を再利用する。
    """
    if not has_history:
        return False
    if _FOLLOWUP_META_RE.search(query or ""):
        return True
    # 日付も内容語もない質問(「表形式でまとめて」「具体的に」等)は形式変更の追問
    return not has_date and not has_content_keywords(query)


def _quote(token: str) -> str:
    return '"' + token.replace('"', '""') + '"'


def fts_query_content(text: str) -> Optional[str]:
    """内容語のみのFTS5 ORクエリ。

    一般語は単語単位で除去してからトークン化する(「の記録に」→「の記/録に」の
    ような語境界バイグラムを残さないため)。さらにひらがなのみのバイグラム
    (助詞・言い回し)を除き、BM25のランキングを内容語に集中させる。
    """
    residual = text or ""
    for w in _GENERIC_WORDS:
        residual = residual.replace(w, " ")
    tokens = index_tokens(residual)
    content = [
        t for t in tokens
        if not _HIRAGANA_ONLY.match(t) and t not in _GENERIC_BIGRAMS
    ]
    seen: set = set()
    uniq = [t for t in content if not (t in seen or seen.add(t))]
    if not uniq:
        return None
    return " OR ".join(_quote(t) for t in uniq)


def build_match_query(clean_text: str, synonyms: List[str]) -> Optional[str]:
    """内容語ORクエリに同義語の完全一致フレーズを合流させたFTS5クエリ。"""
    from services.rag.tokenizer import fts_query_exact

    parts: List[str] = []
    base = fts_query_content(clean_text)
    if base:
        parts.append(base)
    for term in synonyms:
        phrase = fts_query_exact(term)
        if phrase:
            parts.append(phrase)
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return " OR ".join(f"({p})" for p in parts)
