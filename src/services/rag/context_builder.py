"""検索ヒットから回答用コンテキストを組み立てる。

チャンク断片をそのまま渡すのではなく、録音単位にグループ化して
「短い録音は全文、長い録音はヒット周辺の連続ウィンドウ」を渡す。
文脈が保たれるため、要約系・履歴参照系の質問に強くなる。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

from sqlalchemy import text
from sqlalchemy.orm import Session

_PARENT_TABLES = {
    "audio": ("audio_transcriptions", "audio_transcription_chunks"),
    "ceo": ("ceo_transcriptions", "ceo_transcription_chunks"),
}


@dataclass
class ContextDoc:
    n: int  # プロンプト内の出典番号 [#n]
    source: str
    transcription_id: int
    title: str
    recorded_date: Optional[str]
    tags: Optional[str]
    text: str
    score: float
    hit_count: int
    is_full_text: bool
    truncated: bool

    def to_dict(self) -> Dict:
        return asdict(self)


def _join_with_overlap(a: str, b: str, max_overlap: int = 200) -> str:
    """チャンク化時のオーバーラップ(既定120文字)を検出して重複なしに連結。"""
    limit = min(len(a), len(b), max_overlap)
    for k in range(limit, 0, -1):
        if a.endswith(b[:k]):
            return a + b[k:]
    return a + b


def _merge_chunk_windows(
    ordered_chunks: List[Tuple[int, str]],
    hit_indexes: set,
    window: int,
) -> str:
    """ヒットindex±windowの連続領域を結合し、間隙は省略記号で示す。"""
    if not ordered_chunks:
        return ""
    keep: set = set()
    for idx in hit_indexes:
        for i in range(idx - window, idx + window + 1):
            keep.add(i)

    parts: List[str] = []
    current = ""
    prev_idx: Optional[int] = None
    for idx, chunk in ordered_chunks:
        if idx not in keep:
            continue
        if prev_idx is not None and idx == prev_idx + 1 and current:
            current = _join_with_overlap(current, chunk)
        else:
            if current:
                parts.append(current)
            current = chunk
        prev_idx = idx
    if current:
        parts.append(current)
    return "\n（…中略…）\n".join(parts)


def build_context_docs(
    db: Session,
    hits: List[Dict],
    *,
    max_docs: int,
    max_chars: int,
    whole_doc_threshold: int = 4000,
    neighbor_window: int = 1,
    order: str = "score",  # "score" | "date"
) -> List[ContextDoc]:
    """検索ヒット(チャンク単位)を録音単位のコンテキストに変換する。"""
    groups: Dict[Tuple[str, int], Dict] = {}
    for h in hits:
        key = (h["source"], int(h["transcription_id"]))
        g = groups.setdefault(
            key,
            {
                "source": h["source"],
                "transcription_id": int(h["transcription_id"]),
                "title": h.get("title") or "",
                "recorded_date": h.get("recorded_date"),
                "tags": h.get("tags"),
                "score": 0.0,
                "hit_indexes": set(),
            },
        )
        g["score"] = max(g["score"], float(h.get("score") or 0.0))
        if h.get("chunk_index") is not None:
            g["hit_indexes"].add(int(h["chunk_index"]))

    ordered = list(groups.values())
    if order == "date":
        ordered.sort(key=lambda g: (g.get("recorded_date") or "", g["transcription_id"]), reverse=True)
    else:
        ordered.sort(key=lambda g: g["score"], reverse=True)

    per_doc_cap = max(whole_doc_threshold, max_chars // max(1, max_docs))
    docs: List[ContextDoc] = []
    used_chars = 0

    for g in ordered:
        if len(docs) >= max_docs or used_chars >= max_chars:
            break
        parent_table, chunk_table = _PARENT_TABLES[g["source"]]
        row = db.execute(
            text(f"SELECT transcript FROM {parent_table} WHERE id = :id"),
            {"id": g["transcription_id"]},
        ).mappings().first()
        if not row or not row["transcript"]:
            continue
        transcript = row["transcript"]

        is_full = len(transcript) <= whole_doc_threshold
        if is_full:
            body = transcript
        else:
            chunk_rows = db.execute(
                text(
                    f"SELECT chunk_index, chunk_text FROM {chunk_table} "
                    f"WHERE transcription_id = :id ORDER BY chunk_index"
                ),
                {"id": g["transcription_id"]},
            ).mappings().all()
            ordered_chunks = [(int(r["chunk_index"]), r["chunk_text"] or "") for r in chunk_rows]
            if g["hit_indexes"] and ordered_chunks:
                body = _merge_chunk_windows(ordered_chunks, g["hit_indexes"], neighbor_window)
            else:
                body = transcript  # チャンク情報がない場合は全文(後段で切り詰め)

        truncated = False
        budget = min(per_doc_cap, max_chars - used_chars)
        if len(body) > budget:
            body = body[:budget] + "\n（文字数上限のため以下省略）"
            truncated = True

        docs.append(
            ContextDoc(
                n=len(docs) + 1,
                source=g["source"],
                transcription_id=g["transcription_id"],
                title=g["title"],
                recorded_date=g.get("recorded_date"),
                tags=g.get("tags"),
                text=body,
                score=round(g["score"], 4),
                hit_count=len(g["hit_indexes"]) or 1,
                is_full_text=is_full,
                truncated=truncated,
            )
        )
        used_chars += len(body)

    return docs
