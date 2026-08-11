"""検索ヒットから回答用コンテキストを組み立てる。

チャンク断片をそのまま渡すのではなく、録音単位にグループ化して
「短い録音は全文、長い録音はヒット周辺の連続ウィンドウ」を渡す。
文脈が保たれるため、要約系・履歴参照系の質問に強くなる。

チャンクに録音内時刻(start_sec/end_sec)がある録音では、本文の各チャンクに
[MM:SS〜MM:SS] 形式の時刻ラベルを付けて渡す(発言時刻を回答で引用するため)。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

from sqlalchemy import text
from sqlalchemy.orm import Session

_PARENT_TABLES = {
    "audio": ("audio_transcriptions", "audio_transcription_chunks"),
    "ceo": ("ceo_transcriptions", "ceo_transcription_chunks"),
    # 業務記録(tags='業務記録')はceoと同じテーブルに保存される(検索時にタグで分岐済み)
    "work": ("ceo_transcriptions", "ceo_transcription_chunks"),
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
    # 本文中の [MM:SS〜MM:SS] の基準。"original"=元音声基準 /
    # "vad"=無音カット(VAD)後基準(元音声の再生位置とズレの可能性) / None=時刻なし
    time_basis: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class _ChunkRow:
    index: int
    text: str
    start_sec: Optional[float] = None
    end_sec: Optional[float] = None
    time_basis: Optional[str] = None

    @property
    def has_time(self) -> bool:
        return self.start_sec is not None and self.end_sec is not None


def _overlap_len(a: str, b: str, max_overlap: int = 200) -> int:
    """aの末尾とbの先頭の重複文字数(チャンク化時のオーバーラップ)を検出。"""
    limit = min(len(a), len(b), max_overlap)
    for k in range(limit, 0, -1):
        if a.endswith(b[:k]):
            return k
    return 0


def _join_with_overlap(a: str, b: str, max_overlap: int = 200) -> str:
    """チャンク化時のオーバーラップ(既定120文字)を検出して重複なしに連結。"""
    return a + b[_overlap_len(a, b, max_overlap):]


def _format_mmss(seconds: float) -> str:
    """録音開始からの経過秒を MM:SS 形式にする(60分以上は分が2桁を超える)。"""
    total = max(0, int(round(seconds)))
    return f"{total // 60:02d}:{total % 60:02d}"


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


def _merge_timed_chunk_windows(
    chunk_rows: Sequence[_ChunkRow],
    keep_indexes: set,
) -> str:
    """時刻付きチャンクを [MM:SS〜MM:SS] ラベル付きの行として結合する。

    連続チャンクはオーバーラップ(前チャンク末尾との重複)を除去し、
    除去分だけラベルの開始時刻を線形補間で進める。間隙は省略記号で示す。
    時刻の無いチャンクはラベルなしの行になる。
    """
    parts: List[str] = []
    lines: List[str] = []
    prev_idx: Optional[int] = None
    prev_text: Optional[str] = None
    for row in chunk_rows:
        if row.index not in keep_indexes:
            continue
        if prev_idx is not None and row.index == prev_idx + 1 and prev_text is not None:
            k = _overlap_len(prev_text, row.text)
        else:
            if lines:
                parts.append("\n".join(lines))
                lines = []
            k = 0
        prev_idx, prev_text = row.index, row.text
        shown = row.text[k:]
        if not shown:
            continue
        if row.has_time:
            display_start = row.start_sec
            if k > 0 and len(row.text) > 0:
                # 重複除去した先頭分だけ開始時刻を進める(近似)
                display_start = row.start_sec + (row.end_sec - row.start_sec) * (
                    k / len(row.text)
                )
            label = f"[{_format_mmss(display_start)}〜{_format_mmss(row.end_sec)}] "
        else:
            label = ""
        lines.append(label + shown)
    if lines:
        parts.append("\n".join(lines))
    return "\n（…中略…）\n".join(parts)


def _fetch_chunk_rows(db: Session, chunk_table: str, transcription_id: int) -> List[_ChunkRow]:
    rows = db.execute(
        text(
            f"SELECT chunk_index, chunk_text, start_sec, end_sec, time_basis "
            f"FROM {chunk_table} "
            f"WHERE transcription_id = :id ORDER BY chunk_index"
        ),
        {"id": transcription_id},
    ).mappings().all()
    out: List[_ChunkRow] = []
    for r in rows:
        start_sec = r["start_sec"]
        end_sec = r["end_sec"]
        try:
            start_sec = float(start_sec) if start_sec is not None else None
            end_sec = float(end_sec) if end_sec is not None else None
        except (TypeError, ValueError):
            start_sec = end_sec = None
        out.append(
            _ChunkRow(
                index=int(r["chunk_index"]),
                text=r["chunk_text"] or "",
                start_sec=start_sec,
                end_sec=end_sec if start_sec is not None else None,
                time_basis=r["time_basis"],
            )
        )
    return out


def build_context_docs(
    db: Session,
    hits: List[Dict],
    *,
    max_docs: int,
    max_chars: int,
    whole_doc_threshold: int = 4000,
    neighbor_window: int = 1,
    order: str = "score",  # "score" | "date"
    per_doc_cap: Optional[int] = None,
) -> List[ContextDoc]:
    """検索ヒット(チャンク単位)を録音単位のコンテキストに変換する。

    per_doc_capを指定すると1録音あたりの文字数上限を固定できる。
    期間要約などで多数の録音を薄く広く読むために使う(未指定時は従来の
    「whole_doc_thresholdと均等割の大きい方」)。
    """
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

    if per_doc_cap is None:
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

        chunk_rows = _fetch_chunk_rows(db, chunk_table, g["transcription_id"])
        timed = any(r.has_time for r in chunk_rows)
        time_basis: Optional[str] = None
        if timed:
            time_basis = next((r.time_basis for r in chunk_rows if r.time_basis), None)

        is_full = len(transcript) <= whole_doc_threshold
        if is_full:
            if timed:
                # 全文でも時刻ラベルを付けるため、全チャンクを結合して構成する
                # (チャンクは文単位で全文を覆うため内容は全文と同等)
                body = _merge_timed_chunk_windows(
                    chunk_rows, {r.index for r in chunk_rows}
                )
                if not body:
                    body = transcript
            else:
                body = transcript
        else:
            ordered_chunks = [(r.index, r.text) for r in chunk_rows]
            if g["hit_indexes"] and ordered_chunks:
                keep: set = set()
                for idx in g["hit_indexes"]:
                    for i in range(idx - neighbor_window, idx + neighbor_window + 1):
                        keep.add(i)
                if timed:
                    body = _merge_timed_chunk_windows(chunk_rows, keep)
                else:
                    body = _merge_chunk_windows(
                        ordered_chunks, g["hit_indexes"], neighbor_window
                    )
            else:
                body = transcript  # チャンク情報がない場合は全文(後段で切り詰め)
                time_basis = None

        if not timed:
            time_basis = None

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
                time_basis=time_basis,
            )
        )
        used_chars += len(body)

    return docs
