"""社長音声 DB ビューア。

`ceo_transcriptions` テーブルを一覧・検索・詳細表示する。
stt-desktop の CeoDbView を Streamlit 向けに簡素化した形。
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from sqlalchemy import delete

from models import CeoTranscription, get_db


def _ensure_state() -> None:
    st.session_state.setdefault(
        "ceo_db_tab_state",
        {
            "loaded": False,
            "table": None,
            "records": [],
        },
    )


def _load_records():
    db = next(get_db())
    try:
        records = (
            db.query(CeoTranscription)
            .order_by(CeoTranscription.recorded_at.desc(), CeoTranscription.created_at.desc())
            .all()
        )
    finally:
        db.close()

    table_rows: list[dict] = []
    detail_rows: list[dict] = []
    for record in records:
        transcript = record.transcript or ""
        table_rows.append(
            {
                "ID": record.id,
                "タイトル": record.title or "-",
                "話者": record.speaker or "-",
                "録音日時": record.recorded_at or "-",
                "登録日時": record.created_at,
                "長さ(s)": record.duration_seconds,
                "ファイル": record.source_file_path or record.file_path or "-",
                "文字起こし": transcript[:50] + ("…" if len(transcript) > 50 else ""),
                "_文字起こし全文": transcript,
            }
        )
        detail_rows.append(
            {
                "id": record.id,
                "title": record.title,
                "speaker": record.speaker,
                "recorded_at": record.recorded_at,
                "created_at": record.created_at,
                "duration_seconds": record.duration_seconds,
                "source_file_path": record.source_file_path,
                "source_file_size_bytes": record.source_file_size_bytes,
                "source_file_modified_at": record.source_file_modified_at,
                "source_file_hash": record.source_file_hash,
                "file_path": record.file_path,
                "local_file_path": record.local_file_path,
                "model_id": record.model_id,
                "language_code": record.language_code,
                "tags": record.tags,
                "transcript": transcript,
                "structured_json": record.structured_json,
            }
        )

    df = pd.DataFrame(table_rows)
    return df, detail_rows


def _delete_record(record_id: int) -> None:
    db = next(get_db())
    try:
        db.execute(delete(CeoTranscription).where(CeoTranscription.id == record_id))
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def run_ceo_db_tab() -> None:
    st.header("📂 社長音声履歴")
    st.caption("`ceo_transcriptions` テーブルの内容を表示します。")

    _ensure_state()
    state = st.session_state["ceo_db_tab_state"]

    col_load, col_clear = st.columns([1, 4])
    with col_load:
        if st.button(
            "🔄 再読み込み" if state["loaded"] else "📥 データを読み込む",
            key="ceo_db_tab_load",
        ):
            df, records = _load_records()
            state["table"] = df
            state["records"] = records
            state["loaded"] = True
    with col_clear:
        if state["loaded"]:
            st.caption(f"レコード数: {len(state['records'])} 件")

    if not state["loaded"]:
        st.info("「データを読み込む」をクリックすると履歴を表示します。")
        return

    df: pd.DataFrame | None = state["table"]
    records: list[dict] = state["records"]

    if df is None or df.empty:
        st.info("社長音声のレコードはまだありません。")
        return

    speakers = ["すべて"] + sorted({r["speaker"] for r in records if r["speaker"]})
    fc1, fc2 = st.columns([1, 3])
    with fc1:
        speaker_filter = st.selectbox("話者でフィルタ", speakers, index=0, key="ceo_db_speaker_filter")
    with fc2:
        text_filter = st.text_input(
            "タイトル/文字起こしで検索",
            value="",
            key="ceo_db_text_filter",
            placeholder="キーワードを入力",
        )

    filtered = df.copy()
    if speaker_filter != "すべて":
        filtered = filtered[filtered["話者"] == speaker_filter]
    if text_filter.strip():
        q = text_filter.strip().lower()
        mask = (
            filtered["タイトル"].astype(str).str.lower().str.contains(q, regex=False, na=False)
            | filtered["_文字起こし全文"].astype(str).str.lower().str.contains(q, regex=False, na=False)
            | filtered["ファイル"].astype(str).str.lower().str.contains(q, regex=False, na=False)
        )
        filtered = filtered[mask]

    visible = filtered.drop(columns=["_文字起こし全文"], errors="ignore")
    st.dataframe(visible, use_container_width=True, hide_index=True)

    if filtered.empty:
        return

    if st.checkbox("詳細を表示", key="ceo_db_show_detail"):
        record_map = {r["id"]: r for r in records}
        selected_id = st.selectbox(
            "ID を選択",
            filtered["ID"].tolist(),
            key="ceo_db_detail_select",
        )
        record = record_map.get(selected_id)
        if record is None:
            return

        st.subheader(f"ID: {record['id']} の詳細")
        meta_l, meta_r = st.columns([1, 1])
        with meta_l:
            st.write(f"**タイトル:** {record['title'] or '-'}")
            st.write(f"**話者:** {record['speaker'] or '-'}")
            st.write(f"**録音日時:** {record['recorded_at'] or '-'}")
            st.write(f"**登録日時:** {record['created_at']}")
            st.write(f"**長さ:** {record['duration_seconds']}")
            st.write(f"**モデル:** {record['model_id'] or '-'}")
            st.write(f"**言語:** {record['language_code'] or '-'}")
            st.write(f"**タグ:** {record['tags'] or '-'}")
        with meta_r:
            st.write(f"**source_file_path:** {record['source_file_path'] or '-'}")
            st.write(f"**source_file_size_bytes:** {record['source_file_size_bytes'] or '-'}")
            st.write(f"**source_file_modified_at:** {record['source_file_modified_at'] or '-'}")
            st.write(f"**source_file_hash:** {record['source_file_hash'] or '-'}")
            st.write(f"**file_path:** {record['file_path'] or '-'}")
            st.write(f"**local_file_path:** {record['local_file_path'] or '-'}")

        st.subheader("文字起こしテキスト")
        st.text_area(
            "",
            record["transcript"] or "",
            height=240,
            key=f"ceo_db_detail_text_{record['id']}",
        )

        if record["structured_json"]:
            st.subheader("構造化データ")
            st.json(record["structured_json"])

        with st.expander("⚠️ このレコードを削除", expanded=False):
            confirm = st.checkbox(
                "削除に同意します（取り消し不可）",
                key=f"ceo_db_confirm_delete_{record['id']}",
            )
            if st.button(
                "🗑️ 削除する",
                disabled=not confirm,
                type="secondary",
                key=f"ceo_db_delete_{record['id']}",
            ):
                _delete_record(record["id"])
                state["loaded"] = False
                st.success(f"ID {record['id']} を削除しました。")
                st.rerun()
