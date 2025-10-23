import os
import pandas as pd
import streamlit as st

from models import AudioTranscription, get_db
from services.cloudflare_r2 import (
    load_r2_config_from_env,
    build_object_key_for_filename,
    build_public_url_for_key,
    generate_presigned_get_url,
    object_exists_in_r2,
)


def _ensure_state():
    if "db_tab_state" not in st.session_state:
        st.session_state.db_tab_state = {
            "loaded": False,
            "table": None,
            "records": [],
        }
    if "r2_exists_cache" not in st.session_state:
        st.session_state.r2_exists_cache = {}


def _load_db_records():
    db = next(get_db())
    try:
        records = db.query(AudioTranscription).order_by(AudioTranscription.録音時刻.desc()).all()
    finally:
        db.close()

    if not records:
        return pd.DataFrame(), []

    r2_cfg = load_r2_config_from_env()
    signed_exp = int(os.getenv("R2_SIGNED_URL_EXPIRES", "900"))
    r2_cache = st.session_state.r2_exists_cache

    table_rows = []
    detail_rows = []

    for record in records:
        text = record.文字起こしテキスト or ""
        tag_value = record.タグ or ""
        download_url = None

        if r2_cfg is not None:
            key = build_object_key_for_filename(record.音声ファイルpath, r2_cfg)
            if key:
                exists = r2_cache.get(key)
                if exists is None:
                    exists = object_exists_in_r2(key, r2_cfg)
                    r2_cache[key] = exists
                if exists:
                    download_url = build_public_url_for_key(key, r2_cfg) or generate_presigned_get_url(
                        key, expires_in=signed_exp, cfg=r2_cfg
                    )

        table_rows.append({
            "音声ID": record.音声ID,
            "発言人数": record.発言人数,
            "録音時刻": record.録音時刻,
            "録音時間(s)": record.録音時間,
            "タグ": tag_value,
            "文字起こし": text[:50] + "..." if len(text) > 50 else text,
            "音声ファイルダウンロード": download_url,
        })

        detail_rows.append({
            "音声ID": record.音声ID,
            "音声ファイルpath": record.音声ファイルpath,
            "発言人数": record.発言人数,
            "録音時刻": record.録音時刻,
            "録音時間": record.録音時間,
            "タグ": tag_value,
            "文字起こしテキスト": text,
            "構造化データ": record.構造化データ,
            "download_url": download_url,
        })

    df = pd.DataFrame(table_rows)
    return df, detail_rows


def run_db_tab():
    st.header("データベース内容")
    _ensure_state()

    state = st.session_state.db_tab_state
    load_trigger = False

    if not state["loaded"]:
        if st.button("データを読み込む", key="db_tab_load"):
            load_trigger = True
    else:
        col_reload, _ = st.columns([1, 1])
        with col_reload:
            if st.button("再読み込み", key="db_tab_reload"):
                load_trigger = True

    if load_trigger:
        df, records = _load_db_records()
        state["table"] = df
        state["records"] = records
        state["loaded"] = True

    if not state["loaded"]:
        st.info("「データを読み込む」をクリックするとデータベースを表示します。")
        return

    df = state["table"]
    records = state["records"]

    if df is None or df.empty:
        st.info("データベースにレコードがありません。")
        return

    tag_options = ["すべて"] + sorted({row["タグ"] for row in records if row["タグ"]})
    if "タグ選択" not in st.session_state:
        st.session_state["タグ選択"] = "すべて"

    selected_tag = st.selectbox("タグでフィルタ", tag_options, index=0 if st.session_state["タグ選択"] not in tag_options else tag_options.index(st.session_state["タグ選択"]))
    st.session_state["タグ選択"] = selected_tag

    filtered_df = df if selected_tag == "すべて" else df[df["タグ"] == selected_tag]

    st.dataframe(
        filtered_df,
        use_container_width=True,
        column_config={
            "音声ファイルダウンロード": st.column_config.LinkColumn(
                label="音声ファイルダウンロード",
                help="Cloudflare R2から音声をダウンロード",
                display_text="リンク",
            ),
        },
        hide_index=True,
    )

    if not filtered_df.empty and st.checkbox("詳細を表示"):
        record_map = {row["音声ID"]: row for row in records}
        selected_id = st.selectbox("音声IDを選択", filtered_df["音声ID"].tolist())
        record = record_map.get(selected_id)

        if record:
            st.subheader(f"音声ID: {record['音声ID']} の詳細")
            col1, col2 = st.columns([1, 1])
            with col1:
                st.write(f"**ファイル:** {record['音声ファイルpath']}")
                st.write(f"**発言人数:** {record['発言人数']}")
                st.write(f"**録音時刻:** {record['録音時刻']}")
                st.write(f"**録音時間:** {record['録音時間']}秒")
                st.write(f"**タグ:** {record['タグ'] or '-'}")
                st.subheader("文字起こしテキスト")
                st.text_area("", record["文字起こしテキスト"], height=200)
            with col2:
                if record["構造化データ"]:
                    st.subheader("構造化データ")
                    st.json(record["構造化データ"])
                if record["download_url"]:
                    st.subheader("ダウンロード")
                    st.link_button("Cloudflare R2 からダウンロード", record["download_url"])
