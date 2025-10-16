import os
import pandas as pd
import streamlit as st

from models import AudioTranscription, get_db
from services.cloudflare_r2 import (
    load_r2_config_from_env,
    build_object_key_for_filename,
    build_public_url_for_key,
    generate_presigned_get_url,
)


def run_db_tab():
    st.header("データベース内容")

    db = next(get_db())
    try:
        records = db.query(AudioTranscription).all()
        if not records:
            st.info("データベースにレコードがありません。")
            return

        # Cloudflare R2 設定（あればダウンロードリンクを生成）
        r2_cfg = load_r2_config_from_env()
        signed_exp = int(os.getenv("R2_SIGNED_URL_EXPIRES", "900"))  # 既定: 15分

        data = []
        for record in records:
            text = record.文字起こしテキスト or ""
            download_url = None
            if r2_cfg is not None:
                key = build_object_key_for_filename(record.音声ファイルpath, r2_cfg)
                if key:
                    # 優先: 公開URL（R2_PUBLIC_BASE_URL が設定されている場合）
                    download_url = build_public_url_for_key(key, r2_cfg) or generate_presigned_get_url(
                        key, expires_in=signed_exp, cfg=r2_cfg
                    )
            data.append({
                "音声ID": record.音声ID,
                "発言人数": record.発言人数,
                "録音時刻": record.録音時刻,
                "録音時間(s)": record.録音時間,
                "タグ": record.タグ,
                "文字起こし": text[:50] + "..." if len(text) > 50 else text,
                "音声ファイルダウンロード": download_url,
            })

        df = pd.DataFrame(data)

        col1, _ = st.columns([1, 1])
        with col1:
            tag_filter = st.selectbox("タグでフィルタ", ["すべて"] + list(df["タグ"].unique()))
        if tag_filter != "すべて":
            df = df[df["タグ"] == tag_filter]

        st.dataframe(
            df,
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

        if st.checkbox("詳細を表示"):
            selected_id = st.selectbox("音声IDを選択", df["音声ID"].tolist())
            if selected_id:
                record = db.query(AudioTranscription).filter_by(音声ID=selected_id).first()
                if record:
                    st.subheader(f"音声ID: {record.音声ID} の詳細")
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.write(f"**ファイル:** {record.音声ファイルpath}")
                        st.write(f"**録音時刻:** {record.録音時刻}")
                        st.write(f"**録音時間:** {record.録音時間}秒")
                        st.write(f"**タグ:** {record.タグ}")
                        st.subheader("文字起こしテキスト")
                        st.text_area("", record.文字起こしテキスト, height=200)
                    with col2:
                        if record.構造化データ:
                            st.subheader("構造化データ")
                            st.json(record.構造化データ)
                        # 右側にダウンロードリンクも提示
                        if r2_cfg is not None:
                            key = build_object_key_for_filename(record.音声ファイルpath, r2_cfg)
                            if key:
                                url = build_public_url_for_key(key, r2_cfg) or generate_presigned_get_url(
                                    key, expires_in=signed_exp, cfg=r2_cfg
                                )
                                if url:
                                    st.subheader("ダウンロード")
                                    st.link_button("Cloudflare R2 からダウンロード", url)
    finally:
        db.close()
