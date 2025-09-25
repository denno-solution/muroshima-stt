import pandas as pd
import streamlit as st

from models import AudioTranscription, get_db


def run_db_tab():
    st.header("データベース内容")

    db = next(get_db())
    try:
        records = db.query(AudioTranscription).all()
        if not records:
            st.info("データベースにレコードがありません。")
            return

        data = []
        for record in records:
            text = record.文字起こしテキスト or ""
            data.append({
                "音声ID": record.音声ID,
                "音声ファイル": record.音声ファイルpath,
                "発言人数": record.発言人数,
                "録音時刻": record.録音時刻,
                "録音時間(s)": record.録音時間,
                "タグ": record.タグ,
                "文字起こし": text[:50] + "..." if len(text) > 50 else text,
            })

        df = pd.DataFrame(data)

        col1, _ = st.columns([1, 1])
        with col1:
            tag_filter = st.selectbox("タグでフィルタ", ["すべて"] + list(df["タグ"].unique()))
        if tag_filter != "すべて":
            df = df[df["タグ"] == tag_filter]

        st.dataframe(df, use_container_width=True)

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
    finally:
        db.close()

