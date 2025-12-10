import streamlit as st


def run_results_tab():
    st.header("処理結果")

    if not st.session_state.get("transcriptions"):
        st.info("処理結果がありません。音声ファイルをアップロードまたは録音してください。")
        return

    for idx, result in enumerate(st.session_state.transcriptions):
        with st.expander(f"📁 {result['file_name']}", expanded=True):
            col1, col2 = st.columns([1, 1])
            with col1:
                st.subheader("基本情報")
                st.write(f"**録音時刻:** {result['created_at'].strftime('%Y/%m/%d %H:%M')}")
                st.write(f"**録音時間:** {result['duration_seconds']:.1f}秒")
                st.write(f"**タグ:** {result['tags']}")
                st.subheader("文字起こしテキスト")
                st.text_area("", result['transcript'], height=200, key=f"text_{idx}")
            with col2:
                if result.get('structured_json'):
                    st.subheader("構造化データ")
                    st.json(result['structured_json'])
                else:
                    st.info("構造化データはありません")

