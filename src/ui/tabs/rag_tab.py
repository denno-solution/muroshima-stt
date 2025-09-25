import streamlit as st

from models import get_db
from services.rag_service import get_rag_service


def run_rag_tab():
    st.header("RAG検索（文字起こしQA）")

    rag_service = get_rag_service()

    if not rag_service.enabled:
        st.warning(
            "RAG機能を利用するには Postgres + pgvector でデータベースを構成し、"
            "OPENAI_API_KEY を設定してください。DATABASE_URL が SQLite の場合は利用できません。"
        )
        st.info(
            "Supabaseでpgvectorを有効化し、audio_transcription_chunks テーブルを作成した後に再度お試しください。"
        )
        return

    if "rag_history" not in st.session_state:
        st.session_state.rag_history = []

    top_k = st.slider("検索するチャンク数", min_value=3, max_value=10, value=5)

    cols = st.columns([1, 1, 1])
    with cols[0]:
        if st.button("履歴をクリア", use_container_width=True):
            st.session_state.rag_history = []
            st.rerun()

    for message in st.session_state.rag_history:
        block = st.chat_message(message["role"])
        block.markdown(message["content"])
        if message["role"] == "assistant" and message.get("contexts"):
            with block.expander("参照したチャンク", expanded=False):
                for idx, ctx in enumerate(message["contexts"], start=1):
                    st.markdown(
                        f"**{idx}. スコア:** {ctx['score']:.3f}"
                        f" / **ファイル:** {ctx.get('file_path') or '-'}"
                        f" / **タグ:** {ctx.get('tag') or '-'}"
                    )
                    if ctx.get("recorded_at"):
                        st.caption(f"録音時刻: {ctx['recorded_at']}")
                    st.write(ctx["chunk_text"])
                    st.divider()

    query = st.chat_input("文字起こしデータへの質問を入力してください")

    if not query:
        return

    st.session_state.rag_history.append({"role": "user", "content": query})
    st.chat_message("user").markdown(query)

    with st.spinner("関連チャンクを検索中..."):
        db = next(get_db())
        try:
            result = rag_service.answer(db, query, top_k=top_k)
        finally:
            db.close()

    answer = result.get("answer", "")
    matches = result.get("matches", [])

    assistant_payload = {
        "role": "assistant",
        "content": answer,
        "contexts": matches,
    }
    st.session_state.rag_history.append(assistant_payload)

    assistant_block = st.chat_message("assistant")
    assistant_block.markdown(answer)

    if matches:
        with assistant_block.expander("参照したチャンク", expanded=False):
            for idx, ctx in enumerate(matches, start=1):
                st.markdown(
                    f"**{idx}. スコア:** {ctx['score']:.3f}"
                    f" / **ファイル:** {ctx.get('file_path') or '-'}"
                    f" / **タグ:** {ctx.get('tag') or '-'}"
                )
                if ctx.get("recorded_at"):
                    st.caption(f"録音時刻: {ctx['recorded_at']}")
                st.write(ctx["chunk_text"])
                st.divider()
