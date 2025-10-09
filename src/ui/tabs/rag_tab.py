import os
import streamlit as st

from models import AudioTranscriptionChunk, USE_VECTOR, VECTOR_BACKEND, get_db
from services.rag_service import get_rag_service


def run_rag_tab():
    st.header("RAG検索（文字起こしQA）")

    rag_service = get_rag_service()

    if not rag_service.enabled:
        if not USE_VECTOR:
            st.warning(
                "RAG機能を利用するには Postgres + pgvector または Turso(libSQL) のベクトル対応データベースを構成し、"
                "audio_transcription_chunks にベクトルインデックスを作成してください。"
            )
            st.info(
                "Postgresの場合はpgvector拡張を有効化、Turso(libSQL)の場合は libsql_vector_idx を作成した上で再度お試しください。"
            )
        else:
            st.warning(
                "OPENAI_API_KEY が未設定、もしくは埋め込みモデル設定に問題があるためRAGが無効化されています。"
            )
            if VECTOR_BACKEND == "libsql":
                st.info("環境変数にOPENAI_API_KEYを設定し、必要に応じて EMBEDDING_MODEL / EMBEDDING_DIM を調整してください。")
            else:
                st.info("OPENAI_API_KEY を設定し、アプリを再起動してください。")
        return

    if "rag_history" not in st.session_state:
        st.session_state.rag_history = []

    db = next(get_db())
    try:
        chunk_count = db.query(AudioTranscriptionChunk.id).count()
    finally:
        db.close()

    # 既定値は控えめに（負荷回避）。データ量に応じて調整。
    default_top_k = max(min(10, chunk_count), 1)

    top_k = int(
        st.number_input(
            "検索するチャンク数",
            min_value=1,
            value=default_top_k,
            step=1,
            help="チャンク数を多くするとリコールは上がりますが、レスポンス時間やコストも増加します。",
        )
    )

    if top_k > 50:
        st.warning("50件を超える検索は応答が遅くなる可能性があります。必要に応じて数を見直してください。")

    # ハイブリッド検索オプション
    default_alpha = float(os.getenv("RAG_HYBRID_ALPHA", "0.6"))
    cols = st.columns([1, 1, 1])
    with cols[0]:
        if st.button("履歴をクリア", use_container_width=True):
            st.session_state.rag_history = []
            st.rerun()
    with cols[1]:
        use_hybrid = st.checkbox(
            "ハイブリッド検索を有効化 (FTS×ベクトル)",
            value=True,
            help=(
                "FTS（全文検索）とベクトル検索を併用し、両者のスコアを重み付きで統合します。\n"
                "libSQLではFTS5の仮想テーブルが必要です（自動作成済み）。"
            ),
        )
    with cols[2]:
        alpha = st.slider(
            "ベクトル重み α",
            min_value=0.0,
            max_value=1.0,
            value=default_alpha,
            step=0.05,
            help="1.0でベクトルのみ、0.0でFTSのみ。既定は0.6。",
        )

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
            result = rag_service.answer(db, query, top_k=top_k, hybrid=use_hybrid, alpha=alpha)
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
                    f"**{idx}. 総合スコア:** {ctx['score']:.3f}"
                    f" / **ファイル:** {ctx.get('file_path') or '-'}"
                    f" / **タグ:** {ctx.get('tag') or '-'}"
                )
                if ctx.get("recorded_at"):
                    st.caption(f"録音時刻: {ctx['recorded_at']}")
                # サブスコア（あれば表示）
                sv = ctx.get("score_vector")
                sf = ctx.get("score_fts")
                if sv is not None or sf is not None:
                    st.caption(
                        f"ベクトル: {sv if sv is not None else '-'} / FTS: {sf if sf is not None else '-'}"
                    )
                st.write(ctx["chunk_text"])
                st.divider()
