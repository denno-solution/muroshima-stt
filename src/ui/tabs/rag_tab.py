import os
import json
import streamlit as st
from datetime import datetime, date

from models import AudioTranscriptionChunk, USE_VECTOR, VECTOR_BACKEND, get_db, RAGChatLog
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

    # 検索は常に全データ対象（ベクトル/FTSインデックスは全体から上位を返す）。
    # 候補取得件数（top_k）は内部既定値で固定し、UIでは非表示。
    st.caption("検索は全データから上位候補を自動抽出します（設定不要）。")

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
    # 回答に使うチャンク上限は内部既定値（RAG_CONTEXT_MAX_CHUNKS）で固定。UIでは非表示。
    context_k = None

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

    if query:
        st.session_state.rag_history.append({"role": "user", "content": query})
        st.chat_message("user").markdown(query)

        with st.spinner("関連チャンクを検索中..."):
            db = next(get_db())
            try:
                result = rag_service.answer(
                    db, query, top_k=None, hybrid=use_hybrid, alpha=alpha, context_k=context_k
                )
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

        # 実際の使用件数などのメタ情報を簡単に表示
        meta = result.get("meta") if isinstance(result, dict) else None
        if meta:
            st.caption(
                f"候補: {meta.get('candidates')} / 使用: {meta.get('used_context_chunks')} 件"
            )

        # チャットの入出力と参照コンテキストをDBに保存（JSONに安全に変換）
        def _json_default(o):
            if isinstance(o, (datetime, date)):
                return o.isoformat()
            return str(o)

        contexts_json = json.loads(json.dumps(matches, default=_json_default))

        with st.spinner("チャットを保存中..."):
            db2 = next(get_db())
            try:
                log = RAGChatLog(
                    user_text=query,
                    answer_text=answer,
                    contexts=contexts_json,
                    used_hybrid=bool(use_hybrid),
                    alpha=float(alpha) if alpha is not None else None,
                )
                db2.add(log)
                db2.commit()
            except Exception:
                db2.rollback()
                st.warning("チャットの保存に失敗しました。ログをご確認ください。")
            finally:
                db2.close()

    # --- 過去のチャット履歴（直近） ---
    st.divider()
    st.subheader("過去のチャット履歴")

    colh1, colh2, colh3 = st.columns([2, 1, 1])
    with colh1:
        kw = st.text_input("キーワード（質問/回答を対象）", value="", placeholder="例: 契約 期限" )
    with colh2:
        limit = st.slider("表示件数", min_value=5, max_value=100, value=20, step=5)
    with colh3:
        hybrid_filter = st.selectbox("ハイブリッド",
                                     options=["すべて", "ON", "OFF"], index=0)

    dbh = next(get_db())
    try:
        q = dbh.query(RAGChatLog).order_by(RAGChatLog.created_at.desc())
        if kw:
            from sqlalchemy import or_
            q = q.filter(
                or_(
                    RAGChatLog.user_text.contains(kw),
                    RAGChatLog.answer_text.contains(kw),
                )
            )
        if hybrid_filter == "ON":
            q = q.filter(RAGChatLog.used_hybrid.is_(True))
        elif hybrid_filter == "OFF":
            q = q.filter(RAGChatLog.used_hybrid.is_(False))

        logs = q.limit(limit).all()
    finally:
        dbh.close()

    if not logs:
        st.info("保存されたチャット履歴がありません。検索/質問後にここへ表示されます。")
    else:
        for log in logs:
            with st.expander(
                f"[{log.created_at}] 質問: { (log.user_text or '')[:40] + ('…' if log.user_text and len(log.user_text) > 40 else '') }",
                expanded=False,
            ):
                st.markdown("**質問**")
                st.write(log.user_text or "")
                st.markdown("**回答**")
                st.write(log.answer_text or "")
                st.caption(
                    f"ハイブリッド: {'ON' if log.used_hybrid else 'OFF'} / α: {log.alpha if log.alpha is not None else '-'}"
                )
                ctxs = log.contexts or []
                if ctxs:
                    st.markdown("**参照したチャンク（最大3件を表示）**")
                    for idx, ctx in enumerate(ctxs[:3], start=1):
                        st.markdown(
                            f"- {idx}. スコア: {ctx.get('score', '-')}; ファイル: {ctx.get('file_path','-')}; タグ: {ctx.get('tag','-')}"
                        )
                        if ctx.get("recorded_at"):
                            st.caption(f"録音時刻: {ctx['recorded_at']}")
                        snippet = (ctx.get("chunk_text") or "")
                        st.text(snippet[:200] + ("…" if len(snippet) > 200 else ""))
