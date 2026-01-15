import os
import json
import time
import uuid
import logging
import streamlit as st
from datetime import datetime, date
from typing import List, Dict, Optional

from sqlalchemy import func, or_, select

from models import AudioTranscriptionChunk, USE_VECTOR, VECTOR_BACKEND, get_db, RAGChatLog
from services.rag import highlight_date_in_query
from services.rag_service import get_rag_service

logger = logging.getLogger(__name__)


def _get_or_create_session_id() -> str:
    """現在のセッションIDを取得、なければ新規作成"""
    if "rag_session_id" not in st.session_state:
        st.session_state.rag_session_id = str(uuid.uuid4())
    return st.session_state.rag_session_id


def _load_session_history(session_id: str) -> List[Dict]:
    """DBからセッションの履歴を復元"""
    db = next(get_db())
    try:
        logs = (
            db.query(RAGChatLog)
            .filter(RAGChatLog.session_id == session_id)
            .order_by(RAGChatLog.created_at.asc())
            .all()
        )
        history = []
        for log in logs:
            history.append({"role": "user", "content": log.user_text})
            history.append({
                "role": "assistant",
                "content": log.answer_text or "",
                "contexts": log.contexts or [],
            })
        return history
    finally:
        db.close()


def _render_date_filter_badge(meta: Dict):
    """日付フィルタ状態をバッジ表示"""
    date_filter = meta.get("date_filter")
    if not date_filter:
        return

    start = date_filter.get("start", "")
    end = date_filter.get("end", "")
    date_str = start if start == end else f"{start} 〜 {end}"

    if meta.get("date_filtered"):
        st.markdown(f"📅 :green[**日付フィルタ適用中**: {date_str}]")
    elif meta.get("date_no_match"):
        st.markdown(f"⚠️ :orange[**日付該当なし**: {date_str}（全データから検索）]")


def _render_context_chunks(contexts: List[Dict], max_display: Optional[int] = None, truncate: bool = False):
    """参照チャンクを統一形式で表示"""
    display_contexts = contexts[:max_display] if max_display else contexts

    for idx, ctx in enumerate(display_contexts, start=1):
        score_parts = [f"**総合スコア:** {ctx.get('score', 0):.3f}"]
        if ctx.get("score_vector") is not None:
            score_parts.append(f"ベクトル: {ctx['score_vector']:.3f}")
        if ctx.get("score_fts") is not None:
            score_parts.append(f"FTS: {ctx['score_fts']:.3f}")
        score_str = " / ".join(score_parts)

        meta_parts = []
        if ctx.get("file_path"):
            meta_parts.append(f"📁 {ctx['file_path']}")
        if ctx.get("tag"):
            meta_parts.append(f"🏷️ {ctx['tag']}")
        if ctx.get("recorded_at"):
            meta_parts.append(f"📅 {ctx['recorded_at']}")
        meta_str = " / ".join(meta_parts) if meta_parts else ""

        st.markdown(f"**{idx}.** {score_str}")
        if meta_str:
            st.caption(meta_str)

        chunk_text = ctx.get("chunk_text", "")
        if truncate and len(chunk_text) > 200:
            st.text(chunk_text[:200] + "…")
        else:
            st.write(chunk_text)

        st.divider()


def _handle_rag_error(error: Exception, context: str = ""):
    """RAGエラーの統一ハンドリング"""
    error_msg = str(error)

    if "OPENAI_API_KEY" in error_msg or "api_key" in error_msg.lower():
        st.error("🔑 APIキーが設定されていないか無効です。環境変数を確認してください。")
    elif "rate_limit" in error_msg.lower() or "429" in error_msg:
        st.warning("⏳ APIのレート制限に達しました。しばらく待ってから再試行してください。")
    elif "timeout" in error_msg.lower():
        st.error("⏰ タイムアウトが発生しました。ネットワーク接続を確認してください。")
    else:
        st.error(f"❌ エラーが発生しました{': ' + context if context else ''}")
        with st.expander("詳細を表示"):
            st.code(error_msg)

    logger.error(f"RAG error ({context}): {error}", exc_info=True)


def _fetch_session_summaries(keyword: str = "", limit: int = 20):
    """セッション一覧を取得（最終更新順）。"""
    db = next(get_db())
    try:
        session_ids_subq = None
        if keyword:
            session_ids_subq = (
                db.query(RAGChatLog.session_id)
                .filter(RAGChatLog.session_id.isnot(None))
                .filter(
                    or_(
                        RAGChatLog.user_text.contains(keyword),
                        RAGChatLog.answer_text.contains(keyword),
                    )
                )
                .distinct()
                .subquery()
            )

        base = (
            db.query(
                RAGChatLog.session_id.label("session_id"),
                func.min(RAGChatLog.created_at).label("first_created"),
                func.max(RAGChatLog.created_at).label("last_updated"),
                func.count(RAGChatLog.id).label("message_count"),
            )
            .filter(RAGChatLog.session_id.isnot(None))
        )
        if session_ids_subq is not None:
            base = base.filter(RAGChatLog.session_id.in_(select(session_ids_subq.c.session_id)))
        base = base.group_by(RAGChatLog.session_id).subquery()

        sessions = (
            db.query(
                base.c.session_id,
                base.c.last_updated,
                base.c.message_count,
                RAGChatLog.user_text.label("first_question"),
            )
            .join(
                RAGChatLog,
                (RAGChatLog.session_id == base.c.session_id)
                & (RAGChatLog.created_at == base.c.first_created),
            )
            .order_by(base.c.last_updated.desc())
            .limit(limit)
            .all()
        )
        return sessions
    finally:
        db.close()


def run_rag_tab():
    st.header("RAG検索（文字起こしQA）")

    rag_service = get_rag_service()

    if not rag_service.enabled:
        if not USE_VECTOR:
            st.warning(
                "RAG機能を利用するには Turso(libSQL) のベクトル対応データベースを構成し、"
                "audio_transcription_chunks に libsql_vector_idx を作成してください（自動作成済みでない場合）。"
            )
            st.info("ローカルの通常SQLiteではRAGは無効化されます。Tursoの `sqlite+libsql://` を使用してください。")
        else:
            st.warning(
                "OPENAI_API_KEY が未設定、もしくは埋め込みモデル設定に問題があるためRAGが無効化されています。"
            )
            st.info("環境変数にOPENAI_API_KEYを設定し、必要に応じて EMBEDDING_MODEL / EMBEDDING_DIM を調整してください。")
        return

    # セッション管理
    session_id = _get_or_create_session_id()

    if "rag_history" not in st.session_state:
        st.session_state.rag_history = []

    # --- セッション管理UI ---
    session_cols = st.columns([2, 1, 1])
    with session_cols[0]:
        st.caption(f"セッションID: {session_id[:8]}...")
    with session_cols[1]:
        if st.button("新規セッション", use_container_width=True, help="新しい会話を開始"):
            st.session_state.rag_session_id = str(uuid.uuid4())
            st.session_state.rag_history = []
            st.rerun()
    with session_cols[2]:
        if st.button("履歴クリア", use_container_width=True, help="現在の会話履歴をクリア"):
            st.session_state.rag_history = []
            st.rerun()

    # --- 高度な設定 ---
    default_alpha = float(os.getenv("RAG_HYBRID_ALPHA", "0.6"))
    default_retrieval_k = int(os.getenv("RAG_RETRIEVAL_K", "100"))
    default_context_k = int(os.getenv("RAG_CONTEXT_MAX_CHUNKS", "12"))

    with st.expander("🔧 検索設定", expanded=False):
        st.caption("検索は全データから上位候補を自動抽出します。実行中にタブを切り替えると処理が中断されることがあります。")

        setting_cols = st.columns(2)
        with setting_cols[0]:
            use_hybrid = st.checkbox(
                "ハイブリッド検索 (FTS×ベクトル)",
                value=True,
                help="FTS（全文検索）とベクトル検索を併用",
            )
            alpha = st.slider(
                "ベクトル重み α",
                min_value=0.0,
                max_value=1.0,
                value=default_alpha,
                step=0.05,
                help="1.0=ベクトルのみ、0.0=FTSのみ",
            )
        with setting_cols[1]:
            retrieval_k = st.number_input(
                "検索候補上限",
                min_value=10,
                max_value=200,
                value=default_retrieval_k,
                step=10,
                help="検索候補の母集団サイズ",
            )
            context_k = st.number_input(
                "使用チャンク上限",
                min_value=3,
                max_value=30,
                value=default_context_k,
                step=1,
                help="プロンプトに含めるチャンク数の上限",
            )

    for message in st.session_state.rag_history:
        block = st.chat_message(message["role"])
        if message["role"] == "user":
            block.markdown(highlight_date_in_query(message["content"]))
        else:
            block.markdown(message["content"])
        if message["role"] == "assistant" and message.get("contexts"):
            with block.expander("参照したチャンク", expanded=False):
                _render_context_chunks(message["contexts"])

    query = st.chat_input("文字起こしデータへの質問を入力してください（例: 「昨日の会議について」「12月3日の打ち合わせ内容」）")

    if query:
        st.session_state.rag_history.append({"role": "user", "content": query})
        # 日付部分をハイライトして表示
        st.chat_message("user").markdown(highlight_date_in_query(query))

        # 会話履歴を構築（現在の質問は除外）
        chat_history_for_rag = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in st.session_state.rag_history[:-1]  # 最後の質問は除外
            if msg["role"] in ("user", "assistant") and msg.get("content")
        ]

        # ストリーミング実行
        with st.spinner("検索を実行中..."):
            db = next(get_db())
            try:
                result2 = rag_service.answer_stream(
                    db,
                    query,
                    top_k=retrieval_k,
                    hybrid=use_hybrid,
                    alpha=alpha,
                    context_k=context_k,
                    chat_history=chat_history_for_rag,
                )
            except Exception as e:
                _handle_rag_error(e, "検索実行")
                db.close()
                return
            finally:
                db.close()

        matches = result2.get("matches", [])
        meta = result2.get("meta") or {}
        stream_fn = result2.get("stream_fn")

        # 日付フィルタバッジを先に表示
        _render_date_filter_badge(meta)

        # ストリーミング表示
        with st.chat_message("assistant"):
            tgen0 = time.time()
            try:
                full_text = st.write_stream(stream_fn()) if callable(stream_fn) else ""
            except Exception as e:
                acc = ""
                placeholder = st.empty()
                try:
                    for chunk in (stream_fn() if callable(stream_fn) else []):
                        acc += str(chunk)
                        placeholder.markdown(acc)
                    full_text = acc
                except Exception as inner_e:
                    _handle_rag_error(inner_e, "ストリーミング出力")
                    full_text = ""
            tgen1 = time.time()

            # 参照チャンク表示（共通コンポーネント使用）
            if matches:
                with st.expander("参照したチャンク", expanded=False):
                    _render_context_chunks(matches)

        # メタ情報（秒単位、日付情報は上部バッジに移動）
        timings = (meta.get("timings_ms") or {}) if isinstance(meta, dict) else {}
        retrieval_s = (timings.get("retrieval") or 0) / 1000.0
        prompt_s = (timings.get("prompt_build") or 0) / 1000.0
        gen_s = (tgen1 - tgen0)
        total_s = retrieval_s + prompt_s + gen_s
        cap = f"候補: {meta.get('candidates')} / 使用: {meta.get('used_context_chunks')} 件"
        cap += f" / 検索: {retrieval_s:.3f}s / 生成: {gen_s:.3f}s / 合計: {total_s:.3f}s"
        st.caption(cap)

        # 履歴・DB保存
        st.session_state.rag_history.append(
            {"role": "assistant", "content": full_text, "contexts": matches}
        )

        def _json_default(o):
            if isinstance(o, (datetime, date)):
                return o.isoformat()
            return str(o)

        contexts_json = json.loads(json.dumps(matches, default=_json_default))

        with st.spinner("チャットを保存中..."):
            db2 = next(get_db())
            try:
                date_filter = meta.get("date_filter")
                log = RAGChatLog(
                    session_id=session_id,
                    user_text=query,
                    answer_text=full_text,
                    contexts=contexts_json,
                    used_hybrid=bool(use_hybrid),
                    alpha=float(alpha) if alpha is not None else None,
                    date_filter_applied=bool(meta.get("date_filtered")) if date_filter else False,
                )
                db2.add(log)
                db2.commit()
            except Exception as e:
                db2.rollback()
                logger.error(f"チャット保存エラー: {e}")
                st.warning("チャットの保存に失敗しました。ログをご確認ください。")
            finally:
                db2.close()

    # --- セッション履歴 ---
    st.divider()
    st.subheader("セッション履歴")

    colh1, colh2 = st.columns([2, 1])
    with colh1:
        kw = st.text_input(
            "キーワード（セッション内の質問/回答を対象）",
            value="",
            placeholder="例: 契約 期限",
        )
    with colh2:
        session_limit = st.slider("表示件数", min_value=5, max_value=50, value=20, step=5)

    sessions = _fetch_session_summaries(keyword=kw, limit=session_limit)

    if not sessions:
        st.info("保存されたセッションがありません。検索/質問後にここへ表示されます。")
    else:
        for s in sessions:
            title = (s.first_question or "").strip()
            if not title:
                title = "（無題）"
            if len(title) > 40:
                title = title[:40] + "…"
            label = f"[{s.last_updated}] {title}"
            if s.session_id == session_id:
                label = f"▶ {label}"
            if st.button(label, key=f"resume_{s.session_id}", use_container_width=True):
                st.session_state.rag_session_id = s.session_id
                st.session_state.rag_history = _load_session_history(s.session_id)
                st.rerun()
            st.caption(f"{s.message_count}件のメッセージ")
