from datetime import datetime
from pathlib import Path
import tempfile
import os
import pandas as pd
import streamlit as st
import librosa

from models import AudioTranscription, get_db
from stt_wrapper import STTModelWrapper
from text_structurer import TextStructurer
from services.rag_service import get_rag_service


def run_upload_tab(selected_model: str, use_structuring: bool, logger):
    st.header("音声ファイルアップロード")

    uploaded_files = st.file_uploader(
        "音声ファイルを選択してください",
        type=["wav", "mp3", "m4a", "flac", "ogg", "webm"],
        accept_multiple_files=True,
        help="複数ファイルを同時にアップロード可能です",
    )

    if not uploaded_files:
        return

    st.success(f"{len(uploaded_files)}個のファイルがアップロードされました")
    df_files = pd.DataFrame([
        {"ファイル名": f.name, "サイズ": f"{f.size / 1024:.1f} KB", "タイプ": f.type}
        for f in uploaded_files
    ])
    st.dataframe(df_files, use_container_width=True)

    if st.button(
        "🚀 文字起こし開始", type="primary", use_container_width=True, disabled=st.session_state.get("processing", False)
    ):
        st.session_state.processing = True
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            stt_wrapper = STTModelWrapper(selected_model)
            text_structurer = TextStructurer() if use_structuring else None
        except Exception as e:
            st.error(f"初期化エラー: {e}")
            st.session_state.processing = False
            st.stop()

        rag_service = get_rag_service()

        for idx, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"処理中: {uploaded_file.name} ({idx + 1}/{len(uploaded_files)})")
            progress_bar.progress((idx + 1) / len(uploaded_files))
            try:
                logger.info(f"処理開始: {uploaded_file.name}")
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                audio_data, sr = librosa.load(tmp_path, sr=None)
                duration = len(audio_data) / sr
                logger.debug(f"音声ファイル情報: 時間={duration:.2f}秒, サンプリングレート={sr}Hz")

                logger.info(f"文字起こし実行中: {uploaded_file.name} (モデル: {selected_model})")
                transcription = stt_wrapper.transcribe(tmp_path)

                error_msg = None
                if isinstance(transcription, tuple) and transcription[0] is None:
                    error_msg = transcription[1]
                    transcription = None
                    logger.error(f"文字起こしエラー: {error_msg}")

                if transcription:
                    structured_data = None
                    tags = "未分類"
                    if use_structuring and text_structurer:
                        structured_data = text_structurer.structure_text(transcription)
                        if structured_data:
                            tags = text_structurer.extract_tags(structured_data)

                    result = {
                        "ファイル名": uploaded_file.name,
                        "録音時刻": datetime.now(),
                        "録音時間": duration,
                        "文字起こしテキスト": transcription,
                        "構造化データ": structured_data,
                        "タグ": tags,
                        "発言人数": 1,
                    }

                    st.session_state.transcriptions.append(result)

                    db = next(get_db())
                    try:
                        audio_record = AudioTranscription(
                            音声ファイルpath=uploaded_file.name,
                            発言人数=1,
                            録音時刻=datetime.now(),
                            録音時間=duration,
                            文字起こしテキスト=transcription,
                            構造化データ=structured_data,
                            タグ=tags,
                        )
                        db.add(audio_record)
                        db.flush()

                        if rag_service.enabled:
                            try:
                                rag_service.index_transcription(db, audio_record.音声ID, transcription)
                            except Exception as exc:  # pragma: no cover - API例外
                                logger.error("RAG埋め込みの生成に失敗: %s", exc, exc_info=True)

                        db.commit()
                    except Exception:
                        db.rollback()
                        raise
                    finally:
                        db.close()
                else:
                    if error_msg:
                        st.error(f"❌ {uploaded_file.name} の文字起こしに失敗しました")
                        st.error(f"エラー詳細: {error_msg}")
                        logger.error(f"文字起こし失敗: {uploaded_file.name}, エラー: {error_msg}")
                    else:
                        st.error(f"❌ {uploaded_file.name} の文字起こしに失敗しました（結果が空）")
                        logger.error(f"文字起こし失敗: {uploaded_file.name}, 結果が空")
                os.unlink(tmp_path)
                logger.debug(f"一時ファイル削除: {tmp_path}")
            except Exception as e:
                error_msg = f"処理エラー ({uploaded_file.name}): {str(e)}"
                st.error(error_msg)
                logger.error(error_msg, exc_info=True)

        progress_bar.progress(1.0)
        status_text.text("✅ すべての処理が完了しました！")
        st.session_state.processing = False
        st.rerun()
