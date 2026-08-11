from pathlib import Path
import tempfile
import os
import pandas as pd
import streamlit as st
import librosa

from models import AudioTranscription, get_db, utcnow_naive
from stt_wrapper import STTModelWrapper
from text_structurer import TextStructurer
from services.rag_service import get_rag_service
from services.vad import trim_non_speech
from services.word_timestamps import build_word_timestamp_columns


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

                # VAD前処理（任意）
                app_settings = st.session_state.get("settings")
                use_vad = bool(getattr(app_settings, "get_use_vad", lambda: True)())
                vad_aggr = int(getattr(app_settings, "get_vad_aggressiveness", lambda: 2)())
                stt_input_path = tmp_path
                vad_note = None
                vad_applied = False  # STT入力がVADトリム後音声か
                vad_kept_ranges = None
                if use_vad:
                    try:
                        vad_res = trim_non_speech(tmp_path, enabled=True, aggressiveness=vad_aggr)
                        stt_input_path = vad_res.output_path
                        vad_applied = True
                        vad_kept_ranges = vad_res.kept_ranges
                        reduced = 0.0
                        if vad_res.orig_sec > 0:
                            reduced = max(0.0, 1.0 - (vad_res.out_sec / vad_res.orig_sec)) * 100.0
                        vad_note = f"VAD有効: 元{vad_res.orig_sec:.2f}s → 送信{vad_res.out_sec:.2f}s (−{reduced:.1f}%) [{vad_res.method}]"
                        st.info(vad_note)
                        logger.info(vad_note)
                    except Exception as e:
                        logger.warning(f"VAD前処理に失敗したためスキップ: {e}")
                        st.warning("VAD前処理に失敗したため、元音声を使用します。")
                        stt_input_path = tmp_path
                        vad_applied = False
                        vad_kept_ranges = None

                logger.info(f"文字起こし実行中: {uploaded_file.name} (モデル: {selected_model})")
                stt_result = stt_wrapper.transcribe_detailed(stt_input_path)
                transcription = stt_result.text
                error_msg = stt_result.error
                if error_msg:
                    transcription = None
                    logger.error(f"文字起こしエラー: {error_msg}")

                if transcription:
                    structured_data = None
                    tags = "未分類"
                    if use_structuring and text_structurer:
                        structured_data = text_structurer.structure_text(transcription)
                        if structured_data:
                            tags = text_structurer.extract_tags(structured_data)

                    # 単語タイムスタンプ: STTが返したまま(VAD後基準)と、
                    # VAD保持区間から元音声基準へ復元した値の両方を保存する
                    word_ts, word_ts_original = build_word_timestamp_columns(
                        stt_result.words,
                        vad_applied=vad_applied,
                        vad_ranges=vad_kept_ranges,
                    )

                    # created_atはnaive UTCで統一(desktop版と日付判定の互換のため)
                    created_at = utcnow_naive()
                    result = {
                        "file_name": uploaded_file.name,
                        "created_at": created_at,
                        "duration_seconds": duration,
                        "transcript": transcription,
                        "structured_json": structured_data,
                        "tags": tags,
                    }

                    st.session_state.transcriptions.append(result)

                    db = next(get_db())
                    try:
                        audio_record = AudioTranscription(
                            file_path=uploaded_file.name,
                            created_at=created_at,
                            duration_seconds=duration,
                            transcript=transcription,
                            structured_json=structured_data,
                            tags=tags,
                            word_timestamps_json=word_ts,
                            word_timestamps_original_json=word_ts_original,
                        )
                        db.add(audio_record)
                        db.flush()

                        if rag_service.enabled:
                            try:
                                rag_service.index_transcription(db, audio_record.id, transcription)
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
                # 一時ファイル削除
                try:
                    os.unlink(tmp_path)
                    logger.debug(f"一時ファイル削除: {tmp_path}")
                except Exception:
                    pass
                # VADで生成した一時ファイルも削除
                if stt_input_path != tmp_path:
                    try:
                        os.unlink(stt_input_path)
                        logger.debug(f"VAD一時ファイル削除: {stt_input_path}")
                    except Exception:
                        pass
            except Exception as e:
                error_msg = f"処理エラー ({uploaded_file.name}): {str(e)}"
                st.error(error_msg)
                logger.error(error_msg, exc_info=True)

        progress_bar.progress(1.0)
        status_text.text("✅ すべての処理が完了しました！")
        st.session_state.processing = False
        st.rerun()
