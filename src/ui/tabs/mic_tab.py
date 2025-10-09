import os
import tempfile
from datetime import datetime
import streamlit as st

from models import AudioTranscription, get_db
from stt_wrapper import STTModelWrapper
from text_structurer import TextStructurer

from services.audio_utils import md5_bytes, should_convert_to_wav, convert_webm_to_wav
from services.rag_service import get_rag_service


def run_mic_tab(selected_model: str, use_structuring: bool, logger):
    st.header("マイク録音")
    st.markdown("**マイクから直接音声を録音して文字起こしします**")

    audio_bytes = st.audio_input("🎙️ マイクで録音してください", help="録音ボタンを押して停止で録音完了")

    if not audio_bytes:
        return

    st.success("録音完了！")

    # 重複抑止（MD5）
    try:
        raw = audio_bytes.getvalue() if hasattr(audio_bytes, 'getvalue') else audio_bytes
        current_digest = md5_bytes(raw)
    except Exception:
        current_digest = None

    is_new_recording = False
    if current_digest is not None:
        is_new_recording = st.session_state.get("mic_last_digest") != current_digest
    else:
        is_new_recording = audio_bytes != st.session_state.get("mic_audio_bytes")

    if not st.session_state.get("mic_processing") and is_new_recording:
        st.session_state.mic_audio_bytes = audio_bytes
        st.session_state.mic_processing = True
        st.session_state.mic_last_digest = current_digest
        st.info("自動で文字起こしを開始します…")
        st.rerun()

    if not st.session_state.mic_processing:
        return

    try:
        # 一時保存（WebM想定）
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmp_file:
            if hasattr(audio_bytes, 'getvalue'):
                tmp_file.write(audio_bytes.getvalue())
            else:
                tmp_file.write(audio_bytes)
            webm_path = tmp_file.name

        logger.info(f"マイク録音処理開始: {webm_path}")

        # 必要なモデルのみWAV変換
        tmp_path = webm_path
        duration = 0.0
        if should_convert_to_wav(selected_model):
            try:
                wav_path, duration = convert_webm_to_wav(webm_path, target_sr=16000)
                os.unlink(webm_path)
                tmp_path = wav_path
                logger.info(f"音声変換完了: WebM → WAV ({wav_path})")
            except Exception as e:
                tmp_path = webm_path
                duration = 0.0
                logger.warning(f"音声変換失敗（WebMで処理継続）: {e}")

        # クラウドストレージ連携は削除（外部依存を排除）
        storage_path = None

        # STT 実行
        stt_wrapper = STTModelWrapper(selected_model)
        text_structurer = TextStructurer() if use_structuring else None
        rag_service = get_rag_service()

        with st.spinner("文字起こし中..."):
            transcription = stt_wrapper.transcribe(tmp_path)
            error_msg = None
            if isinstance(transcription, tuple) and transcription[0] is None:
                error_msg = transcription[1]
                transcription = None
                logger.error(f"マイク録音文字起こしエラー: {error_msg}")

            if transcription:
                structured_data = None
                tags = "マイク録音"
                if use_structuring and text_structurer:
                    with st.spinner("テキスト構造化中..."):
                        structured_data = text_structurer.structure_text(transcription)
                        if structured_data:
                            tags = text_structurer.extract_tags(structured_data)

                timestamp = datetime.now()
                file_extension = ".wav" if tmp_path.endswith('.wav') else ".webm"
                result = {
                    "ファイル名": f"マイク録音_{timestamp.strftime('%Y%m%d_%H%M%S')}{file_extension}",
                    "録音時刻": timestamp,
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
                        音声ファイルpath=result["ファイル名"],
                        発言人数=1,
                        録音時刻=timestamp,
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
                    logger.info(f"マイク録音結果をデータベースに保存: {result['ファイル名']}")
                except Exception:
                    db.rollback()
                    raise
                finally:
                    db.close()

                st.success("✅ 文字起こし完了！")
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.subheader("文字起こし結果")
                    st.text_area("", transcription, height=200, key="mic_transcription")
                    st.write(f"**録音時間:** {duration:.1f}秒")
                    st.write(f"**タグ:** {tags}")
                with col2:
                    if structured_data:
                        st.subheader("構造化データ")
                        st.json(structured_data)
                    else:
                        st.info("構造化データはありません")
            else:
                if error_msg:
                    st.error(f"❌ マイク録音の文字起こしに失敗しました")
                    if "invalid_api_key" in str(error_msg).lower():
                        st.error("🔑 APIキーが無効です")
                        st.info("💡 別のSTTモデルに切り替えるか、APIキーを確認してください。")
                    elif "internal server error" in str(error_msg).lower():
                        st.error("🔧 サーバーで一時的な問題が発生しました")
                        st.info("💡 数分後に再試行するか、別のSTTモデルに切り替えてください。")
                    else:
                        st.error(f"エラー詳細: {error_msg}")
                else:
                    st.error("❌ マイク録音の文字起こしに失敗しました（結果が空）")

        # 一時ファイル削除
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        logger.debug(f"一時ファイル削除: {tmp_path}")

        # 状態リセット
        st.session_state.mic_processing = False
        st.session_state.mic_audio_bytes = None

    except Exception as e:
        error_msg = f"マイク録音処理エラー: {str(e)}"
        st.error(error_msg)
        logger.error(error_msg, exc_info=True)
        st.session_state.mic_processing = False

    st.divider()
    st.markdown("**💡 使い方のヒント:**")
    st.markdown("- 録音ボタンを押してから話してください")
    st.markdown("- 録音終了後は自動で処理が始まります")
    st.markdown("- 録音データは一時的に保存され、処理後に削除されます")
