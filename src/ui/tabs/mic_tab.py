import os
import tempfile
from datetime import datetime
import streamlit as st

from models import AudioTranscription, get_db
from stt_wrapper import STTModelWrapper
from text_structurer import TextStructurer

from pathlib import Path
from services.audio_utils import (
    md5_bytes,
    should_convert_to_wav,
    convert_webm_to_wav,
    get_audio_duration,
)
from services.cloudflare_r2 import load_r2_config_from_env, upload_file_to_r2
from services.rag_service import get_rag_service


def run_mic_tab(selected_model: str, use_structuring: bool, logger):
    st.header("マイク録音")
    st.markdown("**マイクから直接音声を録音して文字起こしします**")
    save_local = (os.getenv("SAVE_MIC_AUDIO_LOCAL", "true").lower() == "true")
    save_dir = os.getenv("MIC_AUDIO_SAVE_DIR", "data/recordings")
    save_to_r2 = (os.getenv("SAVE_MIC_AUDIO_TO_R2", "false").lower() == "true")

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
                duration = get_audio_duration(webm_path)
                logger.warning(f"音声変換失敗（WebMで処理継続）: {e}")
        else:
            # 変換不要でも長さは計測
            duration = get_audio_duration(tmp_path)
        # ローカルへ永続保存（必要なら）
        final_path: str | None = None
        timestamp = datetime.now()
        file_extension = ".wav" if tmp_path.endswith('.wav') else ".webm"
        final_filename = f"mic_{timestamp.strftime('%Y%m%d_%H%M%S')}{file_extension}"
        if save_local:
            try:
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                final_path = str(Path(save_dir) / final_filename)
                # tmpを所定の保存先へ移動（同一FS前提。失敗時はコピーでも良い）
                os.replace(tmp_path, final_path)
                tmp_path = final_path  # 以降のSTTも保存先ファイルを使用
                logger.info(f"ローカル保存: {final_path}")
            except Exception as e:
                logger.warning(f"ローカル保存に失敗（処理は継続）: {e}")
                final_path = None
        else:
            final_path = None

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

                # R2へアップロード（必要なら）
                r2_info = None
                if save_to_r2:
                    cfg = load_r2_config_from_env()
                    if cfg is None:
                        logger.error("R2設定が不足しています。R2_* 環境変数を確認してください。")
                    else:
                        try:
                            # 保存済みのローカルパスがある場合はそれを使い、なければ一時ファイルを送る
                            source_path = final_path or tmp_path
                            key = final_filename
                            r2_info = upload_file_to_r2(source_path, key, cfg)
                            logger.info(f"R2アップロード成功: s3://{r2_info['bucket']}/{r2_info['key']}")
                        except Exception as exc:
                            logger.error(f"R2アップロード失敗: {exc}")

                result = {
                    "ファイル名": final_filename,
                    "録音時刻": timestamp,
                    "録音時間": duration,
                    "文字起こしテキスト": transcription,
                    "構造化データ": structured_data,
                    "タグ": tags,
                    "発言人数": 1,
                    "保存先": final_path,
                    "r2_url": (r2_info or {}).get("url") if (r2_info) else None,
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
                    st.subheader("保存情報")
                    if save_local and result.get("保存先"):
                        st.success(f"ローカル保存: {result['保存先']}")
                    elif save_local:
                        st.warning("ローカル保存が有効ですが保存に失敗しました")
                    else:
                        st.info("ローカル保存は無効です（SAVE_MIC_AUDIO_LOCAL=false）")

                    if save_to_r2 and result.get("r2_url"):
                        st.success(f"R2 URL: {result['r2_url']}")
                    elif save_to_r2:
                        st.warning("R2アップロードに失敗またはURL未設定（R2_PUBLIC_BASE_URL）")
                    else:
                        st.info("R2アップロードは無効です（SAVE_MIC_AUDIO_TO_R2=false）")
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
        # 一時ファイル削除（ローカル保存できていない場合のみ）
        if not save_local:
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
    if save_local:
        st.markdown(f"- 録音データは `{save_dir}` に保存されます（ファイル名: `mic_YYYYMMDD_HHMMSS.*`）")
    else:
        st.markdown("- 録音データは一時保存のみで処理後に削除されます（SAVE_MIC_AUDIO_LOCAL=false）")
    if save_to_r2:
        st.markdown("- Cloudflare R2 にもアップロードされます（環境変数が正しく設定されている場合）")
