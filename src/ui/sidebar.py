import os
from pathlib import Path
import streamlit as st

from app_settings import AppSettings
from env_watcher import display_env_status
from stt_wrapper import STTModelWrapper
from auth import logout


def build_sidebar(settings: AppSettings, log_dir: Path, logger):
    st.header("⚙️ 設定")

    available_models = STTModelWrapper.get_available_models()
    saved_model = settings.get_selected_stt_model()
    default_index = 4
    if saved_model and saved_model in available_models:
        default_index = available_models.index(saved_model)

    selected_model = st.selectbox(
        "STTモデルを選択",
        available_models,
        index=default_index,
        help="使用する音声認識モデルを選択してください",
    )
    if selected_model != saved_model:
        settings.set_selected_stt_model(selected_model)
        logger.info(f"STTモデルの選択を保存: {selected_model}")

    # モデル要件チェック
    try:
        wrapper = STTModelWrapper(selected_model)
        requirements = wrapper.check_requirements()
        if requirements:
            st.subheader("環境変数の設定状況")
            for key, is_set in requirements.items():
                if is_set:
                    st.success(f"✅ {key}")
                else:
                    st.error(f"❌ {key} が設定されていません")
    except Exception as e:
        st.error(f"モデルの初期化エラー: {e}")

    st.divider()

    # 構造化
    st.subheader("構造化設定")
    use_structuring = st.checkbox(
        "Gemini Flash 2.5-liteで自動構造化",
        value=settings.get_use_structuring(),
    )
    if use_structuring != settings.get_use_structuring():
        settings.set_use_structuring(use_structuring)
        logger.info(f"構造化設定を保存: {use_structuring}")
    if use_structuring:
        gemini_key_set = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_AI_API_KEY"))
        if gemini_key_set:
            st.success("✅ Gemini API キー設定済み")
        else:
            st.error("❌ GEMINI_API_KEY が設定されていません")

    st.divider()

    # デバッグ
    st.subheader("🐛 デバッグ設定")
    debug_mode = st.checkbox("デバッグモードを有効化", value=settings.get_debug_mode())
    if debug_mode != settings.get_debug_mode():
        settings.set_debug_mode(debug_mode)
        logger.info(f"デバッグモード設定を保存: {debug_mode}")
    if debug_mode:
        if st.button("📋 ログファイルを表示"):
            log_files = {
                "Streamlitログ": log_dir / "streamlit_app.log",
                "ElevenLabsデバッグログ": log_dir / "elevenlabs_debug.log",
            }
            for log_name, log_path in log_files.items():
                if log_path.exists():
                    st.subheader(f"📄 {log_name}")
                    try:
                        with open(log_path, "r", encoding="utf-8") as f:
                            lines = f.readlines()
                            recent = lines[-50:] if len(lines) > 50 else lines
                            st.code("".join(recent), language="log")
                    except Exception as e:
                        st.error(f"ログ読み込みエラー: {e}")
                else:
                    st.info(f"{log_name}はまだ存在しません")

    st.divider()
    display_env_status(sidebar=True)

    if os.getenv("BASIC_AUTH_USERNAME") and os.getenv("BASIC_AUTH_PASSWORD"):
        st.divider()
        if st.button("🚪 ログアウト", type="secondary", use_container_width=True):
            logout()
            st.rerun()

    return selected_model, use_structuring, debug_mode

