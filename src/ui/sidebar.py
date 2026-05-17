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

    # 前処理（VAD）
    st.subheader("🎚️ 前処理")
    use_vad = st.checkbox(
        "非音声区間のカット（VAD）",
        value=settings.get_use_vad(),
        help="webrtcvad（軽量）で人の声がない区間を削除してからSTTに送信し、コストを削減します。"
    )
    if use_vad != settings.get_use_vad():
        settings.set_use_vad(use_vad)
        logger.info(f"VAD設定を保存: {use_vad}")

    vad_aggr = st.slider(
        "VAD積極度 (0=緩い, 3=厳しい)",
        min_value=0,
        max_value=3,
        value=settings.get_vad_aggressiveness(),
        help="値が大きいほど非音声と判定しやすくなります。誤カットが増える場合は下げてください。",
    )
    if vad_aggr != settings.get_vad_aggressiveness():
        settings.set_vad_aggressiveness(vad_aggr)
        logger.info(f"VAD積極度を保存: {vad_aggr}")

    st.divider()

    # 社長音声: 参照フォルダ
    st.subheader("🎤 社長音声")
    current_source_dir = settings.get_ceo_source_dir()

    # text_input のキーに対しては、ウィジェット生成後に session_state を直接書き換えできない
    # ため、選択/リセットは on_click コールバック内で session_state を更新する。
    if "sidebar_ceo_source_dir" not in st.session_state:
        st.session_state["sidebar_ceo_source_dir"] = current_source_dir

    def _pick_ceo_source_dir():
        """OS ネイティブのフォルダ選択ダイアログを起動して session_state に反映する。"""

        import platform
        import subprocess
        import sys as _sys

        initial = st.session_state.get("sidebar_ceo_source_dir") or str(Path.home())
        picked = ""
        error_message: Optional[str] = None

        try:
            if platform.system() == "Darwin":
                script = (
                    'tell application "Finder" to activate\n'
                    f'set targetFolder to choose folder with prompt "社長音声の参照フォルダを選択" default location POSIX file "{initial}"\n'
                    "return POSIX path of targetFolder"
                )
                proc = subprocess.run(
                    ["osascript", "-e", script],
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                if proc.returncode == 0:
                    picked = (proc.stdout or "").strip().rstrip("/")
                else:
                    stderr = (proc.stderr or "").strip()
                    if "User canceled" in stderr or "User cancelled" in stderr or proc.returncode == 1:
                        picked = ""  # ユーザーキャンセル
                    else:
                        error_message = stderr[:300] or f"exit {proc.returncode}"
            else:
                picker_code = (
                    "import tkinter as tk\n"
                    "from tkinter import filedialog\n"
                    "root = tk.Tk()\n"
                    "root.withdraw()\n"
                    "root.attributes('-topmost', True)\n"
                    f"picked = filedialog.askdirectory(title='社長音声の参照フォルダを選択', initialdir={initial!r}, mustexist=True)\n"
                    "root.destroy()\n"
                    "print(picked or '', end='')\n"
                )
                proc = subprocess.run(
                    [_sys.executable, "-c", picker_code],
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                if proc.returncode != 0:
                    error_message = (
                        f"picker subprocess exited {proc.returncode}: "
                        f"{(proc.stderr or '').strip()[:300]}"
                    )
                else:
                    picked = (proc.stdout or "").strip()
        except subprocess.TimeoutExpired:
            error_message = "フォルダ選択ダイアログがタイムアウトしました（5分）"
            logger.warning("ceo source picker timed out")
        except Exception as exc:
            error_message = str(exc)
            logger.warning(f"ceo source picker failed: {exc}")

        if error_message:
            st.session_state["_ceo_pick_error"] = error_message
        elif picked:
            st.session_state["sidebar_ceo_source_dir"] = picked
            settings.set_ceo_source_dir(picked)
            logger.info(f"社長音声 参照フォルダを選択ダイアログから保存: {picked}")
        # picked == "" かつエラーなしはキャンセル扱い、何もしない

    def _reset_ceo_source_dir():
        st.session_state["sidebar_ceo_source_dir"] = ""
        settings.set_ceo_source_dir("")
        logger.info("社長音声 参照フォルダをリセット")

    ceo_source_dir = st.text_input(
        "参照フォルダ（サーバー側パス）",
        key="sidebar_ceo_source_dir",
        help=(
            "「未処理を自動取り込み」で再帰スキャンするフォルダの絶対パスを指定します。"
            "Streamlit はサーバー側で動くため、サーバー上で読めるパスを指定してください。"
        ),
        placeholder="例: /var/data/ceo_audio または C:\\\\ceo_audio",
    )
    col_pick, col_reset = st.columns([1, 1])
    with col_pick:
        st.button(
            "📁 選択",
            use_container_width=True,
            key="sidebar_ceo_source_pick",
            help="サーバー側のフォルダ選択ダイアログを開きます（ローカル実行時のみ動作）",
            on_click=_pick_ceo_source_dir,
        )
    with col_reset:
        st.button(
            "🗑️ リセット",
            use_container_width=True,
            key="sidebar_ceo_source_reset",
            disabled=not st.session_state.get("sidebar_ceo_source_dir"),
            help="参照フォルダの設定をクリアします",
            on_click=_reset_ceo_source_dir,
        )

    # 直前のフォルダ選択で発生したエラーがあれば表示してクリア
    if "_ceo_pick_error" in st.session_state:
        st.error(
            f"フォルダ選択ダイアログを開けませんでした: {st.session_state.pop('_ceo_pick_error')}\n"
            "手で入力欄にパスを貼り付けてください。"
        )

    # 手入力での変更も永続化（コールバック経由の変更とは別経路）
    if ceo_source_dir.strip() != current_source_dir:
        settings.set_ceo_source_dir(ceo_source_dir)
        logger.info(f"社長音声 参照フォルダを保存: {ceo_source_dir.strip()}")

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
