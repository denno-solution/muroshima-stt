import streamlit as st
from pathlib import Path
import os
from dotenv import load_dotenv
import logging

from ui.sidebar import build_sidebar
from ui.tabs.upload_tab import run_upload_tab
from ui.tabs.mic_tab import run_mic_tab
from ui.tabs.results_tab import run_results_tab
from ui.tabs.db_tab import run_db_tab
from ui.tabs.rag_tab import run_rag_tab
# .envファイルを読み込む
load_dotenv()

from models import AudioTranscription, get_db
from stt_wrapper import STTModelWrapper
from text_structurer import TextStructurer
from env_watcher import check_env_changes, display_env_status
from app_settings import AppSettings
from auth import check_password, logout

# ロガーの設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# ログディレクトリの作成
log_dir = Path(__file__).parent.parent / "logs"
log_dir.mkdir(exist_ok=True)

# ファイルハンドラー
file_handler = logging.FileHandler(log_dir / "streamlit_app.log", encoding='utf-8')
file_handler.setLevel(logging.DEBUG)

# フォーマッター
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

# ページ設定
st.set_page_config(
    page_title="音声文字起こしWebアプリ",
    page_icon="🎙️",
    layout="wide"
)

# 環境変数の変更をチェック
check_env_changes()

# アプリ設定の初期化
settings = AppSettings()

# Basic認証チェック
if not check_password():
    st.stop()

# タイトル
st.title("🎙️ 音声文字起こしWebアプリ")
st.markdown("音声ファイルをアップロードして、文字起こしと構造化を行います。")

# セッション状態の初期化
if "transcriptions" not in st.session_state:
    st.session_state.transcriptions = []
if "processing" not in st.session_state:
    st.session_state.processing = False
if "mic_processing" not in st.session_state:
    st.session_state.mic_processing = False
if "mic_audio_bytes" not in st.session_state:
    st.session_state.mic_audio_bytes = None
if "settings" not in st.session_state:
    st.session_state.settings = settings

with st.sidebar:
    selected_model, use_structuring, debug_mode = build_sidebar(settings, log_dir, logger)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📤 アップロード",
    "🎙️ マイク録音",
    "📊 処理結果",
    "🗄️ データベース",
    "💬 QA検索",
])
with tab1:
    run_upload_tab(selected_model, use_structuring, logger)
with tab2:
    run_mic_tab(selected_model, use_structuring, logger)
with tab3:
    run_results_tab()
with tab4:
    run_db_tab()
with tab5:
    run_rag_tab()

# フッター
st.divider()
st.markdown("🎙️ 音声文字起こしWebアプリ v1.0")
