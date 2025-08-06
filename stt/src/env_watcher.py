import os
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv, dotenv_values
import hashlib

def get_env_hash():
    """現在の.envファイルのハッシュ値を取得"""
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        with open(env_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    return None

def check_env_changes():
    """環境変数の変更をチェックして必要に応じてリロード"""
    # 現在のハッシュ値を取得
    current_hash = get_env_hash()
    
    # セッション状態に前回のハッシュ値を保存
    if 'env_hash' not in st.session_state:
        st.session_state.env_hash = current_hash
    
    # ハッシュ値が変更されていたら環境変数を再読み込み
    if current_hash != st.session_state.env_hash:
        load_dotenv(override=True)  # 強制的に再読み込み
        st.session_state.env_hash = current_hash
        st.rerun()  # アプリを再実行
        
def display_env_status(sidebar=True):
    """環境変数の設定状況を表示"""
    if sidebar:
        container = st.sidebar
    else:
        container = st
        
    with container.expander("🔧 環境変数の設定状況", expanded=False):
        env_vars = {
            "OPENAI_API_KEY": "OpenAI",
            "GEMINI_API_KEY": "Gemini",
            "GOOGLE_AI_API_KEY": "Google AI (代替)",
            "GOOGLE_CLOUD_PROJECT": "Google Cloud",
            "AWS_ACCESS_KEY_ID": "AWS",
            "AZURE_SPEECH_KEY": "Azure",
            "ELEVENLABS_API_KEY": "ElevenLabs"
        }
        
        for key, name in env_vars.items():
            value = os.getenv(key)
            if value:
                if len(value) > 10:
                    masked_value = value[:4] + "..." + value[-4:]
                else:
                    masked_value = "***設定済み***"
                st.success(f"✅ {name}: {masked_value}")
            else:
                st.info(f"⚪ {name}: 未設定")
                
        if st.button("🔄 環境変数を再読み込み"):
            load_dotenv(override=True)
            st.rerun()