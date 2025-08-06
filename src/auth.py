import streamlit as st
import os
import hmac
import secrets
import json
from datetime import datetime, timedelta
import extra_streamlit_components as stx


# Cookie管理用のキー
_AUTH_COOKIE_NAME = "stt_auth_token"
_AUTH_TOKENS_KEY = "auth_tokens"  # 有効なトークンを保存するセッションキー

# CookieManagerのシングルトンインスタンス
def get_cookie_manager():
    """クッキーマネージャーのシングルトンインスタンスを取得"""
    return stx.CookieManager(key="auth_cookie_manager")

def _generate_token():
    """セキュアなランダムトークンを生成"""
    return secrets.token_urlsafe(32)

def _initialize_auth_storage():
    """認証ストレージの初期化"""
    if _AUTH_TOKENS_KEY not in st.session_state:
        st.session_state[_AUTH_TOKENS_KEY] = {}
    if "cookie_ready" not in st.session_state:
        st.session_state.cookie_ready = False

def _save_auth_token(username, token):
    """認証トークンをセッションに保存"""
    _initialize_auth_storage()
    # トークンと有効期限をセッションに保存
    st.session_state[_AUTH_TOKENS_KEY][token] = {
        "username": username,
        "expires": (datetime.now() + timedelta(days=1)).isoformat()
    }
    st.session_state.auth_token_to_save = token

def _check_auth_token(token):
    """トークンの有効性をチェック"""
    _initialize_auth_storage()
    
    if token in st.session_state[_AUTH_TOKENS_KEY]:
        token_data = st.session_state[_AUTH_TOKENS_KEY][token]
        expires = datetime.fromisoformat(token_data["expires"])
        if datetime.now() < expires:
            return True
        else:
            # 期限切れトークンを削除
            del st.session_state[_AUTH_TOKENS_KEY][token]
    return False

def _handle_cookie_operations():
    """Cookie保存・削除処理を実行"""
    cookie_manager = get_cookie_manager()
    
    # Cookieを保存する必要がある場合
    if "save_auth_cookie" in st.session_state and st.session_state.save_auth_cookie:
        if "auth_token_to_save" in st.session_state:
            cookie_manager.set(
                _AUTH_COOKIE_NAME,
                st.session_state.auth_token_to_save,
                expires_at=datetime.now() + timedelta(days=1)
            )
            del st.session_state.auth_token_to_save
        st.session_state.save_auth_cookie = False
    
    # Cookieをクリアする必要がある場合
    if "clear_auth_cookie" in st.session_state and st.session_state.clear_auth_cookie:
        cookie_manager.delete(_AUTH_COOKIE_NAME)
        st.session_state.clear_auth_cookie = False
    
    return cookie_manager

def check_password():
    """
    Basic認証のチェック（Cookie認証対応）
    環境変数BASIC_AUTH_USERNAMEとBASIC_AUTH_PASSWORDが設定されている場合のみ認証を要求
    """
    # 環境変数から認証情報を取得
    expected_username = os.getenv("BASIC_AUTH_USERNAME")
    expected_password = os.getenv("BASIC_AUTH_PASSWORD")
    
    # 認証情報が設定されていない場合は認証をスキップ
    if not expected_username or not expected_password:
        return True
    
    # セッション状態の初期化
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    
    # Cookie操作を処理し、Cookieマネージャーを取得
    cookie_manager = _handle_cookie_operations()
    
    # Cookieからトークンを読み取り
    saved_token = cookie_manager.get(_AUTH_COOKIE_NAME)
    
    # Cookie認証のチェック
    if not st.session_state.authenticated and saved_token and _check_auth_token(saved_token):
        st.session_state.authenticated = True
    
    # すでに認証済みの場合
    if st.session_state.authenticated:
        return True
    
    # ログインフォームの表示
    with st.container():
        st.markdown("## 🔒 ログイン")
        
        with st.form("login_form"):
            username = st.text_input("ユーザー名")
            password = st.text_input("パスワード", type="password")
            submitted = st.form_submit_button("ログイン")
            
            if submitted:
                # 認証チェック（タイミング攻撃対策のため比較にhmac.compare_digestを使用）
                username_match = hmac.compare_digest(username, expected_username)
                password_match = hmac.compare_digest(password, expected_password)
                
                if username_match and password_match:
                    st.session_state.authenticated = True
                    # 認証トークンを生成して保存
                    token = _generate_token()
                    _save_auth_token(username, token)
                    st.success("ログインに成功しました！")
                    # Cookieを設定するために保存フラグを立てる
                    st.session_state.save_auth_cookie = True
                    st.rerun()
                else:
                    st.error("ユーザー名またはパスワードが正しくありません。")
    
    return False


def logout():
    """ログアウト処理"""
    if "authenticated" in st.session_state:
        st.session_state.authenticated = False
    
    # Cookieをクリアするフラグを立てる
    st.session_state.clear_auth_cookie = True
    
    # セッションからトークンを削除
    if _AUTH_TOKENS_KEY in st.session_state:
        st.session_state[_AUTH_TOKENS_KEY].clear()
    
    # 他の認証関連のセッション状態をクリア
    keys_to_remove = ["auth_token_to_save", "save_auth_cookie"]
    for key in keys_to_remove:
        if key in st.session_state:
            del st.session_state[key]