import streamlit as st
import os
import hmac


def check_password():
    """
    Basic認証のチェック
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
                    st.success("ログインに成功しました！")
                    st.rerun()
                else:
                    st.error("ユーザー名またはパスワードが正しくありません。")
    
    return False


def logout():
    """ログアウト処理"""
    if "authenticated" in st.session_state:
        st.session_state.authenticated = False