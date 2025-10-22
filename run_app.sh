#!/bin/bash

# 音声文字起こしWebアプリ起動スクリプト

echo "🎙️ 音声文字起こしWebアプリを起動します..."

# uvがインストールされているか確認
if ! command -v uv &> /dev/null; then
    echo "❌ uvがインストールされていません。"
    echo "インストール方法: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# 依存関係のインストール
if [ "${SKIP_UV_SYNC:-0}" = "1" ]; then
  echo "依存関係の同期をスキップします (SKIP_UV_SYNC=1)"
else
  echo "依存関係を確認しています..."
  uv sync
fi

# Streamlitアプリを起動
echo "アプリを起動しています..."
uv run streamlit run src/app.py
