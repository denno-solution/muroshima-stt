# Streamlit Cloud デプロイ手順

## 準備

1. **GitHub リポジトリ**にコードをプッシュ
2. **PostgreSQL データベース**を準備（以下のいずれか）：
   - [Supabase](https://supabase.com) - 無料プランあり ← 選択済み！
   - [Neon](https://neon.tech) - 無料プランあり
   - [Render PostgreSQL](https://render.com/docs/databases)

### Supabaseセットアップ手順
1. `create_tables.sql`の内容をSupabaseのSQL Editorで実行
2. Settings → Database → Connection stringからURLをコピー
3. .envファイルのDATABASE_URLを更新
4. `python test_db_connection.py`で接続テスト

## デプロイ手順

### 1. Streamlit Cloud アカウント作成
1. https://streamlit.io/cloud にアクセス
2. GitHubアカウントでサインイン

### 2. アプリのデプロイ
1. "New app" をクリック
2. GitHubリポジトリを選択
3. ブランチ: main
4. Main file path: src/app.py

### 3. 環境変数の設定
Streamlit Cloud管理画面で Settings > Secrets から設定：

```toml
# PostgreSQL接続
DATABASE_URL = "postgresql://user:password@host:5432/dbname"

# 必要なAPIキー（使用するもののみ）
OPENAI_API_KEY = "sk-..."
GEMINI_API_KEY = "AIza..."
ELEVENLABS_API_KEY = "..."

# その他のAPIキー（必要に応じて）
GOOGLE_CLOUD_PROJECT = "..."
AWS_ACCESS_KEY_ID = "..."
AWS_SECRET_ACCESS_KEY = "..."
AZURE_SPEECH_KEY = "..."
AZURE_SPEECH_REGION = "..."
```

### 4. デプロイ完了
- アプリのURLが発行されます
- 初回起動時は数分かかることがあります

## その他のデプロイオプション

### Render.com
```yaml
# render.yaml
services:
  - type: web
    name: stt-app
    env: docker
    dockerfilePath: ./Dockerfile
    envVars:
      - key: DATABASE_URL
        fromDatabase:
          name: stt-db
          property: connectionString
```

### Railway
1. https://railway.app でプロジェクト作成
2. GitHubリポジトリを接続
3. 環境変数を設定
4. PostgreSQLサービスを追加

### ファイルストレージについて
音声ファイルの永続化が必要な場合：
- Amazon S3
- Google Cloud Storage
- Cloudinary（メディアファイル専用）

を検討してください。

## トラブルシューティング

### メモリ不足エラー
- Streamlit Cloudの無料プランは1GBメモリ制限
- 大きな音声ファイルの処理時に問題になる可能性
- 有料プランへのアップグレードを検討

### ファイルアップロードサイズ
- .streamlit/config.tomlで`maxUploadSize`を調整
- デフォルトは200MB

### データベース接続エラー
- DATABASE_URLの形式を確認
- PostgreSQLサービスのファイアウォール設定を確認
- SSL接続が必要な場合は`?sslmode=require`を追加