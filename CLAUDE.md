# 音声文字起こしWebアプリ（STT Web App）

Streamlitを使用した音声文字起こしWebアプリ。複数のSTTモデルに対応し、Gemini Flash 2.5-liteによる自動構造化機能を搭載。

## 機能

- 複数音声ファイルの同時アップロード/マイク録音
- 5つのSTTモデル対応（OpenAI、Google Cloud、Amazon、Azure、ElevenLabs）
- Gemini Flash 2.5-liteによる文字起こしテキストの自動構造化
- PostgreSQL/SQLiteデータベース保存
- Basic認証によるアクセス制限（オプション）

## クイックスタート

### 1. セットアップ

```bash
# uvのインストール（未インストールの場合）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 依存関係をインストール
uv sync

# 環境変数を設定（.env.exampleをコピーして編集）
cp .env.example .env
nano .env
```

### 2. データベース設定

**Turso(libSQL) 使用（推奨）**:
1. TursoでDBを作成しURL/トークンを取得
2. `.env` の `DATABASE_URL` を `sqlite+libsql://<db>-<org>.turso.io?secure=true&authToken=...` に設定
3. 初回起動時に `audio_transcription_chunks` のベクトル式インデックス（libsql_vector_idx）が自動作成されます

**ローカル開発**: SQLiteが自動使用されます

### 3. 起動と使用

```bash
# アプリ起動
./run_app.sh

# ブラウザで http://localhost:8501 を開く
```

**使い方**:
1. サイドバーでSTTモデルを選択（デフォルト: ElevenLabs）
2. 音声入力:
   - **アップロード**: ファイル選択 → 「文字起こし開始」
   - **マイク録音**: 録音 → 「文字起こしてデータベースに保存」
3. 「処理結果」タブで確認、「データベース」タブで過去の結果を検索

## 環境変数設定

### 必須APIキー

| 用途 | 環境変数 | 備考 |
|------|---------|------|
| **STTモデル** | 下記いずれか1つ | 選択したモデル用 |
| **構造化** | GEMINI_API_KEY | Gemini Flash 2.5-lite用 |
| **データベース** | DATABASE_URL | PostgreSQL使用時 |
| **Basic認証** | BASIC_AUTH_USERNAME<br>BASIC_AUTH_PASSWORD | オプション |

### STTモデル別環境変数

| サービス | 環境変数 | ファイルサイズ制限 |
|---------|---------|------------------|
| OpenAI | OPENAI_API_KEY | 25MB |
| Google Cloud | GOOGLE_CLOUD_PROJECT<br>GOOGLE_APPLICATION_CREDENTIALS | 10分（約10MB） |
| Amazon | AWS_ACCESS_KEY_ID<br>AWS_SECRET_ACCESS_KEY | 2GB（S3経由） |
| Azure | AZURE_SPEECH_KEY<br>AZURE_SPEECH_REGION | 100MB |
| ElevenLabs | ELEVENLABS_API_KEY | 1GB、4.5時間 |

### サンプル.env

```env
# データベース（例1: Turso/libSQL）
DATABASE_URL=sqlite+libsql://your-db-your-org.turso.io?secure=true&authToken=your-turso-token

# データベース（例2: ローカルSQLite）
# DATABASE_URL=sqlite:///./audio_transcriptions.db

# STTモデル（ElevenLabsの例）
ELEVENLABS_API_KEY=xi-xxxxxxxxxxxxxxxxxxxxx

# 構造化機能
GEMINI_API_KEY=AIzaSyxxxxxxxxxxxxxxxxxxxxx

# Basic認証（オプション）
BASIC_AUTH_USERNAME=admin
BASIC_AUTH_PASSWORD=secure-password
```

## 設定とデータベース

### 設定の永続化
- STTモデル選択、構造化機能、デバッグモードの設定は`.app_settings.json`に自動保存

### Turso(libSQL)利用時のポイント
- `DATABASE_URL` に `sqlite+libsql://<db名>-<org>.turso.io?secure=true&authToken=...` を設定するとリモートTursoに接続可能
- `audio_transcription_chunks` の `libsql_vector_idx` 作成と `OPENAI_API_KEY` 設定でRAGタブが有効化

### データベーススキーマ

| カラム名 | 型 | 説明 |
|---------|-----|------|
| 音声ID | SERIAL | 主キー |
| 音声ファイルpath | VARCHAR(500) | ファイル名 |
| 発言人数 | INTEGER | デフォルト: 1 |
| 録音時刻 | TIMESTAMP | 処理時刻 |
| 録音時間 | FLOAT | 秒 |
| 文字起こしテキスト | TEXT | 結果 |
| 構造化データ | JSONB | Gemini出力 |
| タグ | VARCHAR(200) | 自動生成 |

### Basic認証とCookie
- 環境変数で有効化、24時間有効なCookie認証トークン使用
- ログアウトボタンはサイドバーに表示

## 対応フォーマット
WAV、MP3、M4A、FLAC、OGG

## プロジェクト構造（主要ファイル）

```
stt/
├── src/
│   ├── app.py               # メインアプリ
│   ├── stt_wrapper.py       # STT統一インターフェース
│   └── text_structurer.py   # Gemini構造化
├── scripts/                 # 各STT実装
├── database/                # DB関連
├── .env.example             # 環境変数サンプル
├── pyproject.toml           # 依存関係
└── run_app.sh               # 起動スクリプト
```

## トラブルシューティング

### よくある問題と対策

| 問題 | 対策 |
|------|------|
| APIキーエラー | 選択モデルの環境変数を確認 |
| .env変更が反映されない | ページリロードまたはサイドバーで手動再読み込み |
| モジュールエラー | `uv sync`で依存関係を再インストール |
| 音声処理失敗 | ファイル形式とサイズ制限を確認 |

### デバッグモード
サイドバーの「デバッグ設定」で有効化。`logs/`ディレクトリにログ出力:
- `streamlit_app.log`: アプリ全体
- `elevenlabs_debug.log`: ElevenLabs詳細

### 環境変数の確認
サイドバーの「環境変数の設定状況」で現在の設定を確認可能

## 開発

```bash
# パッケージ追加/削除
uv add package-name
uv remove package-name

# 依存関係更新
uv lock --upgrade
```

## デプロイ

### Streamlit Cloud
1. GitHubリポジトリを接続
2. Main file path: `src/app.py` を指定
3. Secretsに `DATABASE_URL` / `OPENAI_API_KEY` など環境変数を設定

詳細は`DEPLOY.md`参照。

### その他のプラットフォーム
Render（Docker）、Railway（PostgreSQL併用）、Heroku（Procfile追加）に対応。

## 重要な注意事項
- **import-instruction-reminders**: 要求されたことのみ実行
- **既存ファイル優先**: 新規作成より既存ファイル編集を優先
- **ドキュメント作成制限**: 明示的に要求されない限り*.mdファイル作成禁止

- RAG機能は Postgres(pgvector) および Turso(libSQL) の双方に対応。
- `DATABASE_URL` がPostgresの場合は初回に `CREATE EXTENSION IF NOT EXISTS vector;` を実行します。
- `.env` では必須の `OPENAI_API_KEY` に加え、必要に応じて `EMBEDDING_MODEL` (既定: text-embedding-3-small), `EMBEDDING_DIM`, `RAG_COMPLETION_MODEL`, `ENABLE_RAG` を設定可能。
- 新規保存分は自動でチャンク化・埋め込み登録。既存データをRAG対応させるには再保存やバックフィルスクリプトが必要。
- Streamlit UIに「💬 QA検索」タブがあり、検索件数スライダーとチャット履歴表示、参照チャンクのスコア/メタ情報の閲覧が可能。
- Supabase関連の機能（Storage・移行ドキュメント等）は削除済みです。
