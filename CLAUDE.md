# 音声文字起こしWebアプリ（STT Web App）

Streamlitを使用した音声文字起こしWebアプリ。複数のSTTモデルに対応し、Gemini Flash 2.5-liteによる自動構造化機能を搭載。

## 運用メモ
- 本番デプロイ先: Streamlit Community Cloud
- Tauri版（デスクトップアプリ）: `../stt-desktop`（`stt-suite/stt-desktop`。本リポジトリの隣にある別リポジトリ）

## 課題管理

- 顧客要望、不具合、対応方針、ステータスは、[室島精工様_課題管理表](https://docs.google.com/spreadsheets/d/1JHO7Rb57ivusmi8NYmu7iSd51pie-Jxa5Ec8AaFVwuU/edit?gid=0#gid=0)を正本として管理します。
- このリポジトリでは、主にカテゴリ「音声DB: Web」の課題を扱います。
- Windows版の課題は `stt-desktop` リポジトリで扱います。

## 機能

- 複数音声ファイルの同時アップロード/マイク録音
- 5つのSTTモデル対応（OpenAI、Google Cloud、Amazon、Azure、ElevenLabs）
- Gemini Flash 2.5-liteによる文字起こしテキストの自動構造化
- Turso(libSQL)/SQLiteデータベース保存（本番はTursoに完全移行）
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

**Turso(libSQL) 使用（本番・推奨）**:
1. TursoでDBを作成しURL/トークンを取得
2. `.env` の `DATABASE_URL` を `sqlite+libsql://<db>-<org>.turso.io?secure=true&authToken=...` に設定
3. 初回起動時に `audio_transcription_chunks` のベクトル式インデックス（libsql_vector_idx）が自動作成されます

**ローカル開発**: 通常のSQLiteが自動使用されます（RAGは無効）

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
| **データベース** | DATABASE_URL | `sqlite+libsql://...`（Turso） |
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

### データベース利用時のポイント（Turso専用）
- `DATABASE_URL` に `sqlite+libsql://<db名>-<org>.turso.io?secure=true&authToken=...` を設定するとリモートTursoに接続可能
- `audio_transcription_chunks` の `libsql_vector_idx` 作成（アプリが初回自動作成）と `OPENAI_API_KEY` 設定でRAGタブが有効化

### データベーススキーマ（Turso）

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


## 重要な注意事項
- **import-instruction-reminders**: 要求されたことのみ実行
- **既存ファイル優先**: 新規作成より既存ファイル編集を優先
- **ドキュメント作成制限**: 明示的に要求されない限り*.mdファイル作成禁止

- RAG機能は Turso(libSQL) 専用です（Postgres対応は削除）。
- `.env` では必須の `OPENAI_API_KEY` に加え、必要に応じて `EMBEDDING_MODEL` (既定: text-embedding-3-small), `EMBEDDING_DIM`, `RAG_COMPLETION_MODEL`, `ENABLE_RAG` を設定可能。
- 新規保存分は自動でチャンク化・埋め込み登録。既存データをRAG対応させるには再保存やバックフィルスクリプトが必要。
- Streamlit UIのQAチャットは「💬 現場録音に質問」「💬 社長音声に質問」の2タブに分離（検索エンジンは共通、検索対象・会話履歴・プロンプトが別）。
- Supabase関連の機能（Storage・移行ドキュメント等）は削除済みです。

## Agent Notes（RAG開発向けメモ）
- 本リポジトリはデータベースをTurso(libSQL)に完全移行済み。Postgres/pgvector対応はコードから削除済みです。関連依存（psycopg2, pgvector）も`pyproject.toml`から除外しました。
- QAチャット（「現場録音に質問」「社長音声に質問」タブ）のアーキテクチャ:
  - 出口は現場録音用と社長音声用で分離（`ui/tabs/rag_tab.py`の`_ChatProfile`）。会話ログは`rag_chat_logs.chat_kind`（"audio"/"ceo"、NULLは旧データ=audio扱い）で区別
  - 新規保存分は保存時に即時索引化（現場録音: upload_tab/mic_tab、社長音声: ceo_processor）。デスクトップ版等の外部保存分はQAタブ表示時に自動取り込み（20件以下は自動、超過時はボタン表示）
  - `services/rag/search_service.py`: 検索実行層。ベクトル検索（`vector_distance_cos`全走査+SQL日付フィルタ）/ キーワード検索（FTS5）/ 期間ブラウズの3操作。Phase 2（agentic search）ではこれらをLLMのツールとして公開する想定
  - `services/rag/tokenizer.py`: FTS5用の文字バイグラムトークナイザ。索引テーブルは`rag_fts_audio`/`rag_fts_ceo`（Python側で行を管理、トリガ無し）。現場用語・型番の完全一致検索を辞書非依存で保証
  - `services/rag/date_utils.py`: クエリからの日付範囲抽出。検索は正規化済み`recorded_date`列（JST, YYYY-MM-DD）へのSQL WHEREで行う（事後フィルタ禁止）
  - `services/rag/context_builder.py`: コンテキストは録音単位（短い録音は全文、長い録音はヒット周辺の結合）。チャンク断片をそのまま渡さない
  - `services/rag/reconcile.py` + `RAGService.reconcile()`: デスクトップ版保存分・社長音声の索引差分を補完。UIから自動/ボタン実行、CLIは`scripts/backfill_rag.py`
- ハイブリッド検索の融合は重み付きRRF（Reciprocal Rank Fusion）。スコアの絶対値でのブレンドはキャリブレーション問題があるため禁止
- `created_at`はWeb版（naive UTC）とデスクトップ版（RFC3339 UTC）で形式が混在。日付判定には必ず`recorded_date`を使う
- QA検索タブの回答生成は「ストリーミングのみ」です。非ストリーミングAPIはコードから撤去済みです。
- 既定のRAGモデル: `EMBEDDING_MODEL=text-embedding-3-small (1536次元)`, `RAG_COMPLETION_MODEL=gpt-5.6-luna`。Responses APIを使用。
- `EMBEDDING_DIM` を変更する場合はDB列定義が固定のため、再作成（既存チャンク削除→再インデックス）が必要。
- 検索品質の確認は `uv run python scripts/eval_rag.py "質問"`（検索計画と参照録音を表示。生成なし）。
