# 音声文字起こしWebアプリ（STT Web App）

Streamlitを使用した音声文字起こしWebアプリケーションです。複数のSTTモデルに対応し、Gemini Flash 2.0による自動構造化機能を備えています。

## 機能

- 複数の音声ファイルの同時アップロード
- マイクからの直接録音機能
- 複数のSTTモデルから選択可能（OpenAI、Google Cloud、Amazon、Azure、ElevenLabs）
  - デフォルト: ElevenLabs
- Gemini Flash 2.5-liteによる文字起こしテキストの自動構造化（thinking mode + structured output対応）
- PostgreSQL/SQLiteデータベースへの結果保存
- 処理結果の閲覧と検索
- Basic認証によるアクセス制限（オプション）
- 処理中のUI制御（ボタン無効化など）

## クイックスタート

### 1. セットアップ

#### uvのインストール（未インストールの場合）
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### プロジェクトのセットアップ
```bash
# 依存関係をインストール
uv sync

# .envファイルを編集
# 最低限、使いたいSTTモデルのAPIキーを設定
nano .env
```

#### データベースのセットアップ
本アプリは**PostgreSQL**（Supabase）を使用します。ローカル開発用のSQLiteも利用可能です。

##### Supabaseを使用する場合（推奨）
1. [Supabase](https://supabase.com)でプロジェクトを作成
2. `database/SUPABASE_SETUP.md`の手順に従ってテーブルを作成
3. `.env`ファイルにDATABASE_URLを設定：
```env
DATABASE_URL=postgresql://postgres.[project-ref]:[password]@aws-0-[region].pooler.supabase.com:6543/postgres
```

### 2. 必要なAPIキー

最低限必要なもの：
- **STTモデル用**: OpenAI、Google Cloud、Amazon、Azure、ElevenLabsのいずれか1つ
- **構造化用**: GEMINI_API_KEY（Gemini Flash 2.5-lite用）

### 3. アプリ起動

```bash
./run_app.sh
# または
uv run streamlit run src/app.py
```

### 4. 使い方

1. ブラウザで http://localhost:8501 を開く
2. サイドバーでSTTモデルを選択（デフォルト: ElevenLabs）
3. 以下のいずれかの方法で音声を入力：
   - **「📤 アップロード」タブ**: 音声ファイルをアップロード（複数可）→「文字起こし開始」をクリック
   - **「🎙️ マイク録音」タブ**: マイクで録音 → 「文字起こしてデータベースに保存しますか？」をクリック
4. 「📊 処理結果」タブで結果を確認
5. 「🗄️ データベース」タブで過去の処理結果を検索・閲覧

## 詳細セットアップ

### 環境変数の設定

#### .envファイルを使用する方法（推奨）

1. `.env.example`をコピーして`.env`ファイルを作成：
```bash
cp .env.example .env
```

2. `.env`ファイルを編集して、必要なAPIキーを設定：
```bash
# テキストエディタで.envファイルを開く
nano .env  # またはお好みのエディタを使用
```

#### サンプル.env設定（ElevenLabsを使う場合）

```env
# データベース（Supabase使用時）
DATABASE_URL=postgresql://postgres.[project-ref]:[password]@aws-0-[region].pooler.supabase.com:6543/postgres

# STTモデル（デフォルトはElevenLabs）
ELEVENLABS_API_KEY=xi-xxxxxxxxxxxxxxxxxxxxx
# 他のSTTモデルを使う場合は該当するAPIキーを設定
# OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxx

# 構造化機能用
GEMINI_API_KEY=AIzaSyxxxxxxxxxxxxxxxxxxxxx

# Basic認証（オプション）
BASIC_AUTH_USERNAME=admin
BASIC_AUTH_PASSWORD=your-secure-password
```

#### 環境変数を直接設定する方法

使用するSTTモデルに応じて、以下の環境変数を設定してください：

##### OpenAI
```bash
export OPENAI_API_KEY="your-api-key"
```

##### Google Cloud (Chirp)
```bash
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_APPLICATION_CREDENTIALS="path/to/credentials.json"
```

##### Amazon Transcribe
```bash
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
```

##### Azure Speech
```bash
export AZURE_SPEECH_KEY="your-speech-key"
export AZURE_SPEECH_REGION="your-region"
```

##### ElevenLabs
```bash
export ELEVENLABS_API_KEY="your-api-key"
```

##### Gemini (構造化機能用)
```bash
export GEMINI_API_KEY="your-api-key"
# または
export GOOGLE_AI_API_KEY="your-api-key"
```

## 使用方法詳細

1. アプリを起動：
```bash
./run_app.sh
# または
uv run streamlit run src/app.py
```

2. ブラウザでアプリが開きます（通常は http://localhost:8501）

3. サイドバーでSTTモデルを選択（デフォルト: ElevenLabs）

4. 音声入力方法を選択：

   ### ファイルアップロード
   - 「📤 アップロード」タブを選択
   - 音声ファイルを選択してアップロード（複数可）
   - 「🚀 文字起こし開始」ボタンをクリック（処理中は自動的に無効化）
   
   ### マイク録音
   - 「🎙️ マイク録音」タブを選択
   - 「🎙️ マイクで録音してください」ボタンをクリック
   - 録音開始 → 話す → 停止ボタンで録音終了
   - 「🚀 文字起こしてデータベースに保存しますか？」ボタンをクリック

5. 「📊 処理結果」タブで結果を確認

6. 「🗄️ データベース」タブで過去の処理結果を検索・閲覧

### 設定の永続化

アプリの設定は自動的に保存され、次回起動時に復元されます：

- **STTモデルの選択**: 最後に選択したモデルが記憶されます
- **構造化機能**: 有効/無効の設定が保存されます
- **デバッグモード**: 有効/無効の設定が保存されます

設定は`.app_settings.json`ファイルに保存されます（gitignore対象）。

### Basic認証

アプリへのアクセスを制限するためのBasic認証機能を搭載しています：

- **設定方法**: 環境変数に`BASIC_AUTH_USERNAME`と`BASIC_AUTH_PASSWORD`を設定
- **動作**: 両方の環境変数が設定されている場合のみ認証が有効化されます
- **ログアウト**: サイドバーにログアウトボタンが表示されます
- **Cookie認証**: ログイン成功時に認証トークンがCookieに保存され、24時間は再認証不要です

```env
# Basic認証を有効化する場合
BASIC_AUTH_USERNAME=admin
BASIC_AUTH_PASSWORD=your-secure-password
```

注意: 本番環境では強力なパスワードを設定してください。

#### Cookie認証の仕組み
- ログイン成功時に安全なランダムトークンを生成
- トークンはCookieに保存（有効期限: 24時間）
- ブラウザをリロードしてもCookieが有効な限り自動ログイン
- ログアウト時にCookieは削除されます

## データベーススキーマ

### PostgreSQL（Supabase）

| カラム名 | 型 | 説明 |
|---------|-----|------|
| 音声ID | SERIAL | 主キー（自動採番） |
| 音声ファイルpath | VARCHAR(500) | ファイル名 |
| 発言人数 | INTEGER | 発言者数（デフォルト: 1） |
| 録音時刻 | TIMESTAMP | 処理時刻 |
| 録音時間 | FLOAT | 音声の長さ（秒） |
| 文字起こしテキスト | TEXT | 文字起こし結果 |
| 構造化データ | JSONB | Geminiによる構造化データ |
| タグ | VARCHAR(200) | 自動生成されたタグ |

### データベース切り替え
環境変数`DATABASE_URL`が設定されている場合は自動的にPostgreSQLを使用します。設定されていない場合はローカルのSQLiteを使用します。

## 対応ファイル形式

- WAV
- MP3
- M4A
- FLAC
- OGG

## uv移行について

プロジェクトをuvで完全に管理するように移行しました。

### 変更内容

1. **pyproject.toml**: すべての依存関係をpyproject.tomlに統合
2. **run_app.sh**: uvを使用するように更新
3. **ドキュメント**: uv対応に更新


### 開発用コマンド

```bash
# 新しいパッケージを追加
uv add package-name

# 開発用パッケージを追加
uv add --dev package-name

# パッケージを削除
uv remove package-name

# 依存関係を更新
uv lock --upgrade
```

## トラブルシューティング

### エラー: APIキーが設定されていません
選択したSTTモデルに必要な環境変数が設定されているか確認してください。

### .envファイルの変更が反映されない場合
アプリには自動リロード機能が実装されています：
1. .envファイルを編集後、ページをリロードすると自動的に環境変数が再読み込みされます
2. サイドバーの「🔧 環境変数の設定状況」で現在の設定を確認できます
3. 「🔄 環境変数を再読み込み」ボタンで手動リロードも可能です

### エラー: モジュールが見つかりません
`uv sync`を実行して、すべての依存関係がインストールされているか確認してください。

### 音声ファイルの処理に失敗する
- ファイル形式が対応しているか確認
- ファイルサイズが大きすぎないか確認（各APIの制限に準拠）
- APIキーの権限が適切か確認

### 文字起こしエラーのデバッグ
アプリには詳細なログ機能が実装されています：

1. **エラー表示の改善**：
   - エラーが発生した場合、画面に詳細なエラーメッセージが表示されます
   - APIのエラーコード（例：422 Unprocessable Entity）も含まれます

2. **デバッグモード**：
   - サイドバーの「🐛 デバッグ設定」で「デバッグモードを有効化」をチェック
   - 「📋 ログファイルを表示」ボタンでログの最新50行を確認可能
   - ログファイルは`logs/`ディレクトリに保存されます：
     - `streamlit_app.log`: アプリ全体のログ
     - `elevenlabs_debug.log`: ElevenLabs専用の詳細ログ

3. **ログに記録される情報**：
   - ファイル情報（サイズ、形式）
   - API呼び出しの各段階
   - エラーの詳細とスタックトレース

## プロジェクト構造

```
stt/
├── src/                    # アプリケーションのソースコード
│   ├── app.py             # Streamlitメインアプリ
│   ├── models.py          # SQLAlchemyデータベースモデル
│   ├── stt_wrapper.py     # STTモデルの統一インターフェース
│   ├── text_structurer.py # Geminiによる構造化処理
│   ├── env_watcher.py     # 環境変数の自動リロード機能
│   ├── app_settings.py    # アプリ設定の永続化
│   └── auth.py            # Basic認証機能
├── scripts/               # 各STTモデルの実装スクリプト
│   ├── transcribe_openai.py
│   ├── transcribe_google.py
│   ├── transcribe_amazon.py
│   ├── transcribe_azure.py
│   └── transcribe_elevenlabs.py
├── database/              # データベース関連
│   ├── create_tables.sql  # テーブル作成SQL
│   ├── SUPABASE_SETUP.md  # Supabaseセットアップ手順
│   └── test_db_connection.py # DB接続テストスクリプト
├── recording/             # 音声録音Webアプリ
│   ├── index-STT.html     # 録音UI
│   ├── index.js           # Node.js Webhookサーバー
│   └── package.json       # 依存関係
├── data/                  # 音声ファイル置き場
├── transcriptions/        # 文字起こし結果
├── logs/                  # ログファイル（gitignore対象）
├── .streamlit/            # Streamlit設定
│   ├── config.toml        # アプリ設定
│   └── secrets.toml.example # 秘密情報設定例
├── .env                   # 環境変数設定（gitignore対象）
├── .env.example           # 環境変数設定のサンプル
├── .app_settings.json     # アプリ設定（gitignore対象）
├── pyproject.toml         # プロジェクト設定と依存関係
├── run_app.sh            # アプリ起動スクリプト
├── DEPLOY.md              # デプロイ手順書
└── CLAUDE.md             # このドキュメント
```

## 各STTサービスの詳細

### 対応サービス

1. **OpenAI** - gpt-4o-transcribe
2. **Google Cloud Speech-to-Text** - Chirp/Chirp 2
3. **Amazon Transcribe**
4. **Microsoft Azure AI Speech**
5. **ElevenLabs** - Scribe v1

### ファイルサイズ制限
- OpenAI: 25MB
- Google Cloud: 10分以下（約10MB）
- Amazon Transcribe: 2GB（S3経由）
- Azure AI Speech: 100MB
- ElevenLabs: 1GB、4.5時間以下

### 対応音声フォーマット
- MP3, MP4, WAV, M4A, FLAC, WEBM など主要なフォーマットに対応
- サービスによって対応フォーマットが異なる場合があります

### コスト
各サービスは使用量に応じて課金されます。料金体系は各サービスの公式ドキュメントを確認してください。

## 旧コマンドライン版の使用方法（参考）

個別のスクリプトを直接実行することも可能です：

```bash
# OpenAI
python scripts/transcribe_openai.py

# Google Cloud (Chirpモデル)
python scripts/transcribe_google.py --model chirp

# Amazon Transcribe
python scripts/transcribe_amazon.py

# Microsoft Azure AI Speech
python scripts/transcribe_azure.py

# ElevenLabs
python scripts/transcribe_elevenlabs.py

# すべてのサービスを一括実行
python transcribe_all.py
```

## デプロイ

### Streamlit Cloudへのデプロイ

詳細は`DEPLOY.md`を参照してください。主な手順：

1. **Supabaseでデータベースを準備**
   - Supabaseプロジェクトを作成
   - SQLエディタで`database/create_tables.sql`を実行
   - 接続URLを取得

2. **Streamlit Cloudでアプリをデプロイ**
   - GitHubリポジトリを接続
   - Main file path: `src/app.py`を指定
   - Secretsに環境変数を設定

3. **必要な環境変数（Secrets）**
   ```toml
   DATABASE_URL = "postgresql://..."
   OPENAI_API_KEY = "sk-..."
   GEMINI_API_KEY = "AIza..."
   # その他使用するSTTモデルのAPIキー
   ```

### その他のデプロイオプション

- **Render**: Dockerfileを使用
- **Railway**: PostgreSQLサービスと併用
- **Heroku**: Procfileを追加して対応

## 音声録音機能（開発中）

`recording/`ディレクトリに音声録音用のWebアプリが含まれています：

- **ブラウザベースの録音機能**: マイク入力をリアルタイムで録音
- **自動音声検出**: 無音を検出して自動的に録音を分割
- **Webhookサーバー**: Google Cloud Speech-to-Textと連携
- **スプレッドシート連携**: 文字起こし結果を自動記録

今後、このStreamlitアプリと統合予定です。