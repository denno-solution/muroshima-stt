# Supabase セットアップ手順

## 1. データベース接続URLの取得

1. Supabaseダッシュボードにログイン
2. プロジェクトを選択
3. 左メニューの「Settings」→「Database」を開く
4. 「Connection string」セクションで「URI」タブを選択
5. 接続文字列をコピー（パスワード部分は自分で設定したものに置き換え）

例：
```
postgresql://postgres.[project-ref]:[password]@aws-0-[region].pooler.supabase.com:6543/postgres
```

## 2. テーブルの作成

Supabaseの「SQL Editor」で以下のSQLを実行：

```sql
-- audio_transcriptionsテーブルを作成
CREATE TABLE audio_transcriptions (
    "音声ID" SERIAL PRIMARY KEY,
    "音声ファイルpath" VARCHAR(500) NOT NULL,
    "発言人数" INTEGER DEFAULT 1,
    "録音時刻" TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "録音時間" FLOAT NOT NULL,
    "文字起こしテキスト" TEXT NOT NULL,
    "構造化データ" JSONB,
    "タグ" VARCHAR(200)
);

-- インデックスを作成（検索性能向上）
CREATE INDEX idx_recording_time ON audio_transcriptions("録音時刻");
CREATE INDEX idx_tags ON audio_transcriptions("タグ");
```

## 3. Row Level Security (RLS) の設定

本番環境では、RLSを有効にすることを推奨：

```sql
-- RLSを有効化
ALTER TABLE audio_transcriptions ENABLE ROW LEVEL SECURITY;

-- 認証なしでの読み書きを許可（開発用）
-- 本番環境では適切なポリシーを設定してください
CREATE POLICY "Enable all operations" ON audio_transcriptions
    FOR ALL USING (true) WITH CHECK (true);
```

## 4. 接続テスト

ローカルで接続をテスト：

```python
import os
from sqlalchemy import create_engine

# .envファイルに追加
# DATABASE_URL=postgresql://postgres.[project-ref]:[password]@aws-0-[region].pooler.supabase.com:6543/postgres

engine = create_engine(os.getenv('DATABASE_URL'))
conn = engine.connect()
print("接続成功！")
conn.close()
```

## 注意事項

- Supabaseの無料プランは500MBまで
- 接続プーリングを使用（URLに`pooler`が含まれる）
- SSL接続がデフォルトで有効