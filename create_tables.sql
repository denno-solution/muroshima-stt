-- Supabase用のテーブル作成SQL
-- このファイルの内容をSupabaseのSQL Editorで実行してください

-- 既存のテーブルがある場合は削除（注意：データが消えます）
-- DROP TABLE IF EXISTS audio_transcriptions;

-- audio_transcriptionsテーブルを作成
CREATE TABLE IF NOT EXISTS audio_transcriptions (
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
CREATE INDEX IF NOT EXISTS idx_recording_time ON audio_transcriptions("録音時刻");
CREATE INDEX IF NOT EXISTS idx_tags ON audio_transcriptions("タグ");

-- Row Level Security (RLS) の設定
ALTER TABLE audio_transcriptions ENABLE ROW LEVEL SECURITY;

-- 開発用：すべての操作を許可
-- 本番環境では適切な認証ポリシーを設定してください
CREATE POLICY "Enable all operations" ON audio_transcriptions
    FOR ALL USING (true) WITH CHECK (true);

-- テーブル情報を確認
SELECT 
    column_name, 
    data_type, 
    is_nullable,
    column_default
FROM 
    information_schema.columns 
WHERE 
    table_name = 'audio_transcriptions'
ORDER BY 
    ordinal_position;