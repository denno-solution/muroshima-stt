# Agent Notes

- Streamlit製音声文字起こしアプリに Supabase/Postgres+pgvector を使ったRAG機能を追加済み。
- `DATABASE_URL` がPostgresの場合のみ有効化。初回起動時に `CREATE EXTENSION IF NOT EXISTS vector;` とチャンク用テーブルを自動作成。
- `.env` では必須の `OPENAI_API_KEY` に加え、必要に応じて `EMBEDDING_MODEL` (既定: text-embedding-3-small), `EMBEDDING_DIM`, `RAG_COMPLETION_MODEL`, `ENABLE_RAG` を設定可能。
- 新規保存分は自動でチャンク化・埋め込み登録。既存データをRAG対応させるには再保存やバックフィルスクリプトが必要。
- Streamlit UIに「💬 QA検索」タブがあり、検索件数スライダーとチャット履歴表示、参照チャンクのスコア/メタ情報の閲覧が可能。
- Supabase Storage アップロード用に `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY` (または `SUPABASE_ANON_KEY`), `SUPABASE_STORAGE_BUCKET` を参照。
