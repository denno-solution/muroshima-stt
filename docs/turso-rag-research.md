もちろん！Python（Streamlit）× Turso（libSQL）の**“拡張いらず・純SQLだけで完結するRAG”**の最小構成をまとめました。最後に参考資料も付けています。

---

# 全体像（Python版）

1. ドキュメントを分割
2. 埋め込みを作成（OpenAIなど）
3. Tursoに `TEXT + F32_BLOB(d)` を保存
4. `libsql_vector_idx()` で**ベクター索引**作成
5. 質問をEmbedding
6. `vector_top_k()` で近傍を取得（必要なら `vector_distance_cos()` で距離）
7. 取り出した本文をプロンプトに詰めて生成

> libSQL/Turso はベクター型（`F32_BLOB(N)`など）・変換関数（`vector32()`）・距離関数（`vector_distance_cos`）・索引（`libsql_vector_idx`）・検索（`vector_top_k`）を**標準で提供**します。拡張の有効化は不要です。([docs.turso.tech][1])

---

# 必要パッケージ

```bash
pip install streamlit libsql openai pypdf
```

* Python SDK（`libsql`）は公式の最新系です。**埋め込みレプリカ**で手元のSQLiteファイルに同期しつつ、クラウドのTursoに書き込み/読み取りできます。([docs.turso.tech][2])

---

# 環境変数（.env 等）

* `TURSO_DATABASE_URL` / `TURSO_AUTH_TOKEN`（Tursoのクイックスタートに沿って発行）([docs.turso.tech][2])
* `OPENAI_API_KEY`（OpenAI Embeddingsで使用）([OpenAI プラットフォーム][3])

---

# スキーマの考え方

* ベクター列は `F32_BLOB(<次元数>)`。例えば OpenAIの `text-embedding-3-small` を使うなら 1536 次元で列を作ります（※他モデルなら数を合わせる）。
* ベクター索引は `CREATE INDEX ... ON <table>(libsql_vector_idx(embedding))`。検索は `vector_top_k('index_name', vector32(?), k)` で行い、返るIDは**ROWIDまたは単一PK**です（JOINで本文取得）。([docs.turso.tech][1])

---

# これだけで動くサンプル（`app.py`）

> 1ファイルで「取り込み（ingest）＋検索＆回答（ask）」が動きます。デフォルトで OpenAI の `text-embedding-3-small`（1536次元）を仮定してスキーマを作ります。

```python
import os, json, uuid, textwrap
import streamlit as st
import libsql
from pypdf import PdfReader
from openai import OpenAI

# ========= 設定 =========
EMBED_MODEL = "text-embedding-3-small"   # 例：1536次元
EMBED_DIMS  = 1536                       # 上のモデルに合わせる
TOP_K_DEFAULT = 5

# ========= 接続 =========
@st.cache_resource
def get_conn():
    url = os.getenv("TURSO_DATABASE_URL")
    token = os.getenv("TURSO_AUTH_TOKEN")
    if not url or not token:
        st.stop()
    # 埋め込みレプリカで接続し、初回に同期
    conn = libsql.connect("rag.db", sync_url=url, auth_token=token)
    conn.sync()
    return conn

def ensure_schema(conn):
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS documents (
          id        INTEGER PRIMARY KEY,        -- ROWIDベース（vector_top_kとJOINしやすい）
          doc_id    TEXT UNIQUE,                -- 論理ID
          content   TEXT NOT NULL,
          metadata  TEXT,
          embedding F32_BLOB({EMBED_DIMS}) NOT NULL
        );
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS documents_embedding_idx
          ON documents (libsql_vector_idx(embedding));
    """)
    conn.commit()

# ========= 埋め込み =========
def get_openai():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.stop()
    return OpenAI(api_key=api_key)

def embed_texts(client: OpenAI, texts):
    # OpenAI Embeddings API（配列でバッチ可）
    res = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in res.data]  # list[list[float]]
# 参考: OpenAIのEmbeddingsガイドとAPIリファレンス :contentReference[oaicite:5]{index=5}

# ========= ユーティリティ =========
def chunk_text(text: str, chunk_size=800, overlap=100):
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]

def read_pdf(file):
    reader = PdfReader(file)
    return "\n".join([p.extract_text() or "" for p in reader.pages])

def upsert_chunks(conn, chunks, metas, embeddings):
    for c, meta, vec in zip(chunks, metas, embeddings):
        # vector32(?) は JSON配列文字列を受け取り、F32_BLOB に変換
        conn.execute(
            "INSERT OR REPLACE INTO documents (doc_id, content, metadata, embedding) "
            "VALUES (?, ?, ?, vector32(?));",
            (str(uuid.uuid4()), c, json.dumps(meta or {}), json.dumps(vec)),
        )
    conn.commit()

def search(conn, qvec, top_k=TOP_K_DEFAULT):
    # vector_top_k() は索引名を受け取り、近傍のROWID/PKを返す
    sql = """
        SELECT d.id, d.doc_id, d.content, d.metadata,
               vector_distance_cos(d.embedding, vector32(?)) AS distance
        FROM vector_top_k('documents_embedding_idx', vector32(?), ?)
        JOIN documents d ON d.id = id
        ORDER BY distance ASC
    """
    cur = conn.execute(sql, (json.dumps(qvec), json.dumps(qvec), top_k))
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]

# ========= Streamlit UI =========
st.set_page_config(page_title="Turso + Streamlit RAG", page_icon="🤖")
st.title("Turso + Streamlit: ネイティブベクター検索でRAG")

with st.sidebar:
    st.markdown("### 接続状態")
    st.write("`TURSO_DATABASE_URL` と `TURSO_AUTH_TOKEN` を環境変数に設定してください。")
    conn = get_conn()
    ensure_schema(conn)
    st.success("Tursoに接続 & スキーマOK")

st.header("1) 取り込み（Ingest）")
tab1, tab2 = st.tabs(["テキスト", "PDF"])

with tab1:
    raw = st.text_area("テキストを貼り付け", height=180, placeholder="ここに本文を入れてください")
    csize = st.number_input("チャンクサイズ", 200, 2000, 800, 50)
    ovlp  = st.number_input("オーバーラップ", 0, 400, 100, 10)
    meta  = st.text_input("メタデータ(JSON、任意)", value='{"source":"paste"}')
    if st.button("取り込み実行", type="primary", disabled=not raw):
        chunks = chunk_text(raw, csize, ovlp)
        metas  = [json.loads(meta)] * len(chunks)
        client = get_openai()
        embs   = embed_texts(client, chunks)
        upsert_chunks(conn, chunks, metas, embs)
        st.success(f"チャンク {len(chunks)} 件を保存しました。")

with tab2:
    up = st.file_uploader("PDFを選択", type=["pdf"])
    csize2 = st.number_input("チャンクサイズ(PDF)", 200, 2000, 900, 50, key="pdfc")
    ovlp2  = st.number_input("オーバーラップ(PDF)", 0, 400, 120, 10, key="pdfo")
    if st.button("PDFを取り込み", disabled=not up):
        text = read_pdf(up)
        chunks = chunk_text(text, csize2, ovlp2)
        metas  = [ {"source": getattr(up, "name", "upload")} ] * len(chunks)
        client = get_openai()
        embs   = embed_texts(client, chunks)
        upsert_chunks(conn, chunks, metas, embs)
        st.success(f"PDFから {len(chunks)} チャンクを保存しました。")

st.header("2) 質問（Search + Generate）")
q = st.text_input("質問を入力")
k = st.slider("Top-K", 1, 20, TOP_K_DEFAULT)
if st.button("質問する", type="primary", disabled=not q):
    client = get_openai()
    qvec = embed_texts(client, [q])[0]
    hits = search(conn, qvec, k)

    # コンテキストを組み立てて回答生成
    context = "\n\n---\n\n".join(
        [textwrap.shorten(h["content"], width=1200, placeholder=" …") for h in hits]
    )
    sys = "あなたは与えられたコンテキストの情報だけを根拠に、日本語で簡潔かつ根拠付きで回答します。分からない場合は正直に分からないと言ってください。"
    user = f"質問: {q}\n\n# コンテキスト\n{context}"

    chat = client.chat.completions.create(
        model="gpt-4o-mini",  # お好みで変更
        messages=[{"role": "system", "content": sys},
                  {"role": "user",   "content": user}],
        temperature=0.2,
    )
    ans = chat.choices[0].message.content
    st.subheader("回答")
    st.write(ans)

    st.subheader("参照チャンク")
    for i, h in enumerate(hits, 1):
        md = json.loads(h.get("metadata") or "{}")
        with st.expander(f"{i}. distance={h['distance']:.4f}  source={md.get('source','') }"):
            st.write(h["content"])
```

---

## 使い方

```bash
export TURSO_DATABASE_URL="..."
export TURSO_AUTH_TOKEN="..."
export OPENAI_API_KEY="..."

streamlit run app.py
```

* 初回アクセスでテーブルと索引を自動作成します（`ROWID`ベースの `INTEGER PRIMARY KEY` を採用しているので、`vector_top_k()` の返すIDとJOINしやすいです）。([docs.turso.tech][1])
* 別モデルを使う場合は `EMBED_DIMS` を**必ず**合わせてください（例：`F32_BLOB(3072)` など）。([docs.turso.tech][1])

---

## 実装のポイント／よくあるハマり

* **索引必須**：データが増える前に `libsql_vector_idx(embedding)` を作っておくと速いです。([docs.turso.tech][1])
* **`vector32(?)` への渡し方**：Python からは **JSON配列文字列**（`json.dumps(list[float])`）をバインドすればOK。([docs.turso.tech][1])
* **距離の見方**：`vector_distance_cos` は「1 − コサイン類似度」。0に近いほど近い。([docs.turso.tech][1])
* **制限**：次元数上限（≤ 65536）、`FLOAT1BIT` ではEuclid不可など。([docs.turso.tech][1])
* **接続**：Python公式SDKの `libsql.connect("local.db", sync_url=..., auth_token=...)` → `conn.sync()` が一番シンプル。([docs.turso.tech][2])
* **インデックスのクエリ**：`vector_top_k('index_name', vector32(?), k)` の返りは **ROWIDまたは単一PRIMARY KEY**。本サンプルはROWID型（`INTEGER PRIMARY KEY`）で素直にJOINしています。([docs.turso.tech][1])
* **高度化**：`libsql_vector_idx(..., 'metric=l2', 'compress_neighbors=float8')` など索引パラメータ調整も可能。([docs.turso.tech][1])

---

## 参考資料（公式中心）

* **AI & Embeddings（libSQL/Tursoのベクター型・関数・索引・`vector_top_k`）**：型 `F32_BLOB(d)`、`vector32()`、`libsql_vector_idx`、`vector_top_k` の定義と使い方。([docs.turso.tech][1])
* **Turso Python Quickstart（`pip install libsql`／埋め込みレプリカの接続手順）**：`libsql.connect(..., sync_url, auth_token)` → `conn.sync()`。([docs.turso.tech][2])
* **Turso Quickstart（CLIでDB作成・トークン発行）**：URL/Tokenの取得手順。([docs.turso.tech][4])
* **ベクター検索の導入背景・DiskANN**（libSQL実装の概要ブログ）([Turso][5])
* **OpenAI Embeddings ガイド**（Pythonからの埋め込み生成の基本）([OpenAI プラットフォーム][3])

---

必要なら、このサンプルを**「PDF一括取り込み」「メタデータ検索（タグ/日付フィルタ）」「FTS5×ベクターのハイブリッド」**まで拡張した版に仕上げて出します。どのデータソースを使うかだけ教えてください。

[1]: https://docs.turso.tech/features/ai-and-embeddings "AI & Embeddings - Turso"
[2]: https://docs.turso.tech/sdk/python "Turso Quickstart (Python) - Turso"
[3]: https://platform.openai.com/docs/guides/embeddings?utm_source=chatgpt.com "Vector embeddings - OpenAI API"
[4]: https://docs.turso.tech/quickstart?utm_source=chatgpt.com "Quickstart"
[5]: https://turso.tech/blog/approximate-nearest-neighbor-search-with-diskann-in-libsql?utm_source=chatgpt.com "Approximate nearest neighbor search with DiskANN in libSQL - Turso"
