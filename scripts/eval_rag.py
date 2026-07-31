"""RAG検索のデバッグ・評価CLI。

質問に対する検索計画(モード/日付判定)と、参照される録音の一覧を表示する。
回答生成は行わない(検索品質のみを確認する)。

使い方:
    uv run python scripts/eval_rag.py "7月28日の作業内容は？"
    uv run python scripts/eval_rag.py --batch questions.txt   # 1行1質問
    uv run python scripts/eval_rag.py --from-logs 20          # 実際の質問ログから20件
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models import get_db  # noqa: E402
from services.rag_service import (  # noqa: E402
    AGGREGATE_MAX_DOCS,
    AGGREGATE_MIN_DOC_CHARS,
    CONTEXT_MAX_CHARS,
    CONTEXT_MAX_DOCS,
    HYBRID_DEFAULT_ALPHA,
    RETRIEVAL_K,
    WHOLE_DOC_THRESHOLD,
    get_rag_service,
)
from services.rag.context_builder import build_context_docs  # noqa: E402
from services.rag.search_service import SearchFilters  # noqa: E402


def evaluate(rag, db, query: str, sources=("audio",)) -> None:
    plan = rag.plan_query(query, sources=sources)
    print(f"\n=== {query}")
    date_desc = (
        f"{plan.date_range.start}〜{plan.date_range.end} ({plan.date_range.kind})"
        if plan.date_range
        else "なし"
    )
    print(
        f"    モード: {plan.mode}{' (集約)' if plan.aggregate else ''}"
        f" / 日付判定: {date_desc}"
    )
    if plan.match_query:
        print(f"    FTS: {plan.match_query[:100]}")

    filters = SearchFilters(
        date_from=plan.date_range.start if plan.date_range else None,
        date_to=plan.date_range.end if plan.date_range else None,
        sources=plan.sources,
    )
    n_docs = (
        AGGREGATE_MAX_DOCS
        if (plan.mode == "browse" or plan.aggregate)
        else CONTEXT_MAX_DOCS
    )
    per_doc_cap = (
        max(AGGREGATE_MIN_DOC_CHARS, CONTEXT_MAX_CHARS // n_docs)
        if n_docs > CONTEXT_MAX_DOCS
        else None
    )
    if plan.mode == "browse":
        hits = rag.search.browse_recent(db, filters, max_recordings=n_docs + 2)
    else:
        qvecs = rag._embed_texts([plan.retrieval_text])
        hits = rag.search.hybrid_search(
            db,
            plan.retrieval_text,
            qvecs[0] if qvecs else None,
            filters,
            RETRIEVAL_K,
            HYBRID_DEFAULT_ALPHA,
            match_query=plan.match_query,
        )
    docs = build_context_docs(
        db,
        hits,
        max_docs=n_docs,
        max_chars=CONTEXT_MAX_CHARS,
        whole_doc_threshold=WHOLE_DOC_THRESHOLD,
        order="date" if plan.mode == "browse" else "score",
        per_doc_cap=per_doc_cap,
    )
    for d in docs:
        kind = "全文" if d.is_full_text else "抜粋"
        print(
            f"    [#{d.n}] {d.recorded_date} {d.title[:60]} "
            f"(score={d.score:.2f}, {kind}{len(d.text)}文字, hits={d.hit_count})"
        )
    if not docs:
        print("    (該当なし)")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG検索の評価")
    parser.add_argument("query", nargs="?", help="質問文")
    parser.add_argument("--batch", help="1行1質問のテキストファイル")
    parser.add_argument("--from-logs", type=int, metavar="N", help="質問ログから直近N件を評価")
    parser.add_argument("--sources", default="audio", help="audio / ceo / audio,ceo")
    args = parser.parse_args()

    rag = get_rag_service()
    if not rag.enabled:
        raise SystemExit("RAGが有効化されていません。")
    sources = tuple(s.strip() for s in args.sources.split(",") if s.strip())

    queries = []
    if args.query:
        queries.append(args.query)
    if args.batch:
        queries.extend(
            line.strip() for line in Path(args.batch).read_text().splitlines() if line.strip()
        )
    db = next(get_db())
    try:
        if args.from_logs:
            from sqlalchemy import text as sql_text

            rows = db.execute(
                sql_text(
                    "SELECT DISTINCT user_text FROM rag_chat_logs ORDER BY id DESC LIMIT :n"
                ),
                {"n": args.from_logs},
            ).fetchall()
            queries.extend(r[0] for r in rows)
        if not queries:
            parser.error("質問を指定してください（引数 / --batch / --from-logs）")
        for q in queries:
            evaluate(rag, db, q, sources)
    finally:
        db.close()


if __name__ == "__main__":
    main()
