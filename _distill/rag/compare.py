"""A/B 对比：BM25 / 向量 / Hybrid(RRF)，同一套评测集，输出分层结果。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import search as S
from cases import CASES, label_of_parent, stratum_of

HERE = os.path.dirname(os.path.abspath(__file__))
EMB = os.path.join(HERE, "embeddings.npy")
MODEL = os.environ.get("EMB_MODEL", "BAAI/bge-small-zh-v1.5")


def rank_of(hits, ok, k=5):
    labels = [label_of_parent(h["parent"]) for h in hits[:k]]
    return next((i + 1 for i, label in enumerate(labels) if label in ok), None)


def aggregate(rows, key, stratum="all"):
    subset = rows if stratum == "all" else [r for r in rows if r["stratum"] == stratum]
    n = len(subset)
    hit = sum(1 for r in subset if r[key])
    rr = sum(1.0 / r[key] for r in subset if r[key])
    return n, hit, (hit / n if n else 0.0), (rr / n if n else 0.0)


def main():
    hybrid = S.Hybrid(emb_path=EMB, model_name=MODEL)
    idx = hybrid.idx

    rows = []
    for q, ok, _, _ in CASES:
        bm25_hits = idx.search(q, 5)
        a, b = hybrid.rankings(q, 20)
        vector_hits = b[:5]
        hybrid_hits = hybrid.fuse(a, b, top_k=5, rrf_k=60)
        rows.append({
            "q": q,
            "ok": ok,
            "stratum": stratum_of(ok),
            "bm25_rank": rank_of(bm25_hits, ok),
            "vector_rank": rank_of(vector_hits, ok),
            "hybrid_rank": rank_of(hybrid_hits, ok),
        })

    lines = [f"模型 {MODEL} | 评测 {len(CASES)} 条 | 子块 {len(idx.children)}"]
    for key, name in (("bm25_rank", "BM25"), ("vector_rank", "Vector"),
                      ("hybrid_rank", "Hybrid(RRF k=60)")):
        parts = []
        for stratum, title in (("book", "书籍"), ("qa_comment", "qa+comment"), ("all", "全体")):
            n, hit, recall, mrr = aggregate(rows, key, stratum)
            parts.append(f"{title} {hit}/{n} R@5={recall:.0%} MRR={mrr:.2f}")
        lines.append(f"{name:18} " + " | ".join(parts))

    with open(os.path.join(HERE, "compare_out.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("compare done -> compare_out.txt")


if __name__ == "__main__":
    main()
