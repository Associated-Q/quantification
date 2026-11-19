"""检索重排实验：对比「混合检索 top-5」与「混合检索 top-20 + cross-encoder 重排」。

只做测量，不改变线上默认行为。用法：python rerank_eval.py [模型名]
"""
import os
import sys
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import search as S  # noqa: E402
from cases import CASES, label_of_parent  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "BAAI/bge-reranker-base"


def book_subset():
    return [c for c in CASES if c[1] <= {f"b{i}" for i in range(1, 11)} | {"kelly"}]


def qa_subset():
    return [c for c in CASES if all(str(x).startswith(("qa#", "comment#")) for x in c[1])]


def rank_of(hits, ok):
    for i, h in enumerate(hits[:5], 1):
        if label_of_parent(h["parent"]) in ok:
            return i
    return None


def score(rows, key):
    hit = sum(1 for r in rows if r[key])
    rr = sum(1.0 / r[key] for r in rows if r[key])
    return hit / len(rows), rr / len(rows)


def main():
    from sentence_transformers import CrossEncoder

    idx = S.Hybrid()
    model = CrossEncoder(MODEL, max_length=384)

    rows = []
    t_rerank = 0.0
    for q, ok, _, _ in CASES:
        base = idx.search(q, 5)
        cands = idx.search(q, 20)
        texts = [c["parent"]["text"][:800] for c in cands]
        t0 = time.time()
        scores = model.predict([(q, t) for t in texts], batch_size=8, show_progress_bar=False)
        t_rerank += time.time() - t0
        order = sorted(range(len(cands)), key=lambda i: -scores[i])
        reranked = [cands[i] for i in order]
        rows.append({"q": q, "ok": ok,
                     "base_rank": rank_of(base, ok),
                     "rerank_rank": rank_of(reranked, ok)})

    lines = [f"模型 {MODEL} | 评测 {len(CASES)} 条"]
    for name, subset in (("仅书籍", book_subset()), ("仅问答+评论", qa_subset()), ("全体", CASES)):
        idx_set = {id(c) for c in subset}
        sub = [r for r, c in zip(rows, CASES) if id(c) in idx_set]
        rb, mb = score(sub, "base_rank")
        rr, mr = score(sub, "rerank_rank")
        lines.append(f"{name:10} N={len(sub):2}  混合 {rb:.0%}/{mb:.2f} → 重排 {rr:.0%}/{mr:.2f}")
    lines.append(f"平均重排耗时 {t_rerank / len(CASES) * 1000:.0f} ms/次（20 个候选）")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rerank_out.txt")
    open(out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
