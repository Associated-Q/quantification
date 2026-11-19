"""检索评测：对比 BM25 与 BM25+LLM重排，统计 Recall@5 与 MRR。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import search as S

CASES = [
    ("看涨吞没形态的判定条件", "b1"),
    ("黄昏之星形态如何识别", "b1"),
    ("锤子线的含义", "b1"),
    ("道氏理论的三种趋势", "b3"),
    ("支撑线和阻力线怎么画", "b2"),
    ("头肩顶形态", "b2"),
    ("移动平均线的用法", "b2"),
    ("成交量与趋势的确认", "b2"),
    ("艾略特波浪的三条铁律", "b5"),
    ("斐波那契回撤比例", "b5"),
    ("平台形调整浪", "b5"),
]


def score(books, want, k=5):
    rank = next((i + 1 for i, b in enumerate(books[:k]) if b.startswith(want)), None)
    return (1, 1 / rank) if rank else (0, 0.0)


idx = S.Index()
res = {"bm25": [0, 0.0], "rerank": [0, 0.0]}
statuses = set()
for q, want in CASES:
    cands = idx.search(q, 20)
    base = [h["parent"]["book"] for h in cands]
    h1, st = S.llm_rerank(q, cands, top_k=5)
    statuses.add(st)
    rer = [h["parent"]["book"] for h in h1]
    for key, books in (("bm25", base), ("rerank", rer)):
        h, r = score(books, want)
        res[key][0] += h
        res[key][1] += r
    print(f"{q:22} want {want} | bm25 {base[:3]} | rerank {rer[:3]}")

n = len(CASES)
for key, (hit, rr) in res.items():
    print(f"{key:7} Recall@5 = {hit}/{n} = {hit / n:.0%} | MRR = {rr / n:.2f}")
print("rerank status:", statuses)
