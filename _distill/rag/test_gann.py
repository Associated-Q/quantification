"""江恩书籍的检索冒烟测试：确认新入库的内容能被命中。

只检查「命中的书名是否在可接受集合内」，不做召回率统计。
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import search as S

CASES = [
    ("江恩角度线怎么画", {"b8"}),
    ("时间与价位的正方形", {"b6", "b8"}),
    ("江恩六角形如何应用", {"b8"}),
    ("周天公度是什么", {"b6"}),
    ("螺旋四方形怎么制作", {"b8"}),
    ("江恩选股的核心原则", {"b10"}),
    ("波动率 时间周期 八分法", {"b6"}),
    ("江恩对期货和股票的看法", {"b6", "b9", "b10"}),
]

idx = S.Index()
hit = 0
for q, ok in CASES:
    hits = idx.search(q, 5)
    def tag(h):
        p = h["parent"]
        m = re.match(r"(b\d+)", p.get("book") or "")
        if m:
            return m.group(1)
        if "kelly" in str(p.get("parent_id", "")):
            return "kelly"
        return p.get("book") or "?"
    books = [tag(h) for h in hits]
    rank = next((i + 1 for i, b in enumerate(books) if b in ok), None)
    hit += 1 if rank else 0
    print(f"{'OK ' if rank else 'MISS'} {q:20} 期望{sorted(ok)} 实际{books}")
print(f"\n命中 {hit}/{len(CASES)}")
