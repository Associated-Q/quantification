"""检索回归测试：用固定评测集守住召回率下限。

阈值取得比当前实测略低一点，允许小幅波动，但明显退化会立刻失败。
"""
import re

import pytest

import search as S
from cases import CASES, label_of_parent

# 只在书里出现过的专有说法，用于确认新入库的书籍能被检索到
GANN_CASES = [
    ("江恩角度线怎么画", {"b8"}),
    ("时间与价位的正方形", {"b6", "b8"}),
    ("江恩六角形如何应用", {"b8"}),
    ("周天公度是什么", {"b6"}),
    ("螺旋四方形怎么制作", {"b8"}),
    ("江恩选股的核心原则", {"b10"}),
    ("波动率 时间周期 八分法", {"b6"}),
    ("江恩对期货和股票的看法", {"b6", "b9", "b10"}),
]


@pytest.fixture(scope="module")
def hybrid():
    return S.Hybrid()


def book_tag(parent):
    m = re.match(r"(b\d+)", parent.get("book") or "")
    if m:
        return m.group(1)
    if "kelly" in str(parent.get("parent_id", "")):
        return "kelly"
    return parent.get("source", "?")


def rank_of(hits, ok, k=5):
    for i, h in enumerate(hits[:k], 1):
        if label_of_parent(h["parent"]) in ok:
            return i
    return None


def test_book_subset_recall(hybrid):
    cases = [c for c in CASES if c[1] <= {f"b{i}" for i in range(1, 11)} | {"kelly"}]
    hit = sum(1 for q, ok, _, _ in cases if rank_of(hybrid.search(q, 5), ok))
    recall = hit / len(cases)
    assert recall >= 0.95, f"书籍召回率跌到 {recall:.0%}（{hit}/{len(cases)}）"


def test_qa_comment_recall(hybrid):
    cases = [c for c in CASES if all(str(x).startswith(("qa#", "comment#")) for x in c[1])]
    hit = sum(1 for q, ok, _, _ in cases if rank_of(hybrid.search(q, 5), ok))
    recall = hit / len(cases)
    assert recall >= 0.60, f"问答/评论召回率跌到 {recall:.0%}（{hit}/{len(cases)}）"


def test_gann_books_are_retrievable(hybrid):
    hit = 0
    misses = []
    for q, ok in GANN_CASES:
        hits = hybrid.search(q, 5, sources=["book"])
        books = [book_tag(h["parent"]) for h in hits]
        if any(b in ok for b in books):
            hit += 1
        else:
            misses.append((q, books))
    assert hit >= 7, f"江恩查询只命中 {hit}/8，未命中：{misses}"


def test_source_filter_returns_only_books(hybrid):
    hits = hybrid.search("仓位控制", 5, sources=["book"])
    assert hits, "应至少返回一条"
    for h in hits:
        assert h["parent"].get("source", "book") == "book"


def test_unknown_query_returns_something_or_nothing_but_never_raises(hybrid):
    hits = hybrid.search("zzzz不存在的词zzzz", 3)
    assert isinstance(hits, list)
