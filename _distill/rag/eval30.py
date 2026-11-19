"""检索评测：BM25 与 Hybrid（BM25+向量+RRF），分层统计 + RRF k 敏感性。

分层：仅书籍子集 / 仅 qa+comment 子集 / 全体，分别输出 Recall@5 与 MRR。
标签先做术语校验：至少一个可接受来源中关键术语出现 >= 2 次，否则告警。
所有中文结果写入 UTF-8 文件（Windows 控制台 GBK，不在控制台直接打印中文）。
"""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import search as S
from cases import BOOK_LABELS, CASES, LEGACY_QUESTIONS, label_of_parent, stratum_of

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
BOOKDIR = os.path.join(ROOT, "_distill", "out")
PARENTS = os.path.join(HERE, "parents.jsonl")
EMB = os.path.join(HERE, "embeddings.npy")
MODEL = os.environ.get("EMB_MODEL", "BAAI/bge-small-zh-v1.5")
RRF_KS = (20, 60, 100)


def load_book_texts():
    texts = {}
    if not os.path.isdir(BOOKDIR):
        raise FileNotFoundError(f"书籍目录不存在: {BOOKDIR}")
    for fn in os.listdir(BOOKDIR):
        if fn.endswith(".md"):
            texts[fn.split("_")[0]] = open(os.path.join(BOOKDIR, fn), encoding="utf-8").read()
    return texts


def load_parent_groups():
    groups = collections.defaultdict(list)
    with open(PARENTS, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                p = json.loads(line)
                groups[label_of_parent(p)].append(p["text"])
    return groups


def term_count(label, term, book_texts, parent_groups):
    if label in book_texts:
        return book_texts[label].count(term)
    return sum(t.count(term) for t in parent_groups.get(label, []))


def validate_cases(book_texts, parent_groups):
    warnings = []
    for q, ok, term, kind in CASES:
        counts = {label: term_count(label, term, book_texts, parent_groups)
                  for label in sorted(ok)}
        if not any(v >= 2 for v in counts.values()):
            warnings.append((q, term, sorted(ok), counts))
    return warnings


def case_source_type(ok):
    ok = set(ok)
    if ok <= BOOK_LABELS:
        return "book"
    if all(label.startswith("qa#") for label in ok):
        return "qa"
    if all(label.startswith("comment#") for label in ok):
        return "comment"
    return "mixed"


def rank_of(hits, ok, k=5):
    labels = [label_of_parent(h["parent"]) for h in hits[:k]]
    return next((i + 1 for i, label in enumerate(labels) if label in ok), None), labels


def aggregate(rows, rank_key, stratum="all"):
    subset = rows if stratum == "all" else [r for r in rows if r["stratum"] == stratum]
    n = len(subset)
    hit = sum(1 for r in subset if r[rank_key])
    rr = sum(1.0 / r[rank_key] for r in subset if r[rank_key])
    return n, hit, (hit / n if n else 0.0), (rr / n if n else 0.0)


def fmt_metric(n, hit, recall, mrr):
    return f"Recall@5 = {hit}/{n} = {recall:.0%} | MRR = {mrr:.2f}"


def main():
    book_texts = load_book_texts()
    parent_groups = load_parent_groups()

    if len(CASES) < 60:
        raise ValueError(f"评测集只有 {len(CASES)} 条，少于 60 条")
    known_labels = set(book_texts) | set(parent_groups)
    for q, ok, term, kind in CASES:
        if not ok:
            raise ValueError(f"可接受来源为空: {q}")
        if kind not in ("P", "D"):
            raise ValueError(f"未知用例类型 {kind}: {q}")
        if stratum_of(ok) == "other":
            raise ValueError(f"用例跨层级或标签不合法，无法分层统计: {q} | {sorted(ok)}")
        for label in ok:
            if label not in known_labels:
                raise ValueError(f"可接受来源标签不存在: {label} | 问题: {q}")

    warnings = validate_cases(book_texts, parent_groups)

    # Hybrid 内部已经构建 Index；直接复用，避免重复建 BM25。
    hybrid = S.Hybrid(emb_path=EMB, model_name=MODEL)
    idx = hybrid.idx
    if hybrid.emb.shape[0] != len(idx.children):
        raise ValueError(
            f"向量数 {hybrid.emb.shape[0]} 与子块数 {len(idx.children)} 不符，拒绝评测")

    rows = []
    for q, ok, term, kind in CASES:
        bm25_hits = idx.search(q, 5)
        bm25_rank, bm25_labels = rank_of(bm25_hits, ok)
        a, b = hybrid.rankings(q, 20)
        row = {
            "q": q,
            "ok": sorted(ok),
            "term": term,
            "kind": kind,
            "stratum": stratum_of(ok),
            "source_type": case_source_type(ok),
            "bm25_rank": bm25_rank,
            "bm25_labels": bm25_labels,
        }
        hy_labels = {}
        for k in RRF_KS:
            hits = hybrid.fuse(a, b, top_k=5, rrf_k=k)
            rank, labels = rank_of(hits, ok)
            row[f"hybrid_k{k}_rank"] = rank
            hy_labels[k] = labels
        row["hybrid_k60_labels"] = hy_labels[60]
        rows.append(row)

    # 未知标签（例如书前缀异常）必须显式暴露，不能静默当作未命中。
    unknown = sorted({label for r in rows for label in
                      (r["bm25_labels"] + r["hybrid_k60_labels"])
                      if label == "book_other"})
    if unknown:
        raise ValueError(f"检索结果出现无法归类的来源标签: {unknown}")

    lines = []
    n = len(CASES)
    book_n = sum(1 for r in rows if r["stratum"] == "book")
    qa_n = sum(1 for r in rows if r["source_type"] == "qa")
    cm_n = sum(1 for r in rows if r["source_type"] == "comment")
    p_n = sum(1 for r in rows if r["kind"] == "P")
    d_n = sum(1 for r in rows if r["kind"] == "D")
    label_cov = collections.Counter(label for _, ok, _, _ in CASES for label in ok)

    lines.append(f"模型 {MODEL} | 评测集 {n} 条 | 子块 {len(idx.children)}")
    lines.append(f"构成：书籍 {book_n} 条（qa+comment {n - book_n} 条：qa {qa_n} / comment {cm_n}）")
    lines.append(f"类型：改述式 {p_n} 条 / 术语直问 {d_n} 条")
    lines.append("qa/comment 的可接受集合精确到具体问答/主评论条目，而非任意来源命中")
    lines.append("书籍覆盖（多书集合重复计数）：" + "，".join(
        f"{lab} {label_cov[lab]} 条" for lab in ("b1", "b2", "b3", "b4", "b5", "kelly")))
    lines.append(f"术语校验：{len(warnings)} 条告警")
    for q, term, ok, counts in warnings:
        lines.append(f"  告警：{q} | 「{term}」| 期望 {ok} | 计数 {counts}")

    lines.append("")
    lines.append("===== 分层统计（Hybrid 默认 RRF k=60）=====")
    lines.append(f"{'层级':<14}{'N':>4}  BM25 / Hybrid")
    for key, title in (("book", "仅书籍"), ("qa_comment", "仅 qa+comment"), ("all", "全体")):
        nb, hb, rb, mb = aggregate(rows, "bm25_rank", key)
        nh, hh, rh, mh = aggregate(rows, "hybrid_k60_rank", key)
        lines.append(f"{title:<14}{nb:>4}  BM25: {fmt_metric(nb, hb, rb, mb)}"
                     f" | Hybrid: {fmt_metric(nh, hh, rh, mh)}")

    lines.append("")
    lines.append("===== 旧 20 条书籍评测集（与 bge-m3 基线同集）=====")
    legacy_rows = [r for r in rows if r["q"] in LEGACY_QUESTIONS]
    if len(legacy_rows) != len(LEGACY_QUESTIONS):
        raise ValueError(
            f"旧 20 条子集匹配到 {len(legacy_rows)} 条，应为 {len(LEGACY_QUESTIONS)} 条")
    for key, name in (("bm25_rank", "BM25"), ("hybrid_k60_rank", "Hybrid k=60")):
        nn, hh, rr, mm = aggregate(legacy_rows, key)
        lines.append(f"{name:<14} {fmt_metric(nn, hh, rr, mm)}")

    lines.append("")
    lines.append("===== RRF k 敏感性（Hybrid）=====")
    lines.append(f"{'k':<6}{'仅书籍':<40}{'仅 qa+comment':<40}{'全体':<40}")
    for k in RRF_KS:
        parts = []
        for stratum in ("book", "qa_comment", "all"):
            nn, hh, rr, mm = aggregate(rows, f"hybrid_k{k}_rank", stratum)
            parts.append(fmt_metric(nn, hh, rr, mm))
        lines.append(f"{k:<6}{parts[0]:<40}{parts[1]:<40}{parts[2]:<40}")

    lines.append("")
    lines.append("===== RRF k 差异（以 k=60 为基准）=====")
    for k in RRF_KS:
        if k == 60:
            continue
        hit_diff = sum(1 for r in rows
                       if bool(r[f"hybrid_k{k}_rank"]) != bool(r["hybrid_k60_rank"]))
        rank_diff = sum(1 for r in rows
                        if r[f"hybrid_k{k}_rank"] != r["hybrid_k60_rank"])
        nn, hh, rr, mm = aggregate(rows, f"hybrid_k{k}_rank")
        n60, h60, r60, m60 = aggregate(rows, "hybrid_k60_rank")
        lines.append(
            f"k={k} vs 60：命中变化 {hit_diff} 条，排名变化 {rank_diff} 条，"
            f"Recall@5 {r60:.0%} -> {rr:.0%}，MRR {m60:.2f} -> {mm:.2f}")

    lines.append("")
    lines.append("===== 未命中用例（Hybrid k=60）=====")
    misses = [r for r in rows if not r["hybrid_k60_rank"]]
    if not misses:
        lines.append("无")
    for r in misses:
        lines.append(f"  [{r['stratum']}] {r['q']} | 期望 {r['ok']} | top5 {r['hybrid_k60_labels']}")

    out_path = os.path.join(HERE, "eval30_out.txt")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    rows_path = os.path.join(HERE, "eval30_rows.jsonl")
    with open(rows_path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"eval done: {n} cases, {len(warnings)} term warnings -> eval30_out.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
