"""把 5 本书的 Markdown 切成父子块，供混合检索使用。

切分原则：
- 父块按小节聚合（800~1500 字），检索命中后返回给模型的是它；
- 子块 250~400 字，严格按段落边界，绝不切断句子；单段超 400 字才强制切并重叠 100 字；
- 每块带 book / page_start / page_end 元数据，便于溯源到原书页码。
"""
import glob
import json
import os
import re
import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = sys.argv[2] if len(sys.argv) > 2 else "rag"
PAGE = re.compile(r"\[P(\d+)\]")
TERM = tuple("。！？；：”」』）!?;:.")
HEAD = re.compile(r"^(第[一二三四五六七八九十百\d]+[章节篇]|\d+\.\d*\s|[一二三四五六七八九十]+、)")

PARENT_MIN, PARENT_MAX = 800, 1500
CHILD_MAX, OVERLAP = 400, 100


def is_heading(t: str) -> bool:
    """标题判定要严：短、无任何标点、不是图注/表注/编号行。"""
    if len(t) > 20 or re.search(r"[，。！？；：、,.;:!?]", t):
        return False
    if re.match(r"^[图表注释]", t) or re.match(r"^\d+(\.\d+)?$", t):
        return False
    return True


def parse(md: str):
    """→ [(page, paragraph)]"""
    out, page = [], 1
    for chunk in PAGE.split(md):
        if chunk.isdigit():
            page = int(chunk)
            continue
        for para in chunk.split("\n\n"):
            p = para.strip()
            if p:
                out.append((page, p))
    return out


def split_long(text: str):
    """单段过长时按句号切，重叠 OVERLAP 字。"""
    if len(text) <= CHILD_MAX:
        return [text]
    parts, cur = [], ""
    for sent in re.split(r"(?<=[。！？；])", text):
        if not sent:
            continue
        if len(cur) + len(sent) > CHILD_MAX and cur:
            parts.append(cur)
            cur = cur[-OVERLAP:] + sent
        else:
            cur += sent
    if cur:
        parts.append(cur)
    return parts


def main():
    os.makedirs(OUT, exist_ok=True)
    parents, children = [], []
    for path in sorted(glob.glob(os.path.join(SRC, "b*.md"))):
        book = os.path.splitext(os.path.basename(path))[0]
        paras = parse(open(path, encoding="utf-8").read())
        section, buf, p0, p1 = [], [], None, None
        pid = 0

        def flush():
            nonlocal buf, p0, p1, pid
            if not buf:
                return
            pid += 1
            parents.append({"parent_id": f"{book}#{pid}", "book": book,
                            "page_start": p0, "page_end": p1,
                            "text": "\n".join(buf)})
            cur = ""
            pieces = []

            def emit(t):
                pieces.append(t)

            for para in buf:
                if len(para) > CHILD_MAX:
                    if cur:
                        emit(cur)
                        cur = ""
                    pieces.extend(split_long(para))
                    continue
                if not cur:
                    cur = para
                elif len(cur) + len(para) + 1 > CHILD_MAX:
                    emit(cur)
                    cur = para
                else:
                    cur += "\n" + para
            if cur:
                emit(cur)
            for piece in pieces:
                children.append({"parent_id": f"{book}#{pid}", "book": book,
                                 "page_start": p0, "page_end": p1, "text": piece})
            buf, p0, p1 = [], None, None

        for page, para in paras:
            heading = is_heading(para) or HEAD.match(para)
            if heading and buf and sum(len(x) for x in buf) >= PARENT_MIN:
                flush()
            buf.append(para)
            p0 = p0 or page
            p1 = page
            if sum(len(x) for x in buf) >= PARENT_MAX:
                flush()
        flush()

    with open(os.path.join(OUT, "parents.jsonl"), "w", encoding="utf-8") as fh:
        for r in parents:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(OUT, "children.jsonl"), "w", encoding="utf-8") as fh:
        for r in children:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    import statistics
    pl = [len(p["text"]) for p in parents]
    cl = [len(c["text"]) for c in children]
    print(f"父块 {len(parents)} 个 | 中位 {int(statistics.median(pl))} 字 | 范围 {min(pl)}-{max(pl)}")
    print(f"子块 {len(children)} 个 | 中位 {int(statistics.median(cl))} 字 | 范围 {min(cl)}-{max(cl)}")
    print(f"超 400 字的子块: {sum(1 for x in cl if x > 400)}")
    books = {}
    for c in children:
        books[c["book"]] = books.get(c["book"], 0) + 1
    for b, n in sorted(books.items()):
        print(f"  {n:>5} 块  {b[:40]}")


if __name__ == "__main__":
    main()
