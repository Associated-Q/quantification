"""把新增材料（充值问答、评论区、凯利公式）解析成统一的分块，供 RAG 使用。

输出 extra_parents.jsonl / extra_children.jsonl，字段：
  parent_id, source(book|qa|comment), author, date, title, text, page_start/page_end
"""
import io
import json
import os
import re
import sys

import pymupdf

SRC = r"C:\Users\AssociatedQuartz\Desktop\技术分析\新增"
QA_MD = os.path.join(SRC, "unzipped", "Re重生的炒股生活", "qa.md")
COMMENTS = os.path.join(SRC, "全部评论_1219141791464292387_20261101_124954.txt")
KELLY = os.path.join(SRC, "凯利公式.pdf")
OUT = os.path.dirname(os.path.abspath(__file__))

PARENT_MAX, CHILD_MAX = 1400, 380


def split_long(text, max_len=CHILD_MAX):
    if len(text) <= max_len:
        return [text]
    parts, cur = [], ""
    for sent in re.split(r"(?<=[。！？；\n])", text):
        if not sent:
            continue
        if len(cur) + len(sent) > max_len and cur:
            parts.append(cur.strip())
            cur = sent
        else:
            cur += sent
    if cur.strip():
        parts.append(cur.strip())
    return parts


def chunk_doc(docs):
    """docs: [{source, author, date, title, text, page_start, page_end}]
    → parents, children（父块约 1200 字，子块 ≤380 字）"""
    parents, children = [], []
    for d in docs:
        paras = [p.strip() for p in d["text"].split("\n") if p.strip()]
        buf, pid = [], 0
        for p in paras:
            buf.append(p)
            if sum(len(x) for x in buf) >= PARENT_MAX:
                pid += 1
                _emit(d, buf, pid, parents, children)
                buf = []
        if buf:
            pid += 1
            _emit(d, buf, pid, parents, children)
    return parents, children


def _emit(d, buf, pid, parents, children):
    pid_full = f'{d["source"]}#{d["doc_id"]}#{pid}'
    parents.append({"parent_id": pid_full, "source": d["source"], "author": d.get("author", ""),
                    "date": d.get("date", ""), "title": d.get("title", ""),
                    "page_start": d.get("page_start"), "page_end": d.get("page_end"),
                    "text": "\n".join(buf)})
    child_max = 1200 if d["source"] == "community" else CHILD_MAX
    cur = ""
    pieces = []
    for para in buf:
        if len(para) > child_max:
            if cur:
                pieces.append(cur)
                cur = ""
            pieces.extend(split_long(para, child_max))
            continue
        if not cur:
            cur = para
        elif len(cur) + len(para) + 1 > child_max:
            pieces.append(cur)
            cur = para
        else:
            cur += "\n" + para
    if cur:
        pieces.append(cur)
    for piece in pieces:
        children.append({"parent_id": pid_full, "source": d["source"],
                         "author": d.get("author", ""), "date": d.get("date", ""),
                         "title": d.get("title", ""),
                         "page_start": d.get("page_start"), "page_end": d.get("page_end"),
                         "text": piece})


def parse_kelly():
    doc = pymupdf.open(KELLY)
    docs = []
    for i, p in enumerate(doc, 1):
        t = p.get_text().strip()
        if t:
            docs.append({"source": "book", "doc_id": "kelly", "author": "凯利公式",
                         "date": "", "title": f"凯利公式 第{i}页",
                         "page_start": i, "page_end": i, "text": t})
    return docs


QA_HEAD = re.compile(r"^## #(\d+) · ([\d-]+ [\d:]+) · (已回答|待回答)")
QA_ASK = re.compile(r"^\*\*提问\*\*（(.+?)）")
QA_ANS = re.compile(r"^\*\*回答\*\*")


def parse_qa():
    lines = io.open(QA_MD, encoding="utf-8").read().splitlines()
    docs, cur, mode = [], None, None
    buf_ask, buf_ans = [], []

    def flush():
        nonlocal cur, buf_ask, buf_ans, mode
        if cur and buf_ans:                     # 只保留有回答的条目
            cur["text"] = "【提问】" + " ".join(buf_ask) + "\n【回答】" + "\n".join(buf_ans)
            docs.append(cur)
        cur, buf_ask, buf_ans, mode = None, [], [], None

    for ln in lines:
        m = QA_HEAD.match(ln)
        if m:
            flush()
            cur = {"source": "qa", "doc_id": m.group(1), "author": "Re重生的炒股生活",
                   "date": m.group(2), "title": f"充电问答 #{m.group(1)}"}
            continue
        if cur is None:
            continue
        s = ln.strip()
        if s.startswith("**提问**"):
            mode = "ask"
            ma = QA_ASK.match(s)
            cur["asker_level"] = ma.group(1) if ma else ""
            continue
        if s.startswith("**回答**"):
            mode = "ans"
            continue
        if not s or s == "---":
            continue
        if mode == "ask":
            buf_ask.append(s.lstrip("> ").strip())
        elif mode == "ans":
            buf_ans.append(s)
    flush()
    return docs


CMT_MAIN = re.compile(r"^【主评论 (\d+)】([\d-]+ [\d:]+)\s+(.+?)\s+\(mid=(\d+)\)")
CMT_REPLY = re.compile(r"^\s+\[(\d+)\]\s+([\d-]+ [\d:]+)\s+(.+?):\s?(.*)$")


def parse_comments():
    """拆成三类文档：UP 主主评论、UP 主楼中楼回复、社区讨论。
    UP 主自己的话单独成文，便于检索和提炼；社区讨论单独标记，检索时可降权。"""
    lines = io.open(COMMENTS, encoding="utf-8-sig", errors="replace").read().splitlines()
    threads, cur = [], None
    for ln in lines:
        m = CMT_MAIN.match(ln)
        if m:
            if cur:
                threads.append(cur)
            cur = {"idx": m.group(1), "date": m.group(2), "author": m.group(3),
                   "body": [], "up_reply": [], "others": []}
            continue
        if cur is None:
            continue
        if ln.strip().startswith("──"):
            continue
        r = CMT_REPLY.match(ln)
        if r:
            who, txt = r.group(3), r.group(4).strip()
            if "up主" in who:
                cur["up_reply"].append(txt)
            else:
                cur["others"].append(f"{who}：{txt}")
            continue
        s = ln.strip()
        if s:
            cur["body"].append(s)
    if cur:
        threads.append(cur)

    docs = []
    for t in threads:
        base = {"doc_id": t["idx"], "date": t["date"],
                "title": f"{t['date'][:10]} 主评论 #{t['idx']}"}
        if t["body"]:
            docs.append({**base, "source": "comment", "author": t["author"],
                         "text": "\n".join(t["body"]), "part": "主评论"})
        if t["up_reply"]:
            docs.append({**base, "source": "comment", "author": t["author"],
                         "text": "【UP主在评论区补充】\n" + "\n".join(t["up_reply"]),
                         "part": "楼中楼回复"})
        if t["others"]:
            docs.append({**base, "source": "community", "author": "社区用户",
                         "text": "\n".join(t["others"]), "part": "社区讨论"})
    return docs


def main():
    docs = parse_kelly() + parse_qa() + parse_comments()
    stats = {}
    for d in docs:
        stats[d["source"]] = stats.get(d["source"], 0) + 1
    parents, children = chunk_doc(docs)
    with open(os.path.join(OUT, "extra_parents.jsonl"), "w", encoding="utf-8") as fh:
        for r in parents:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(OUT, "extra_children.jsonl"), "w", encoding="utf-8") as fh:
        for r in children:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("文档数", stats)
    print(f"父块 {len(parents)} | 子块 {len(children)}")


if __name__ == "__main__":
    main()
