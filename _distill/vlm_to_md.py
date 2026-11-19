"""把 out_vlm 里的转写结果转成带 [P页码] 锚点的 Markdown。

用法：
  python vlm_to_md.py <out_vlm 目录> <out 目录> [--match 江恩] [--prefix g]
只转换文件名包含 --match 的书，避免重复生成已有书籍。
"""
import argparse
import glob
import json
import os
import re

PAGENO = re.compile(r'^[\s\-–—·\.]*\d{1,4}[\s\-–—·\.]*$')
TERM = tuple('。！？；：”」』）!?;:')
CJK = re.compile(r'[一-鿿]')


def clean_page(text):
    lines = [l.strip() for l in text.split("\n")]
    lines = [l for l in lines if l and not PAGENO.match(l)]
    paras, cur = [], ""
    for l in lines:
        if not cur:
            cur = l
        elif cur.endswith(TERM) or len(cur) < 6:
            paras.append(cur)
            cur = l
        else:
            sep = "" if CJK.match(cur[-1]) and CJK.match(l[0]) else " "
            cur += sep + l
    if cur:
        paras.append(cur)
    return paras


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--match", default="江恩")
    ap.add_argument("--prefix", default="b")
    ap.add_argument("--start", type=int, default=6)
    args = ap.parse_args()

    files = [p for p in sorted(glob.glob(os.path.join(args.src, "vlm_*.jsonl")))
             if args.match in os.path.basename(p)]
    if not files:
        print("没有匹配的转写文件")
        return
    for i, path in enumerate(files, 1):
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        rows.sort(key=lambda r: r["page"])
        stem = os.path.basename(path)[4:-6]
        out_name = f"{args.prefix}{args.start + i - 1}_{stem[:20]}.md"
        md, total = [], 0
        for r in rows:
            paras = clean_page(r["text"])
            md.append(f'\n\n[P{r["page"]}]\n\n')
            for par in paras:
                md.append(par + "\n\n")
                total += len(par)
        with open(os.path.join(args.dst, out_name), "w", encoding="utf-8") as fh:
            fh.write("".join(md))
        print(f"{out_name}: {len(rows)} 页 / {total} 字")


if __name__ == "__main__":
    main()
