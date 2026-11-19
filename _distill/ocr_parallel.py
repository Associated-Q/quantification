"""并行版扫描件转写：复用 vlm_ocr.py 的渲染与调用，按页并发，支持断点续跑。

用法：
  python ocr_parallel.py <pdf> <outdir> [--pages 1-10] [--dpi 200] [--split 1] [--workers 5]
"""
import argparse
import base64
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pymupdf  # noqa: E402
import vlm_ocr as V  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("outdir")
    ap.add_argument("--pages")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--split", type=int, default=1, choices=[1, 2])
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()

    V.load_env()
    base = os.environ.get("OCR_API_BASE")
    key = os.environ.get("OCR_API_KEY")
    model = os.environ.get("OCR_MODEL")
    if not (base and key and model):
        sys.exit("缺少环境变量 OCR_API_BASE / OCR_API_KEY / OCR_MODEL")

    os.makedirs(args.outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.pdf))[0][:24]
    jsonl = os.path.join(args.outdir, f"vlm_{stem}.jsonl")

    done = set()
    if os.path.exists(jsonl):
        with open(jsonl, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    done.add(json.loads(line)["key"])

    doc = pymupdf.open(args.pdf)
    first, last = V.parse_range(args.pages, doc.page_count)
    todo = [p for p in range(first, last + 1) if f"{p + 1}-{args.split}" not in done]
    print(f"{stem}: {doc.page_count} 页，待处理 {len(todo)}，并发 {args.workers}", flush=True)
    if not todo:
        print("全部已完成")
        return

    render_lock = threading.Lock()      # PyMuPDF 的页面渲染不保证线程安全

    def work(pno):
        with render_lock:
            parts = V.page_parts(doc[pno], args.dpi, args.split)
        texts, tok = [], 0
        for part in parts:
            txt, usage = V.call_api(base64.b64encode(part).decode(), base, key, model)
            texts.append(txt)
            tok += usage.get("total_tokens", 0)
        return pno, "\n".join(texts), tok

    t0, total, ok, fail = time.time(), 0, 0, []
    with open(jsonl, "a", encoding="utf-8") as fh, \
            ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, p): p for p in todo}
        for fut in as_completed(futs):
            pno = futs[fut]
            try:
                pno, text, tok = fut.result()
            except Exception as e:
                fail.append((pno + 1, f"{type(e).__name__}: {e}"[:120]))
                print(f"  !! page {pno + 1} 失败 {type(e).__name__}", flush=True)
                continue
            fh.write(json.dumps({"key": f"{pno + 1}-{args.split}", "page": pno + 1,
                                 "text": text, "tokens": tok}, ensure_ascii=False) + "\n")
            fh.flush()
            total += tok
            ok += 1
            if ok % 25 == 0:
                rate = ok / max(time.time() - t0, 1)
                left = (len(todo) - ok) / rate / 60 if rate else 0
                print(f"  {ok}/{len(todo)}  用时 {time.time() - t0:.0f}s  "
                      f"剩余约 {left:.0f} 分钟  tokens={total}", flush=True)
    print(f"DONE {stem}: ok={ok} fail={len(fail)} tokens={total} "
          f"用时 {time.time() - t0:.0f}s", flush=True)
    if fail:
        print("失败页:", fail[:10], flush=True)


if __name__ == "__main__":
    main()
