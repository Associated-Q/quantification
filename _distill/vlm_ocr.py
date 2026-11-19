"""扫描件转写：调用 OpenAI 兼容的多模态接口逐页转写，支持断点续跑。

环境变量：
  OCR_API_BASE  例如 https://dashscope.aliyuncs.com/compatible-mode/v1
  OCR_API_KEY   你的 key
  OCR_MODEL     例如 qwen-vl-max / glm-4v-plus / doubao-1.5-vision-pro

用法：
  python vlm_ocr.py <pdf> <outdir> [--pages 1-10] [--split 1|2] [--dpi 200]
"""
import argparse
import base64
import json
import os
import sys
import time
import uuid

import pymupdf
import requests

SESSION = str(uuid.uuid4())

PROMPT = (
    "逐字转写这一页的全部正文，要求：\n"
    "1. 严格按照原文，不增删、不改写、不总结、不解释；\n"
    "2. 保留原有的段落与章节标题，标题单独成行；\n"
    "3. 图表只输出图号和图注文字（如“图3.2 道氏理论的三种趋势”），不要描述图形内容；\n"
    "4. 页眉、页脚、页码不要输出；\n"
    "5. 只输出转写结果本身，不要任何前后说明。"
)


def load_env():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def render_clip(page, dpi, clip=None):
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, clip=clip)
    return pix.tobytes("jpeg", jpg_quality=90)


def call_api(b64, base, key, model):
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}},
        ]}],
    }
    r = requests.post(f"{base.rstrip('/')}/chat/completions",
                      headers={"Authorization": f"Bearer {key}",
                               "x-opencode-session": SESSION,
                               "Content-Type": "application/json"},
                      json=payload, timeout=180)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    j = r.json()
    return j["choices"][0]["message"]["content"].strip(), j.get("usage", {})


def page_parts(page, dpi, split):
    if split == 1:
        return [render_clip(page, dpi)]
    r = page.rect
    mid = r.y0 + r.height / 2
    return [render_clip(page, dpi, pymupdf.Rect(r.x0, r.y0, r.x1, mid)),
            render_clip(page, dpi, pymupdf.Rect(r.x0, mid, r.x1, r.y1))]


def parse_range(spec, total):
    if not spec:
        return 0, total - 1
    if "-" in spec:
        a, b = spec.split("-")
        return int(a) - 1, min(int(b) - 1, total - 1)
    return int(spec) - 1, int(spec) - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("outdir")
    ap.add_argument("--pages")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--split", type=int, default=1, choices=[1, 2],
                    help="1=整页一次；2=上下两半各一次（密排文字更准，成本翻倍）")
    args = ap.parse_args()

    load_env()
    base, key = os.environ.get("OCR_API_BASE", ""), os.environ.get("OCR_API_KEY", "")
    model = os.environ.get("OCR_MODEL", "")
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
    first, last = parse_range(args.pages, doc.page_count)
    total_tok, t0, failed = 0, time.time(), []
    with open(jsonl, "a", encoding="utf-8") as fh:
        for pno in range(first, last + 1):
            k = f"{pno + 1}-{args.split}"
            if k in done:
                continue
            texts, tok = [], 0
            try:
                for part in page_parts(doc[pno], args.dpi, args.split):
                    txt, usage = call_api(base64.b64encode(part).decode(), base, key, model)
                    texts.append(txt)
                    tok += usage.get("total_tokens", 0)
            except Exception as e:
                failed.append(pno + 1)
                print(f"  !! page {pno + 1} 失败: {type(e).__name__} {str(e)[:160]}",
                      file=sys.stderr, flush=True)
                continue
            text = "\n".join(texts)
            total_tok += tok
            fh.write(json.dumps({"key": k, "page": pno + 1, "text": text,
                                 "tokens": tok}, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"page {pno + 1}/{last + 1} chars={len(text)} tok={tok} "
                  f"elapsed={time.time() - t0:.0f}s", flush=True)

    print(f"DONE tokens={total_tok} failed_pages={failed} -> {jsonl}")


if __name__ == "__main__":
    main()
