"""把所有书籍 Markdown 与新增材料合并成统一索引，并重算向量。

流程：build_index.py（书籍分块）→ 追加 qa/评论/社区/凯利 → 用 bge-small 重新编码。
用法：python rebuild_all.py [模型名]
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RAG = os.path.join(HERE, "rag")
OUT = os.path.join(HERE, "out")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def load(name):
    path = os.path.join(RAG, name)
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def save(name, rows):
    with open(os.path.join(RAG, name), "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "BAAI/bge-small-zh-v1.5"

    # 1) 书籍分块（会覆盖 rag/children.jsonl 与 parents.jsonl）
    print("→ 重新切分书籍…", flush=True)
    subprocess.run([sys.executable, "-I", os.path.join(RAG, "build_index.py"),
                    OUT, RAG], check=True)

    base_c, base_p = load("children.jsonl"), load("parents.jsonl")
    for r in base_c + base_p:
        r.setdefault("source", "book")
        r.setdefault("author", r.get("book", ""))
        r.setdefault("date", "")
        r.setdefault("title", r.get("book", ""))

    extra_c, extra_p = load("extra_children.jsonl"), load("extra_parents.jsonl")
    children, parents = base_c + extra_c, base_p + extra_p
    save("children.jsonl", children)
    save("parents.jsonl", parents)
    print(f"→ 合并完成：父块 {len(parents)} | 子块 {len(children)}", flush=True)

    from sentence_transformers import SentenceTransformer
    import numpy as np
    import time

    model = SentenceTransformer(model_name)
    texts = [c["text"] for c in children]
    t0, vecs, step = time.time(), [], 512
    for i in range(0, len(texts), step):
        vecs.append(model.encode(texts[i:i + step], batch_size=32,
                                 normalize_embeddings=True, show_progress_bar=False))
        print(f"  {min(i + step, len(texts))}/{len(texts)}  {time.time() - t0:.0f}s",
              flush=True)
    emb = np.vstack(vecs)
    np.save(os.path.join(RAG, "embeddings.npy"), emb)
    print(f"→ 向量完成 {emb.shape} 用时 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
