"""把书内语料与新增材料（问答/评论/凯利公式）合并成统一索引，并重建向量。"""
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def load(name):
    return [json.loads(l) for l in open(os.path.join(HERE, name), encoding="utf-8")]


def save(name, rows):
    with open(os.path.join(HERE, name), "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    # 备份原始索引（只备份一次，避免反复运行覆盖）
    for n in ("children.jsonl", "parents.jsonl", "embeddings.npy"):
        p = os.path.join(HERE, n)
        b = p + ".base.bak"
        if os.path.exists(p) and not os.path.exists(b):
            shutil.copy2(p, b)

    base_c = load("children.jsonl.base.bak")
    base_p = load("parents.jsonl.base.bak")
    for r in base_c:
        r.setdefault("source", "book")
        r.setdefault("author", r.get("book", ""))
        r.setdefault("date", "")
        r.setdefault("title", r.get("book", ""))
    for r in base_p:
        r.setdefault("source", "book")
        r.setdefault("author", r.get("book", ""))
        r.setdefault("date", "")
        r.setdefault("title", r.get("book", ""))

    extra_c = load("extra_children.jsonl")
    extra_p = load("extra_parents.jsonl")

    children = base_c + extra_c
    parents = base_p + extra_p
    save("children.jsonl", children)
    save("parents.jsonl", parents)
    print(f"合并完成：父块 {len(parents)} | 子块 {len(children)}")

    from sentence_transformers import SentenceTransformer
    import numpy as np
    import time

    model_name = sys.argv[1] if len(sys.argv) > 1 else "BAAI/bge-small-zh-v1.5"
    model = SentenceTransformer(model_name)
    texts = [c["text"] for c in children]
    t0 = time.time()
    vecs = []
    step = 512
    for i in range(0, len(texts), step):
        vecs.append(model.encode(texts[i:i + step], batch_size=32,
                                 normalize_embeddings=True, show_progress_bar=False))
        print(f"  {min(i + step, len(texts))}/{len(texts)}  "
              f"{time.time() - t0:.0f}s", flush=True)
    emb = np.vstack(vecs)
    np.save(os.path.join(HERE, "embeddings.npy"), emb)
    print(f"向量完成 {emb.shape} 模型 {model_name} 用时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
