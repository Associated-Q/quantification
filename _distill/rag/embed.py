"""为子块生成向量，默认 bge-m3；可用 --model 换小模型做对比。"""
import json
import os
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = sys.argv[1] if len(sys.argv) > 1 else "BAAI/bge-m3"
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "embeddings.npy")

from sentence_transformers import SentenceTransformer  # noqa: E402
import numpy as np  # noqa: E402

children = [json.loads(l) for l in
            open(os.path.join(HERE, "children.jsonl"), encoding="utf-8")]
texts = [c["text"] for c in children]
print(f"模型 {MODEL} | 子块 {len(texts)} 个", flush=True)

model = SentenceTransformer(MODEL)
emb = model.encode(texts, batch_size=16, normalize_embeddings=True,
                   show_progress_bar=False)
np.save(OUT, emb)
print(f"saved {OUT} shape={emb.shape}", flush=True)
