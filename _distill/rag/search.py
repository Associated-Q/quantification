"""检索：BM25 + 父子块回填。

子块参与召回（粒度细、命中准），命中后返回所属父块（上下文完整）。
"""
import json
import os
import re
import sys

import jieba
from rank_bm25 import BM25Okapi

HERE = os.path.dirname(os.path.abspath(__file__))
STOP = set("的了和是在有就不都一也这中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说产种面而方后多定行学法所民得经十三之进着等部度家电力里如水化高自二理起小物现实加量都两体制机当使点从业本去把性好应开它合还因由其些然前外天政四日那社义事平形相全表间样与关各重新线内数正心反你明看原又么利比或但质气第向道命此变条只没结解问意建月公无系军很情者最立代想已通并提直题党程展五果料象员革位入常文总次品式活设及管特件长求老头基资边流路级少图山统接知较将组见计别她手角期根论运农指几九区强放决西被干做必战先回则任取据处队南给色光门即保治北造百规热领七海口东导器压志世金增争济阶油思术极交受联什认六共权收证改清己美再采转更单风切打白教速花带安场身车例真务具万每目至达走积示议声报斗完类八离华名确才科张信马节话米整空元况今集温传土许步群广石记需段研界拉林律叫且究观越织装影算低持音众书布复容儿须际商非验连断深难近矿千周委素技备半办青省列习响约支般史感劳便团往酸历市克何除消构府称太准精值号率族维划选标写存候毛亲快效斯院查江型眼王按格养易置派层片始却专状育厂京识适属圆包火住调满县局照参红细引听该铁价严龙飞")

sys.path.insert(0, HERE)
TEXT = re.compile(r"[一-鿿]+|[A-Za-z]+|\d+")


def tokenize(s: str):
    out = []
    for tok in jieba.cut(s):
        tok = tok.strip()
        if not tok or tok in STOP:
            continue
        if re.fullmatch(r"[\W_]+", tok):
            continue
        out.append(tok.lower())
    return out


class Index:
    def __init__(self, rag_dir=os.path.join(HERE, "..", "rag")):
        self.children = [json.loads(l) for l in
                         open(os.path.join(rag_dir, "children.jsonl"), encoding="utf-8")]
        self.parents = {json.loads(l)["parent_id"]: json.loads(l)
                        for l in open(os.path.join(rag_dir, "parents.jsonl"), encoding="utf-8")}
        self.corpus = [tokenize(c["text"]) for c in self.children]
        self.bm25 = BM25Okapi(self.corpus)

    def search(self, query: str, top_k: int = 5):
        scores = self.bm25.get_scores(tokenize(query))
        order = scores.argsort()[::-1]
        hits, seen = [], set()
        for i in order:
            if scores[i] <= 0:
                break
            child = self.children[i]
            pid = child["parent_id"]
            if pid in seen:            # 同一父块只返回一次
                continue
            seen.add(pid)
            hits.append({"score": float(scores[i]), "child": child,
                         "parent": self.parents[pid]})
            if len(hits) >= top_k:
                break
        return hits


def load_env():
    path = os.path.join(os.path.dirname(HERE), "..", ".env")
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


RERANK_PROMPT = """下面是用户问题和一个候选段落列表。请判断每个段落在多大程度上直接回答了该问题，输出 0-3 的分数（3=直接回答，2=相关，1=沾边，0=无关）。
只输出 JSON 数组，形如 [{"i":0,"s":2},...]，不要任何解释。

问题：{q}

段落：
{docs}"""


def llm_rerank(query, hits, top_k=5):
    """对候选做 LLM 相关性打分，失败时原样返回，不静默降级到错误结果。"""
    import json
    import requests
    load_env()
    base, key = os.environ.get("OCR_API_BASE"), os.environ.get("OCR_API_KEY")
    model = os.environ.get("LLM_MODEL") or os.environ.get("OCR_MODEL")
    if not (base and key and model):
        return hits[:top_k], "no_api"
    docs = "\n".join(f"[{i}] {h['child']['text'][:300]}" for i, h in enumerate(hits))
    prompt = RERANK_PROMPT.replace("{q}", query).replace("{docs}", docs)
    try:
        r = requests.post(f"{base.rstrip('/')}/chat/completions",
                          headers={"Authorization": f"Bearer {key}",
                                   "x-opencode-session": "rag-rerank"},
                          json={"model": model, "temperature": 0,
                                "messages": [{"role": "user", "content": prompt}]},
                          timeout=60)
        if r.status_code != 200:
            return hits[:top_k], f"http_{r.status_code}"
        txt = r.json()["choices"][0]["message"]["content"].strip()
        txt = txt[txt.find("["):txt.rfind("]") + 1]
        data = json.loads(txt)
        if isinstance(data, dict):
            data = next((v for v in data.values() if isinstance(v, list)), [])
        scores = {}
        for pos, item in enumerate(data):
            if isinstance(item, dict):
                i = int(item.get("i", item.get("index", pos)))
                s = float(item.get("s", item.get("score", 0)))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                i, s = int(item[0]), float(item[1])
            else:
                continue
            scores[i] = s
        if not scores:
            return hits[:top_k], f"parse_fail:{txt[:60]}"
    except Exception as e:
        return hits[:top_k], f"err_{type(e).__name__}"
    ranked = sorted(range(len(hits)), key=lambda i: (-scores.get(i, 0), i))
    return [hits[i] for i in ranked[:top_k]], "ok"


class Hybrid:
    """BM25 + 向量 + RRF 融合；RRF 常数 k 可配置，默认 60。"""

    def __init__(self, emb_path=None, model_name="BAAI/bge-small-zh-v1.5"):
        import numpy as np
        from sentence_transformers import SentenceTransformer

        self.idx = Index()
        self.emb = np.load(emb_path or os.path.join(HERE, "embeddings.npy"))
        # 模型已在本地缓存；优先离线加载，避免启动时联网检查被墙导致数分钟阻塞
        try:
            self.st = SentenceTransformer(model_name, local_files_only=True)
        except Exception:
            self.st = SentenceTransformer(model_name)

    def _to_hits(self, ids, scores):
        out, seen = [], set()
        for i, sc in zip(ids, scores):
            pid = self.idx.children[i]["parent_id"]
            if pid in seen:
                continue
            seen.add(pid)
            out.append({"score": float(sc), "child": self.idx.children[i],
                        "parent": self.idx.parents[pid]})
        return out

    def _bm25(self, q, k):
        sc = self.idx.bm25.get_scores(tokenize(q))
        order = sc.argsort()[::-1][:k * 3]
        return self._to_hits(order, [sc[i] for i in order])

    def _vector(self, q, k):
        qv = self.st.encode([q], normalize_embeddings=True)[0]
        sims = self.emb @ qv
        order = sims.argsort()[::-1][:k * 3]
        return self._to_hits(order, [sims[i] for i in order])

    def rankings(self, query, k=20):
        """返回 (BM25 排序, 向量排序)，供多组 RRF 参数复用，避免重复编码。"""
        return self._bm25(query, k), self._vector(query, k)

    # 口语化短文本（问答/评论）上向量召回的噪声大于收益：实测 qa+comment 子集
    # 纯 BM25 69% 优于等权混合 58%，故整体降低向量通道权重，让词法匹配主导。
    VECTOR_WEIGHT = {"book": 0.2, "qa": 0.2, "comment": 0.2, "community": 0.2}

    @classmethod
    def _weights(cls):
        raw = os.environ.get("VECTOR_WEIGHT")
        if not raw:
            return cls.VECTOR_WEIGHT
        v = float(raw)
        return {k: v for k in cls.VECTOR_WEIGHT}

    def fuse(self, a, b, top_k=5, rrf_k=60, books=None, sources=None, vw=None):
        """按 RRF 融合两路排序；rrf_k 为 RRF 常数（默认 60）。"""
        weights = self._weights() if vw is None else vw
        score, store = {}, {}
        for li, lst in enumerate((a, b)):
            for r, h in enumerate(lst, 1):
                pid = h["parent"]["parent_id"]
                src = h["parent"].get("source", "book")
                if sources and src not in sources:
                    continue
                if books:
                    book = h["parent"].get("book", "")
                    if not any(book.startswith(p) for p in books):
                        continue
                contrib = 1 / (rrf_k + r)
                if li == 1:
                    contrib *= weights.get(src, 1.0)
                score[pid] = score.get(pid, 0) + contrib
                store[pid] = h
        order = sorted(score, key=lambda p: -score[p])[:top_k]
        return [dict(store[p], score=score[p]) for p in order]

    def search(self, query, top_k=5, books=None, sources=None, rrf_k=60):
        a, b = self.rankings(query, 20)
        return self.fuse(a, b, top_k=top_k, rrf_k=rrf_k,
                         books=books, sources=sources)


def main():
    idx = Index()
    q = " ".join(sys.argv[1:]) or "看涨吞没形态的判定条件"
    cands = idx.search(q, 20)
    hits, status = llm_rerank(q, cands, top_k=5)
    print(f"rerank: {status}")
    for n, h in enumerate(hits, 1):
        p = h["parent"]
        print(f"\n[{n}] bm25={h['score']:.2f} | {p['book'][:24]} | P{p['page_start']}-{p['page_end']}")
        print(f"    父块开头: {p['text'][:140]}...")


if __name__ == "__main__":
    main()
