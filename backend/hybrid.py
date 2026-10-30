"""
Hybrid search: meaning (embeddings) plus exact words (BM25), merged with
Reciprocal Rank Fusion (RRF).

Why: embeddings capture meaning but can blur rare technical terms such as
"NCCL", "Verkle" or "T-SAR". BM25 is the classic keyword scorer and is strong
on exactly those. RRF merges two ranked lists using only the ranks, so the two
score scales never need to be compared.

No extra API calls: BM25 is built from the chunk text on startup.
"""

import math
import re
import time

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = set(
    "the a an and or of to in on for with by is are was were be been this that these those it its as at from "
    "which what how does do did can could should would will than then into over under about between "
    "using use used based via per not no we our their".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if len(t) >= 2 and t not in _STOP]


class BM25:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(texts)
        post: dict[str, dict[int, int]] = {}
        lengths = np.zeros(self.n, dtype=np.float32)
        for i, t in enumerate(texts):
            toks = tokenize(t)
            lengths[i] = max(len(toks), 1)
            for tok in toks:
                d = post.setdefault(tok, {})
                d[i] = d.get(i, 0) + 1
        self.avgdl = float(lengths.mean()) if self.n else 1.0
        self.lengths = lengths
        self.index = {
            tok: (np.fromiter(d.keys(), dtype=np.int64), np.fromiter(d.values(), dtype=np.float32))
            for tok, d in post.items()
        }

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(self.n, dtype=np.float32)
        for tok in set(tokenize(query)):
            if tok not in self.index:
                continue
            ids, tf = self.index[tok]
            df = len(ids)
            idf = math.log(1 + (self.n - df + 0.5) / (df + 0.5))
            denom = tf + self.k1 * (1 - self.b + self.b * self.lengths[ids] / self.avgdl)
            out[ids] += idf * tf * (self.k1 + 1) / denom
        return out

    def top(self, query: str, k: int) -> list[int]:
        sc = self.scores(query)
        k = min(k, self.n)
        if k == 0 or not sc.any():
            return []
        idx = np.argpartition(sc, -k)[-k:]
        idx = idx[np.argsort(-sc[idx])]
        return [int(i) for i in idx if sc[i] > 0]


def hybrid_search(engine, bm25: BM25, text: str, qvec: np.ndarray, top_k: int = 5,
                  n_candidates: int = 50, w_dense: float = 1.0, w_bm25: float = 1.0,
                  rrf_k: int = 60) -> dict:
    """Same return shape as engine.search; each result's score stays the embedding cosine."""
    dense = engine.search(qvec, top_k=n_candidates)
    t0 = time.perf_counter()
    kw = bm25.top(text, n_candidates)
    bm25_ms = (time.perf_counter() - t0) * 1000

    fused: dict[int, float] = {}
    cos = {}
    for rank, r in enumerate(dense["results"]):
        fused[r["index"]] = w_dense / (rrf_k + rank + 1)
        cos[r["index"]] = r["score"]
    for rank, i in enumerate(kw):
        fused[i] = fused.get(i, 0.0) + w_bm25 / (rrf_k + rank + 1)

    order = sorted(fused, key=fused.get, reverse=True)[:top_k]
    missing = [i for i in order if i not in cos]
    for i, c in zip(missing, engine.cosine(qvec, missing)):
        cos[i] = float(c)

    return {
        "results": [{"document": engine.documents[i], "score": cos[i], "index": i, "fused": fused[i]} for i in order],
        "search_time_ms": dense["search_time_ms"],
        "bm25_time_ms": round(bm25_ms, 3),
        "backend": dense["backend"],
        "num_documents_searched": dense["num_documents_searched"],
        "mode": "hybrid",
    }
