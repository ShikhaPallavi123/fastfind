"""
Retrieval evaluation for FastFind.

    python scripts/eval.py                          # uses data/eval_questions.json
    python scripts/eval.py data/my_questions.json

Reports, over a labeled question set:
  recall@1/3/5  how often the right source file is among the top k results
  MRR           mean reciprocal rank of the first correct result
  search time   mean and p95 time of the similarity search alone
  abstain rate  for unanswerable questions, how often the relevance filter
                (same rule the app uses) correctly returns nothing
Only the question embeddings call Token Factory (one batch, a few cents at most).
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from backend.embeddings import embed_texts, is_fake  # noqa: E402
from backend.hybrid import BM25, hybrid_search  # noqa: E402
from gpu_search.search_engine import SimilaritySearchEngine  # noqa: E402

DATA = os.path.join(ROOT, "data")
MIN_SCORE = float(os.environ.get("MIN_SCORE", "0.30"))
REL_SCORE = float(os.environ.get("REL_SCORE", "0.60"))


def has_name(q: str) -> bool:
    """True if the question contains a system-style name: acronym, CamelCase, digits or hyphenated."""
    for tok in q.replace("?", " ").replace(",", " ").split():
        t = tok.strip("()[]\"'.")
        if len(t) >= 3 and t[0].isalpha() and (any(c.isupper() for c in t[1:]) or any(c.isdigit() for c in t) or "-" in t):
            return True
    return False


def main(path: str, mode: str) -> None:
    with open(path, encoding="utf-8") as f:
        qs = json.load(f)["questions"]
    extra = os.path.join(DATA, "eval_offtopic.json")
    if os.path.exists(extra) and os.path.abspath(path) != os.path.abspath(extra):
        with open(extra, encoding="utf-8") as f:
            qs += json.load(f)["questions"]
    with open(os.path.join(DATA, "chunks.json"), encoding="utf-8") as f:
        meta = json.load(f)
    emb = np.load(os.path.join(DATA, "index.npz"))["embeddings"]

    engine = SimilaritySearchEngine(use_gpu=True, dims=int(os.environ.get("FASTFIND_DIMS", "0")) or None)
    engine.index(emb, [{"text": t, "source": s} for t, s in zip(meta["chunks"], meta["sources"])])
    bm25 = BM25(meta["chunks"]) if mode == "hybrid" else None
    print(f"Search mode: {mode}, vector length: {engine.embeddings.shape[1]}")
    print(f"Index: {len(meta['chunks'])} chunks, backend={engine.backend}, "
          f"embeddings={'OFFLINE FAKE (not meaningful)' if is_fake() else 'Nebius'}")

    vecs = embed_texts([q["q"] for q in qs])
    hits = {1: 0, 3: 0, 5: 0}
    rr, times = [], []
    grp = {True: [0, 0, 0], False: [0, 0, 0]}   # has_name -> [count, hits@1, hits@5]
    answerable = [q for q in qs if q["answerable"]]
    abstained = total_unans = 0
    misses = []
    top_scores = {"answerable": [], "nearmiss": [], "offtopic": []}

    for q, v in zip(qs, vecs):
        found = hybrid_search(engine, bm25, q["q"], v, top_k=5) if bm25 else engine.search(v, top_k=5)
        times.append(found["search_time_ms"])
        res = found["results"]
        top_dense = max((r["score"] for r in res), default=0.0)
        top_scores["answerable" if q["answerable"] else q.get("kind", "offtopic")].append(top_dense)
        if q["answerable"]:
            ranks = [i + 1 for i, r in enumerate(res) if r["document"]["source"] == q["source"]]
            first = ranks[0] if ranks else None
            for k in hits:
                if first is not None and first <= k:
                    hits[k] += 1
            rr.append(1.0 / first if first else 0.0)
            g = grp[has_name(q["q"])]
            g[0] += 1
            g[1] += first == 1
            g[2] += first is not None
            if first != 1:
                misses.append((q["q"], q["source"], [r["document"]["source"] for r in res[:3]]))
        else:
            total_unans += 1
            top = res[0]["score"] if res else 0.0
            kept = [r for r in res if r["score"] >= max(MIN_SCORE, top * REL_SCORE)]
            if top < MIN_SCORE or not kept:
                abstained += 1

    n = len(answerable)
    if n == 0:
        sys.exit("No answerable questions in the file, so there is nothing to score.")
    print(f"\nAnswerable questions: {n}")
    for k in (1, 3, 5):
        print(f"  recall@{k}: {hits[k] / n:.1%}  ({hits[k]}/{n})")
    print(f"  MRR:      {sum(rr) / n:.3f}")
    for named, label in ((False, "plain-language questions"), (True, "questions naming a system")):
        c, h1, h5 = grp[named]
        if c:
            print(f"    {label}: {c}  ->  recall@1 {h1 / c:.0%}, recall@5 {h5 / c:.0%}")
    print(f"\nSearch time (similarity search only): mean {np.mean(times):.2f} ms, "
          f"p95 {np.percentile(times, 95):.2f} ms over {len(times)} queries")
    if total_unans:
        print(f"\nUnanswerable questions: {total_unans}")
        print(f"  correctly rejected by the relevance filter: {abstained}/{total_unans}")
        print("  (a miss here is not fatal: the model is also told to say when sources lack the answer)")
    ans, near, off = (np.array(top_scores[k]) for k in ("answerable", "nearmiss", "offtopic"))
    print("\nBest embedding score per question (higher = more confident match):")
    if len(ans):
        print(f"  answerable ({len(ans)}): min {ans.min():.2f}, 5th pct {np.percentile(ans, 5):.2f}, "
              f"median {np.median(ans):.2f}")
    if len(near):
        print(f"  near-miss  ({len(near)}): median {np.median(near):.2f}, max {near.max():.2f}")
    if len(off):
        print(f"  off-topic  ({len(off)}): median {np.median(off):.2f}, max {off.max():.2f}")
    if len(ans) and len(off):
        lo, hi = float(off.max()), float(np.percentile(ans, 5))
        if lo < hi:
            print(f"  -> separable: a cutoff near {(lo + hi) / 2:.2f} rejects every off-topic question "
                  f"and keeps 95% of answerable ones")
            if len(near):
                print(f"     (it would reject {(near < (lo + hi) / 2).mean():.0%} of the near-miss questions)")
        else:
            print("  -> off-topic scores overlap answerable scores; a score cutoff alone is not enough")
    if misses:
        print("\nNot ranked first:")
        for q, want, got in misses:
            print(f"  - {q}\n      wanted {want}, top3 {got}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    mode = sys.argv[sys.argv.index("--mode") + 1] if "--mode" in sys.argv else "dense"
    if "--mode" in sys.argv:
        args = [a for a in args if a != mode]
    main(args[0] if args else os.path.join(DATA, "eval_questions.json"), mode)
