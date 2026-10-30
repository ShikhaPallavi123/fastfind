"""
FastFind API.

    uvicorn backend.app:app --port 8000

Flow for POST /ask:
    question text -> embed (Token Factory) -> GPU similarity search
                  -> Nemotron writes a grounded answer -> answer + sources + timings
"""

import json
import logging
import os
import sys
import time
from collections import defaultdict, deque
from datetime import date

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.INFO)   # shows "Retrying request..." in the terminal

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from backend.embeddings import embed_texts, is_fake  # noqa: E402
from backend.hybrid import BM25, hybrid_search  # noqa: E402
from gpu_search.search_engine import SimilaritySearchEngine  # noqa: E402

DATA_DIR = os.path.join(ROOT, "data")
FRONTEND_DIR = os.path.join(ROOT, "frontend")

app = FastAPI(title="FastFind", version="0.3.0")

# Guardrails so a public demo cannot run up the Token Factory bill.
DAILY_LIMIT = int(os.environ.get("DAILY_LIMIT", "300"))        # questions per day, all users
PER_MIN_LIMIT = int(os.environ.get("PER_MIN_LIMIT", "6"))      # questions per minute, per visitor
MAX_QUESTION_CHARS = 500
# Relevance cutoff, calibrated on the eval set (rejects every off-topic question, keeps 95% of answerable ones).
# Shorter vectors score a little higher, so the cutoff depends on the vector length. Override with MIN_SCORE.
def min_score_for(dim: int) -> float:
    env = os.environ.get("MIN_SCORE")
    if env:
        return float(env)
    return 0.51 if dim <= 1024 else 0.45
REL_SCORE = float(os.environ.get("REL_SCORE", "0.60"))         # keep sources >= 60% of the best score
usage = {"day": date.today(), "count": 0}
recent = defaultdict(deque)


def check_limits(ip: str) -> None:
    today = date.today()
    if usage["day"] != today:
        usage.update(day=today, count=0)
    if usage["count"] >= DAILY_LIMIT:
        raise HTTPException(429, "Daily demo limit reached. Please try again tomorrow.")
    now = time.time()
    q = recent[ip]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= PER_MIN_LIMIT:
        raise HTTPException(429, "Too many questions. Please wait a minute.")
    q.append(now)
    usage["count"] += 1
engine = SimilaritySearchEngine(use_gpu=True, dims=int(os.environ.get("FASTFIND_DIMS", "0")) or None)
index_info = {"loaded": False, "num_chunks": 0, "fake": False}
SEARCH_MODE = os.environ.get("SEARCH_MODE", "hybrid")   # "hybrid" beat "dense" on the 100-question eval (recall@1 89% vs 77%)
bm25 = None


def load_index() -> None:
    vec_path = os.path.join(DATA_DIR, "index.npz")
    meta_path = os.path.join(DATA_DIR, "chunks.json")
    if not (os.path.exists(vec_path) and os.path.exists(meta_path)):
        return
    embeddings = np.load(vec_path)["embeddings"]
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    docs = [
        {"text": t, "source": s} for t, s in zip(meta["chunks"], meta["sources"])
    ]
    engine.index(embeddings, docs)
    global bm25
    bm25 = BM25(meta["chunks"])
    index_info.update(loaded=True, num_chunks=len(docs), fake=meta.get("fake", False))
    # Warm-up: the first search after startup is slow (memory pages load). Pay that cost now, not on a visitor.
    probe = np.random.default_rng(0).standard_normal(embeddings.shape[1]).astype(np.float32)
    for _ in range(3):
        engine.search(probe, top_k=5)


load_index()


class AskRequest(BaseModel):
    question: str
    top_k: int = 5


@app.get("/health")
def health():
    return {
        "status": "ok",
        "search_backend": engine.backend,
        "embedding_mode": "fake-offline" if is_fake() else "nebius",
        "daily_remaining": max(DAILY_LIMIT - usage["count"], 0),
        "search_mode": SEARCH_MODE,
        "vector_length": int(engine.embeddings.shape[1]) if index_info["loaded"] else 0,
        "min_score": min_score_for(int(engine.embeddings.shape[1])) if index_info["loaded"] else None,
        **index_info,
    }


@app.post("/ask")
def ask(req: AskRequest, request: Request):
    question = req.question.strip()
    if not question:
        raise HTTPException(400, "Question is empty.")
    if len(question) > MAX_QUESTION_CHARS:
        raise HTTPException(400, f"Question is too long (max {MAX_QUESTION_CHARS} characters).")
    if not index_info["loaded"]:
        raise HTTPException(
            503, "No index loaded. Run: python scripts/ingest.py data/docs"
        )

    fwd = request.headers.get("x-forwarded-for", "")
    ip = fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "?")
    check_limits(ip)

    t0 = time.perf_counter()
    try:
        query_vec = embed_texts([question], retries=0)[0]
    except Exception as e:
        raise HTTPException(502, f"Embedding call failed: {e}")
    embed_ms = (time.perf_counter() - t0) * 1000

    k = max(1, min(req.top_k, 10))
    if SEARCH_MODE == "hybrid" and bm25 is not None:
        found = hybrid_search(engine, bm25, question, query_vec, top_k=k)
    else:
        found = engine.search(query_vec, top_k=k)
    sources = [
        {
            "text": r["document"]["text"],
            "source": r["document"]["source"],
            "score": round(r["score"], 4),
        }
        for r in found["results"]
    ]

    # Keep only sources that are actually close to the best match.
    min_score = 0.0 if is_fake() else min_score_for(engine.embeddings.shape[1])   # offline test vectors score on a different scale
    top_score = max((x["score"] for x in sources), default=0.0)
    dropped = len(sources)
    sources = [s for s in sources if s["score"] >= max(min_score, top_score * REL_SCORE)]
    dropped -= len(sources)

    t1 = time.perf_counter()
    if not sources or top_score < min_score:
        # Nothing relevant: say so instead of paying for a model call that would guess.
        sources = []
        answer_text = "I could not find this in the indexed documents."
    elif is_fake():
        answer_text = "[offline test mode: no Nemotron call] Top sources shown below."
    else:
        try:
            from backend.nebius_client import answer

            answer_text = answer(question, [s["text"] for s in sources])
        except Exception as e:
            raise HTTPException(502, f"Nemotron call failed: {e}")
    generate_ms = (time.perf_counter() - t1) * 1000

    return {
        "answer": answer_text,
        "sources": sources,
        "timings_ms": {
            "embed_question": round(embed_ms, 1),
            "gpu_search": found["search_time_ms"],
            "keyword_search": found.get("bm25_time_ms", 0.0),
            "nemotron_answer": round(generate_ms, 1),
        },
        "search_backend": found["backend"],
        "chunks_searched": found["num_documents_searched"],
        "sources_dropped": dropped,
        "daily_remaining": max(DAILY_LIMIT - usage["count"], 0),
    }


@app.get("/")
def home():
    page = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(page):
        return FileResponse(page)
    return {"message": "FastFind API is running. UI not built yet."}
