"""
Embedding front door used by both ingestion and the API.

Real mode (default): calls the embedding model on Nebius Token Factory.
Offline test mode: set FASTFIND_FAKE_EMBED=1 to use a simple hashed
bag-of-words vector. It needs no API key and gives rough keyword-based
similarity, which is enough to test the pipeline logic. NEVER use it for
the real demo or the submission.
"""

import hashlib
import os
import re
import time

import numpy as np

FAKE_DIM = 256


def _fake_embed_one(text: str) -> list[float]:
    vec = np.zeros(FAKE_DIM, dtype=np.float32)
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        if len(token) < 3:
            continue
        bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % FAKE_DIM
        vec[bucket] += 1.0
    return vec.tolist()


def is_fake() -> bool:
    return os.environ.get("FASTFIND_FAKE_EMBED") == "1"


def embed_texts(texts: list[str], batch_size: int = 64, retries: int = 4) -> np.ndarray:
    """Return an (n, dim) float32 array of embeddings for the given texts.

    retries: extra attempts after a temporary error. Use 0 for live questions so a visitor
    never waits through long backoff (the openai client still retries briefly by itself).
    """
    if is_fake():
        return np.array([_fake_embed_one(t) for t in texts], dtype=np.float32)

    from backend.nebius_client import embed

    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        for attempt in range(retries + 1):
            try:
                vectors.extend(embed(batch))
                break
            except Exception as e:  # retry only temporary problems, never auth or billing
                code = getattr(e, "status_code", None)
                temporary = code in (429, 500, 502, 503, 504) or code is None
                if not temporary or attempt == retries:
                    raise
                wait = 2 ** attempt
                print(f"[FastFind] embedding call failed ({code or type(e).__name__}); retrying in {wait}s", flush=True)
                time.sleep(wait)
    return np.array(vectors, dtype=np.float32)
