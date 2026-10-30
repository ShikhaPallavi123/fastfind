"""
Build the search index from a folder of .md / .txt files.

    python scripts/ingest.py data/docs
    python scripts/ingest.py data/arxiv --yes     # large runs ask for --yes first

What it does:
  1. Reads every .md and .txt file in the folder.
  2. Splits each into chunks at sentence boundaries (short documents stay whole).
  3. Embeds every chunk (Nebius Token Factory, or fake mode for offline tests).
     Progress is saved in data/.partial/, so an interrupted run resumes
     instead of paying for the same embeddings twice.
  4. Saves data/index.npz (vectors) and data/chunks.json (text + source file).

The API loads those two files on startup.
"""

import hashlib
import json
import os
import re
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from backend.embeddings import embed_texts, is_fake

WHOLE_DOC_WORDS = 220   # documents up to this long stay as one chunk
MAX_CHUNK_WORDS = 170   # longer ones are split at sentence boundaries
GROUP = 256             # chunks embedded per saved checkpoint
CONFIRM_ABOVE = 300     # real-mode runs bigger than this need --yes
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[\"'])")


def chunk_text(text: str) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []
    if len(text.split()) <= WHOLE_DOC_WORDS:
        return [text]
    sentences = _SENT.split(text)
    chunks, current = [], []
    for sent in sentences:
        n = len(sent.split())
        if current and sum(len(x.split()) for x in current) + n > MAX_CHUNK_WORDS:
            chunks.append(" ".join(current))
            current = [current[-1]] if len(current) > 1 else []   # carry one sentence of overlap
        current.append(sent)
    if current:
        chunks.append(" ".join(current))
    return chunks


def main(folder: str, confirmed: bool) -> None:
    chunks, sources = [], []
    for name in sorted(os.listdir(folder)):
        if not name.lower().endswith((".md", ".txt")):
            continue
        with open(os.path.join(folder, name), encoding="utf-8") as f:
            for piece in chunk_text(f.read()):
                chunks.append(piece)
                sources.append(name)

    if not chunks:
        sys.exit(f"No .md or .txt files with text found in {folder}")

    mode = "FAKE (offline test only)" if is_fake() else "Nebius Token Factory"
    words = sum(len(c.split()) for c in chunks)
    print(f"{len(chunks)} chunks from {len(set(sources))} files, about {int(words * 1.4):,} tokens, via {mode}")
    if not is_fake() and len(chunks) > CONFIRM_ABOVE and not confirmed:
        sys.exit("This run will make many paid API calls. Re-run with --yes to continue.")

    sig = hashlib.md5("\n".join(chunks).encode()).hexdigest()[:12]
    part = os.path.join(OUT_DIR, ".partial", sig)
    os.makedirs(part, exist_ok=True)

    start_time = time.perf_counter()
    pieces = []
    for start in range(0, len(chunks), GROUP):
        f = os.path.join(part, f"{start:08d}.npy")
        if os.path.exists(f):
            pieces.append(np.load(f))
            continue
        vecs = embed_texts(chunks[start : start + GROUP])
        np.save(f, vecs)
        pieces.append(vecs)
        done = min(start + GROUP, len(chunks))
        print(f"  embedded {done}/{len(chunks)}", flush=True)
    vectors = np.vstack(pieces)
    print(f"Done in {time.perf_counter() - start_time:.1f}s, vector size {vectors.shape[1]}")

    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(os.path.join(OUT_DIR, "index.npz"), embeddings=vectors)
    with open(os.path.join(OUT_DIR, "chunks.json"), "w", encoding="utf-8") as fh:
        json.dump({"chunks": chunks, "sources": sources, "fake": is_fake()}, fh, ensure_ascii=False)
    shutil.rmtree(os.path.join(OUT_DIR, ".partial"), ignore_errors=True)
    print(f"Saved index to {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 1:
        sys.exit("Usage: python scripts/ingest.py <folder-with-md-or-txt-files> [--yes]")
    main(args[0], "--yes" in sys.argv)
