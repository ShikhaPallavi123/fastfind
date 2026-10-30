"""
Assemble a folder ready to push to a Hugging Face Space.

    python scripts/make_space.py              # full 4096-number vectors
    python scripts/make_space.py --dims 1024  # smaller, for hosts with little memory

Creates space_build/ with the app code, Dockerfile, the built index, and the
README header Spaces needs. Your API key is NOT included: add it later as a
Space secret named NEBIUS_API_KEY.
"""

import json
import os
import shutil
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "space_build")

README = """---
title: FastFind
emoji: 🔎
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# FastFind

Ask questions about {n} arXiv abstracts (distributed systems, performance, hardware, OS).
GPU-style parallel similarity search plus keyword search finds the evidence;
NVIDIA Nemotron on Nebius Token Factory writes a short answer that cites its sources.

Built for the Nebius x NVIDIA Global AI Hackathon. Source code and measured results are in the project repository.
This hosted copy searches on CPU; the GPU speed test is reported separately.
"""

GITATTRIBUTES = "*.npz filter=lfs diff=lfs merge=lfs -text\ndata/chunks.json filter=lfs diff=lfs merge=lfs -text\n"


def main(allow_fake: bool, dims: int) -> None:
    meta_path = os.path.join(ROOT, "data", "chunks.json")
    idx_path = os.path.join(ROOT, "data", "index.npz")
    if not (os.path.exists(meta_path) and os.path.exists(idx_path)):
        sys.exit("No index found. Run scripts/ingest.py first.")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("fake") and not allow_fake:
        sys.exit("The index was built with offline FAKE embeddings. Rebuild it with real ones before deploying.")

    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(os.path.join(OUT, "data"))
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    for d in ("backend", "gpu_search", "frontend"):
        shutil.copytree(os.path.join(ROOT, d), os.path.join(OUT, d), ignore=ignore)
    for f in ("Dockerfile", "requirements.txt", "LICENSE"):
        shutil.copy(os.path.join(ROOT, f), OUT)
    if dims:
        import numpy as np
        emb = np.load(idx_path)["embeddings"][:, :dims]
        emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8)
        np.savez_compressed(os.path.join(OUT, "data", "index.npz"), embeddings=emb.astype(np.float32))
        print(f"Truncated vectors to {dims} numbers each (queries are cut to match automatically)")
    else:
        shutil.copy(idx_path, os.path.join(OUT, "data"))
    shutil.copy(meta_path, os.path.join(OUT, "data"))

    n_papers = len(set(meta["sources"]))
    with open(os.path.join(OUT, "README.md"), "w", encoding="utf-8") as f:
        f.write(README.format(n=f"{n_papers:,}"))
    with open(os.path.join(OUT, ".gitattributes"), "w") as f:
        f.write(GITATTRIBUTES)
    size = sum(os.path.getsize(os.path.join(dp, fn)) for dp, _, fns in os.walk(OUT) for fn in fns) / 1e6
    print(f"Built {OUT} ({size:.0f} MB, {len(meta['chunks'])} chunks from {n_papers} papers)")


if __name__ == "__main__":
    dims = int(sys.argv[sys.argv.index("--dims") + 1]) if "--dims" in sys.argv else 0
    main("--allow-fake" in sys.argv, dims)
