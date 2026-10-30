# FastFind

Ask questions about **8,397 arXiv abstracts** (distributed systems, performance, hardware, operating systems) and get a short answer that **cites its sources**, with links to the papers.

Built for the Nebius x NVIDIA Global AI Hackathon.

```
question -> embed (Qwen3-Embedding on Nebius Token Factory)
         -> exact similarity search over every chunk at once (one matrix multiply)
            + keyword search (BM25), merged with Reciprocal Rank Fusion
         -> relevance check (below a calibrated score: "not in the documents", no model call)
         -> NVIDIA Nemotron 3 Nano on Nebius Token Factory writes a short cited answer
```

- **NVIDIA model:** `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B`, called at runtime for every answer.
- **Nebius Token Factory:** both the embeddings and the Nemotron answer run through its OpenAI-compatible API.
- **Search:** exact brute-force similarity (no approximation), on CUDA through PyTorch when a GPU is present, NumPy on CPU otherwise. The page badge shows which one is active.

## Measured results

All numbers come from `scripts/eval.py` and `scripts/benchmark.py` in this repository.

**Retrieval quality.** 100 questions over 8,397 papers (11,297 chunks). The questions are *synthetic*: Nemotron wrote one question per randomly chosen abstract (`scripts/make_eval.py`), and that paper is the ground truth. About 20 of them are vague ("the proposed framework"), so no search could answer them. Treat 100 questions as a test with roughly ±6 points of noise.

| Setup | recall@1 | recall@3 | recall@5 | MRR |
|---|---|---|---|---|
| Embeddings only | 77% | 83% | 89% | 0.811 |
| **Hybrid (embeddings + BM25)** | **89%** | **94%** | **96%** | **0.919** |
| Hybrid, vectors cut to 1024 numbers | 88% | 94% | 95% | 0.912 |
| Hybrid, vectors cut to 512 numbers | 86% | 94% | 95% | 0.897 |

On the 32 questions that do not name a specific system, hybrid search reaches 84% recall@1 and 91% recall@5.

## Run it yourself

```bash
pip install -r requirements.txt            # add a CUDA build of torch for GPU search
export NEBIUS_API_KEY='your-token-factory-key'     # never commit this

python scripts/fetch_arxiv.py              # downloads arXiv abstracts to data/arxiv
python scripts/ingest.py data/arxiv --yes  # embeds them (paid API calls; prints a token estimate first)
uvicorn backend.app:app --port 8000        # open http://localhost:8000
```

Reproduce the measurements:

```bash
python scripts/make_eval.py data/arxiv 100                  # writes data/eval_arxiv.json
python scripts/eval.py data/eval_arxiv.json --mode dense
python scripts/eval.py data/eval_arxiv.json --mode hybrid
FASTFIND_DIMS=1024 python scripts/eval.py data/eval_arxiv.json --mode hybrid
python scripts/benchmark.py --sizes 100000,250000,500000   # CPU vs GPU search timing
```

Settings (environment variables): `NEBIUS_API_KEY` (required), `CHAT_MODEL`, `EMBED_MODEL`, `SEARCH_MODE` (`hybrid` or `dense`), `FASTFIND_DIMS`, `DAILY_LIMIT`, `PER_MIN_LIMIT`, `MIN_SCORE`, `NEMOTRON_NO_THINK=1` (optional, skips the model's thinking phase if the server supports it).

## What was built during the submission period

All code in this repository was written during the hackathon submission period, starting October 23, 2026. The GPU search approach builds on the author's graduate coursework in parallel programming and high-performance computing. The data is public arXiv metadata (titles and abstracts), fetched with the public arXiv API.

## License

MIT, see LICENSE.
