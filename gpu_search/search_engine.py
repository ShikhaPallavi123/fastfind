"""
FastFind GPU-Parallel Similarity Search Engine
================================================

Core idea: instead of comparing a query vector against each document
vector ONE AT A TIME (slow, sequential), we compare the query against
ALL document vectors AT ONCE using matrix multiplication, which GPUs
are built to do in parallel.

This module is device-agnostic:
- On your laptop (no GPU) -> runs on CPU, correct but slow.
- On a Nebius GPU instance -> automatically uses CUDA, fast.

This is the "brute-force" version (compares against every document).
It's the safety-net / Week 1 deliverable described in the project plan.
An approximate version (IVF-style clustering) can be layered on top
later as a stretch goal without changing this core.
"""

import time
import numpy as np


class SimilaritySearchEngine:
    """
    Stores document embeddings and finds the top-k most similar
    documents to a query embedding, using parallel matrix operations.
    """

    def __init__(self, use_gpu: bool = True, dims: int | None = None):
        """
        use_gpu: if True and a GPU/CUDA device is available (via torch),
                 computations run on GPU. Otherwise falls back to numpy (CPU).
        dims:    optional. Keep only the first `dims` numbers of every vector (Matryoshka-style
                 truncation, supported by Qwen3-Embedding). Smaller index, faster search.
                 Queries are cut to the same length automatically.
        """
        self.dims = dims
        self.backend = "numpy"  # default, overwritten below if torch+cuda available
        self._torch = None
        self._device = None
        self.embeddings = None  # shape: (num_documents, embedding_dim)
        self.documents = []     # parallel list of raw text/metadata

        if use_gpu:
            try:
                import torch
                self._torch = torch
                if torch.cuda.is_available():
                    self._device = torch.device("cuda")
                    self.backend = "torch-cuda"
                else:
                    self._device = torch.device("cpu")
                    self.backend = "torch-cpu"
            except Exception:
                # torch not installed, or broken/incompatible environment
                # (e.g. no CUDA driver present) -> fall back to plain numpy.
                self._torch = None
                self.backend = "numpy"

        print(f"[FastFind] Search engine initialized using backend: {self.backend}")

    def index(self, embeddings: np.ndarray, documents: list[str]) -> None:
        """
        Load document embeddings into the engine.

        embeddings: numpy array of shape (num_documents, embedding_dim)
                    Each row is a vector representing one document.
        documents:  list of raw text (or metadata) matching each row.
        """
        assert embeddings.shape[0] == len(documents), (
            "Number of embeddings must match number of documents"
        )

        if self.dims and embeddings.shape[1] > self.dims:
            embeddings = embeddings[:, : self.dims]

        # Normalize embeddings so that dot product == cosine similarity.
        # This is a standard trick: if vectors are unit length, their
        # dot product directly measures how "similar" they are.
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8  # avoid divide-by-zero
        normalized = embeddings / norms

        self.documents = documents

        if self.backend.startswith("torch"):
            self.embeddings = self._torch.tensor(
                normalized, dtype=self._torch.float32, device=self._device
            )
        else:
            self.embeddings = normalized.astype(np.float32)

        print(f"[FastFind] Indexed {len(documents)} documents "
              f"(embedding dim = {embeddings.shape[1]}) on {self.backend}")

    def _fit_query(self, q: np.ndarray) -> np.ndarray:
        """Cut the query to the index's vector length (the index may be truncated)."""
        d = self.embeddings.shape[1]
        return q[:d] if q.shape[0] > d else q

    def cosine(self, query_embedding: np.ndarray, indices: list[int]) -> np.ndarray:
        """Cosine similarity of the query against specific documents (used by hybrid search)."""
        if not indices:
            return np.zeros(0, dtype=np.float32)
        query_embedding = self._fit_query(query_embedding)
        q = (query_embedding / (np.linalg.norm(query_embedding) + 1e-8)).astype(np.float32)
        if self.backend.startswith("torch"):
            idx = self._torch.tensor(indices, device=self._device)
            return (self.embeddings[idx] @ self._torch.tensor(q, device=self._device)).cpu().numpy()
        return self.embeddings[indices] @ q

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> dict:
        """
        Find the top_k most similar documents to the query.

        Returns a dict with:
          - results: list of (document, similarity_score)
          - search_time_ms: how long the search took, in milliseconds
          - backend: which backend was used (numpy / torch-cpu / torch-cuda)
        """
        if self.embeddings is None:
            raise RuntimeError("No documents indexed yet. Call .index() first.")

        # Normalize the query the same way we normalized documents.
        query_embedding = self._fit_query(query_embedding)
        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        q_norm = q_norm.astype(np.float32)

        start = time.perf_counter()

        if self.backend.startswith("torch"):
            q_tensor = self._torch.tensor(q_norm, device=self._device)
            # This single matrix-vector multiply compares the query
            # against EVERY document simultaneously -> this is the
            # "parallel" part that makes GPU search fast.
            scores = self.embeddings @ q_tensor
            top_scores, top_indices = self._torch.topk(
                scores, k=min(top_k, scores.shape[0])
            )
            top_scores = top_scores.cpu().numpy()
            top_indices = top_indices.cpu().numpy()
        else:
            scores = self.embeddings @ q_norm
            top_k = min(top_k, scores.shape[0])
            top_indices = np.argpartition(scores, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(-scores[top_indices])]
            top_scores = scores[top_indices]

        elapsed_ms = (time.perf_counter() - start) * 1000

        results = [
            {"document": self.documents[i], "score": float(top_scores[j]), "index": int(i)}
            for j, i in enumerate(top_indices)
        ]

        return {
            "results": results,
            "search_time_ms": round(elapsed_ms, 4),
            "backend": self.backend,
            "num_documents_searched": len(self.documents),
        }


def run_benchmark(num_documents: int = 100_000, embedding_dim: int = 384):
    """
    Quick benchmark: generates random embeddings and times a search.
    Use this to produce the "before/after" speedup numbers for the
    hackathon demo video.
    """
    print(f"\n[Benchmark] Generating {num_documents} random embeddings "
          f"of dimension {embedding_dim}...")
    rng = np.random.default_rng(42)
    fake_embeddings = rng.random((num_documents, embedding_dim), dtype=np.float32)
    fake_documents = [f"document_{i}" for i in range(num_documents)]
    query = rng.random(embedding_dim, dtype=np.float32)

    engine = SimilaritySearchEngine(use_gpu=True)
    engine.index(fake_embeddings, fake_documents)

    result = engine.search(query, top_k=5)

    print(f"\n[Benchmark] Backend: {result['backend']}")
    print(f"[Benchmark] Searched {result['num_documents_searched']} documents")
    print(f"[Benchmark] Search time: {result['search_time_ms']} ms")
    print(f"[Benchmark] Top result: {result['results'][0]}")
    return result


if __name__ == "__main__":
    run_benchmark()
