"""
FastFind speed test: exact brute-force similarity search on CPU vs GPU.

    python scripts/benchmark.py                          # sizes 100k, 250k, 500k at dim 4096
    python scripts/benchmark.py --sizes 100000,1000000   # your own sizes
    python scripts/benchmark.py --dim 4096 --k 10

Self-contained: needs only numpy, and torch for the GPU part.
Without a GPU it still runs and reports the CPU numbers.

What is measured (each timing is the mean of several runs after warm-up):
  - single query: one question against all vectors (what the app does)
  - batch of 64:  64 questions at once (throughput)
  - the CPU run uses NumPy (multi-threaded BLAS) on the same machine
  - GPU time includes copying the top-k result back to the host
  - fp16 GPU run is reported separately, with how many of its top-k match fp32

Vectors are random unit vectors. Brute-force search cost depends only on the
number and size of the vectors, not on their content, so timings carry over to
real embeddings. The script also checks that CPU and GPU return the same
top-k (exact search, so they should).

Results are printed and saved to benchmark_results.json.
"""

import json
import os
import platform
import statistics
import sys
import time

import numpy as np

try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except Exception:
    torch = None
    HAS_CUDA = False


def mem_available_gb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable"):
                    return int(line.split()[1]) / 1e6
    except OSError:
        pass
    try:
        import psutil
        return psutil.virtual_memory().available / 1e9
    except Exception:
        return 8.0


def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def make_host(n: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty((n, dim), dtype=np.float32)
    step = 50_000
    for s in range(0, n, step):
        c = rng.standard_normal((min(step, n - s), dim), dtype=np.float32)
        c /= np.linalg.norm(c, axis=1, keepdims=True)
        out[s : s + len(c)] = c
    return out


def topk_cpu(E: np.ndarray, q: np.ndarray, k: int):
    s = E @ q
    idx = np.argpartition(s, -k)[-k:]
    return idx[np.argsort(-s[idx])]


def topk_cpu_batch(E: np.ndarray, Q: np.ndarray, k: int):
    s = Q @ E.T
    idx = np.argpartition(s, -k, axis=1)[:, -k:]
    return idx


def timeit(fn, reps: int, sync=None) -> dict:
    for _ in range(2):
        fn()
    if sync:
        sync()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        times.append((time.perf_counter() - t0) * 1000)
    return {"mean_ms": statistics.mean(times), "median_ms": statistics.median(times), "min_ms": min(times)}


def main(sizes: list[int], dim: int, k: int) -> None:
    info = {"cpu": cpu_name(), "cpu_threads": os.cpu_count(), "numpy": np.__version__,
            "gpu": None, "torch": getattr(torch, "__version__", None), "dim": dim, "k": k}
    if HAS_CUDA:
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    print(json.dumps(info, indent=1))
    if not HAS_CUDA:
        print("\nNo CUDA GPU found: running the CPU measurements only.\n")

    rng = np.random.default_rng(123)
    Q1 = rng.standard_normal(dim).astype(np.float32); Q1 /= np.linalg.norm(Q1)
    Qb = rng.standard_normal((64, dim)).astype(np.float32); Qb /= np.linalg.norm(Qb, axis=1, keepdims=True)

    rows = []
    for n in sizes:
        need_gb = n * dim * 4 / 1e9
        row = {"n": n, "matrix_gb": round(need_gb, 2)}
        print(f"\n=== {n:,} vectors x {dim} dims ({need_gb:.1f} GB as float32)")
        host = None
        if need_gb * 1.5 < mem_available_gb():
            host = make_host(n, dim)
            reps = 10 if n <= 250_000 else 5
            row["cpu_single"] = timeit(lambda: topk_cpu(host, Q1, k), reps)
            row["cpu_batch64"] = timeit(lambda: topk_cpu_batch(host, Qb, k), 3)
            print(f"  CPU  single query : {row['cpu_single']['mean_ms']:.2f} ms")
            print(f"  CPU  batch of 64  : {row['cpu_batch64']['mean_ms']:.1f} ms")
        else:
            print("  (not enough RAM for the CPU baseline at this size; GPU only)")

        if HAS_CUDA:
            if need_gb * 1.1 > torch.cuda.mem_get_info()[0] / 1e9:
                print("  (matrix does not fit in GPU memory; skipped)")
                rows.append(row); continue
            dev = torch.device("cuda")
            if host is not None:
                E = torch.empty((n, dim), dtype=torch.float32, device=dev)
                for s in range(0, n, 100_000):
                    E[s : s + 100_000] = torch.from_numpy(host[s : s + 100_000]).to(dev)
            else:
                g = torch.Generator(device=dev); g.manual_seed(0)
                E = torch.randn((n, dim), generator=g, device=dev)
                E /= E.norm(dim=1, keepdim=True)
            q = torch.from_numpy(Q1).to(dev); Q = torch.from_numpy(Qb).to(dev)
            sync = torch.cuda.synchronize

            def gpu_single():
                return torch.topk(E @ q, k).indices.cpu()

            def gpu_batch():
                return torch.topk(Q @ E.T, k, dim=1).indices.cpu()

            row["gpu_single"] = timeit(gpu_single, 50, sync)
            row["gpu_batch64"] = timeit(gpu_batch, 10, sync)
            print(f"  GPU  single query : {row['gpu_single']['mean_ms']:.3f} ms")
            print(f"  GPU  batch of 64  : {row['gpu_batch64']['mean_ms']:.2f} ms")

            if host is not None:
                same = set(topk_cpu(host, Q1, k).tolist()) == set(gpu_single().tolist())
                row["same_topk_as_cpu"] = same
                print(f"  GPU top-{k} identical to CPU top-{k}: {same}")
                row["speedup_single"] = row["cpu_single"]["mean_ms"] / row["gpu_single"]["mean_ms"]
                row["speedup_batch64"] = row["cpu_batch64"]["mean_ms"] / row["gpu_batch64"]["mean_ms"]
                print(f"  speedup: {row['speedup_single']:.0f}x single, {row['speedup_batch64']:.0f}x batch of 64")

            del E
            torch.cuda.empty_cache()
            if need_gb / 2 * 1.2 < torch.cuda.mem_get_info()[0] / 1e9 and host is not None:
                E16 = torch.empty((n, dim), dtype=torch.float16, device=dev)
                for s in range(0, n, 100_000):
                    E16[s : s + 100_000] = torch.from_numpy(host[s : s + 100_000]).to(dev).half()
                q16 = q.half()
                row["gpu_fp16_single"] = timeit(lambda: torch.topk(E16 @ q16, k).indices.cpu(), 50, sync)
                ref = set(topk_cpu(host, Q1, k).tolist())
                got = set(torch.topk(E16 @ q16, k).indices.cpu().tolist())
                row["fp16_topk_overlap"] = len(ref & got) / k
                print(f"  GPU fp16 single   : {row['gpu_fp16_single']['mean_ms']:.3f} ms  "
                      f"(top-{k} overlap with fp32: {row['fp16_topk_overlap']:.0%})")
                del E16
            torch.cuda.empty_cache()
        rows.append(row)
        del host

    out = {"environment": info, "results": rows}
    path = "benchmark_results.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    sizes = [100_000, 250_000, 500_000]
    dim, k = 4096, 10
    if "--sizes" in sys.argv:
        sizes = [int(x) for x in sys.argv[sys.argv.index("--sizes") + 1].split(",")]
    if "--dim" in sys.argv:
        dim = int(sys.argv[sys.argv.index("--dim") + 1])
    if "--k" in sys.argv:
        k = int(sys.argv[sys.argv.index("--k") + 1])
    main(sizes, dim, k)
