"""
Run the FastFind answer-quality eval end to end.

    python -m evals.run_evals                 # 20 answerable + all unanswerable questions
    python -m evals.run_evals --limit 100     # every answerable question
    python -m evals.run_evals --no-judge      # rules 1 and 4 only (cheaper)

For each test question it runs the SAME pipeline as the live app (embed ->
hybrid search -> relevance cutoff -> Nemotron answer), then scores the answer:

  rule 1  citations      evals/citations.py   (plain Python, free)
  rule 2  faithfulness   evals/judge.py       (LLM judge)
  rule 3  relevance      evals/judge.py       (LLM judge)
  rule 4  abstention     off-topic questions should get "I could not find this ..."

It also records time and tokens per answer. Prints a summary and saves every
answer, verdict and timing to evals/results/<timestamp>.json so runs can be
compared after a prompt or model change.

Needs NEBIUS_API_KEY. A full default run is ~40 questions: about 20 answer
calls and 20 judge calls (off-topic questions usually stop before any model
call). Use --limit to start small.
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from evals.citations import check_citations, is_abstention  # noqa: E402

DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(os.path.dirname(__file__), "results")
NO_ANSWER = "I could not find this in the indexed documents."


def load_questions(limit: int, seed: int) -> list[dict]:
    with open(os.path.join(DATA, "eval_arxiv.json"), encoding="utf-8") as f:
        answerable = json.load(f)["questions"]
    with open(os.path.join(DATA, "eval_offtopic.json"), encoding="utf-8") as f:
        unanswerable = json.load(f)["questions"]
    random.Random(seed).shuffle(answerable)
    for q in answerable[:limit]:
        q["kind"] = "answerable"
    return answerable[:limit] + unanswerable


def generate(question: str, sources: list[str]) -> tuple[str, dict]:
    """Nemotron answer, built exactly like backend.nebius_client.answer, plus token usage."""
    from backend import nebius_client as nc

    client = nc.get_client()
    model = os.environ.get("CHAT_MODEL", nc.DEFAULT_CHAT_MODEL)
    context = "\n".join(f"[{i}] {s}" for i, s in enumerate(sources, start=1))
    resp = nc._chat(client, model, question, context, {})
    u = resp.usage
    usage = {"prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
             "completion_tokens": getattr(u, "completion_tokens", 0) or 0}
    return nc.clean_answer(resp.choices[0].message.content), usage


def ask(app, question: str, top_k: int = 5) -> dict:
    """One question through the live app's retrieval and cutoff logic (no rate limits)."""
    t0 = time.perf_counter()
    qvec = app.embed_texts([question], retries=2)[0]
    embed_ms = (time.perf_counter() - t0) * 1000

    if app.SEARCH_MODE == "hybrid" and app.bm25 is not None:
        found = app.hybrid_search(app.engine, app.bm25, question, qvec, top_k=top_k)
    else:
        found = app.engine.search(qvec, top_k=top_k)
    sources = [{"text": r["document"]["text"], "source": r["document"]["source"], "score": r["score"]}
               for r in found["results"]]
    min_score = app.min_score_for(app.engine.embeddings.shape[1])
    top = max((s["score"] for s in sources), default=0.0)
    sources = [s for s in sources if s["score"] >= max(min_score, top * app.REL_SCORE)]

    t1 = time.perf_counter()
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    if not sources or top < min_score:
        sources, answer = [], NO_ANSWER
    else:
        answer, usage = generate(question, [s["text"] for s in sources])
    return {"answer": answer, "sources": sources, "usage": usage,
            "timings_ms": {"embed": round(embed_ms, 1), "search": found["search_time_ms"],
                           "generate": round((time.perf_counter() - t1) * 1000, 1)}}


def pct(n: int, d: int) -> str:
    return f"{n / d:.0%} ({n}/{d})" if d else "n/a"


def summarize(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["kind"] == "answerable"]
    answered = [r for r in ans if not r["abstained"]]
    judged = [r for r in answered if r.get("judge") and "error" not in r["judge"]]
    faith = [r["judge"]["faithfulness"] for r in judged if r["judge"]["faithfulness"] is not None]
    prec = [r["judge"]["citation_precision"] for r in judged if r["judge"]["citation_precision"] is not None]
    rel = [r["judge"]["answers_question"] for r in judged]
    gen_ms = [r["timings_ms"]["generate"] for r in answered]
    toks = [r["usage"]["prompt_tokens"] + r["usage"]["completion_tokens"] for r in answered]
    jtoks = [r["judge"]["usage"]["prompt_tokens"] + r["judge"]["usage"]["completion_tokens"] for r in judged]

    s = {
        "answerable": len(ans),
        "answered": len(answered),
        "wrongly_abstained": len(ans) - len(answered),
        "citation_pass": sum(r["citations"]["passed"] for r in answered),
        "citation_coverage_mean": round(float(np.mean([r["citations"]["coverage"] for r in answered])), 3) if answered else None,
        "invalid_citation_answers": sum(bool(r["citations"]["invalid"]) for r in answered),
        "judged": len(judged),
        "judge_errors": sum(1 for r in answered if r.get("judge") and "error" in r["judge"]),
        "faithfulness_mean": round(float(np.mean(faith)), 3) if faith else None,
        "fully_faithful": sum(r["judge"]["fully_faithful"] for r in judged),
        "citation_precision_mean": round(float(np.mean(prec)), 3) if prec else None,
        "answers_question": {k: rel.count(k) for k in ("yes", "partial", "no")},
        "generate_ms_mean": round(float(np.mean(gen_ms)), 1) if gen_ms else None,
        "generate_ms_p95": round(float(np.percentile(gen_ms, 95)), 1) if gen_ms else None,
        "answer_tokens_mean": round(float(np.mean(toks)), 1) if toks else None,
        "judge_tokens_mean": round(float(np.mean(jtoks)), 1) if jtoks else None,
    }
    for kind in ("offtopic", "nearmiss"):
        group = [r for r in rows if r["kind"] == kind]
        s[f"{kind}_total"] = len(group)
        s[f"{kind}_abstained"] = sum(r["abstained"] for r in group)
    return s


def print_summary(s: dict) -> None:
    print("\n=== FastFind answer-quality eval ===")
    print(f"Answerable questions: {s['answerable']}  (answered {s['answered']}, "
          f"said 'not found' on {s['wrongly_abstained']})")
    print(f"  Rule 1  citations pass:        {pct(s['citation_pass'], s['answered'])}"
          f"   mean coverage {s['citation_coverage_mean']}")
    print(f"          answers citing a source that does not exist: {s['invalid_citation_answers']}")
    if s["judged"]:
        print(f"  Rule 2  faithfulness (mean):   {s['faithfulness_mean']:.1%}")
        print(f"          fully faithful answers: {pct(s['fully_faithful'], s['judged'])}")
        print(f"          citation precision:     {s['citation_precision_mean']}")
        a = s["answers_question"]
        print(f"  Rule 3  answers the question:  yes {a['yes']}, partial {a['partial']}, no {a['no']}")
        if s["judge_errors"]:
            print(f"          judge errors: {s['judge_errors']}")
    print(f"  Rule 4  off-topic declined:    {pct(s['offtopic_abstained'], s['offtopic_total'])}")
    print(f"          near-miss declined:    {pct(s['nearmiss_abstained'], s['nearmiss_total'])}")
    if s["generate_ms_mean"] is not None:
        print(f"  Cost    answer time mean {s['generate_ms_mean'] / 1000:.1f} s, p95 {s['generate_ms_p95'] / 1000:.1f} s;"
              f" tokens per answer {s['answer_tokens_mean']:.0f}"
              + (f", per judge call {s['judge_tokens_mean']:.0f}" if s["judge_tokens_mean"] else ""))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limit", type=int, default=20, help="answerable questions to run (max 100)")
    p.add_argument("--seed", type=int, default=0, help="which answerable questions are picked")
    p.add_argument("--no-judge", action="store_true", help="skip the LLM judge (rules 2 and 3)")
    p.add_argument("--min-faithfulness", type=float, default=None,
                   help="exit with code 1 if mean faithfulness is below this (for CI)")
    args = p.parse_args()

    if not os.environ.get("NEBIUS_API_KEY"):
        sys.exit("NEBIUS_API_KEY is not set. Run: export NEBIUS_API_KEY='...'")

    from backend import app  # loads the index exactly as the server does
    if not app.index_info["loaded"]:
        sys.exit("No index found in data/. Run the ingest step first.")
    from evals.judge import judge

    questions = load_questions(args.limit, args.seed)
    print(f"Running {len(questions)} questions "
          f"(chat model {os.environ.get('CHAT_MODEL', 'default')}, "
          f"judge {'off' if args.no_judge else os.environ.get('JUDGE_MODEL', 'default')})")

    rows = []
    for i, q in enumerate(questions, start=1):
        try:
            out = ask(app, q["q"])
        except Exception as e:
            print(f"  [{i}/{len(questions)}] FAILED: {e}")
            continue
        row = {"q": q["q"], "kind": q.get("kind", "answerable"), "expected_source": q.get("source"),
               **out, "abstained": is_abstention(out["answer"])}
        row["citations"] = check_citations(out["answer"], len(out["sources"]))
        if not args.no_judge and not row["abstained"]:
            row["judge"] = judge(q["q"], [s["text"] for s in out["sources"]], out["answer"])
        rows.append(row)
        verdict = "declined" if row["abstained"] else (
            f"faith {row['judge'].get('faithfulness')}" if row.get("judge") else "answered")
        print(f"  [{i}/{len(questions)}] {row['kind']:<10} {verdict:<14} {q['q'][:60]}")

    summary = summarize(rows)
    print_summary(summary)

    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, datetime.now().strftime("%Y%m%d-%H%M%S") + ".json")
    meta = {"chat_model": os.environ.get("CHAT_MODEL"), "judge_model": os.environ.get("JUDGE_MODEL"),
            "limit": args.limit, "seed": args.seed, "search_mode": app.SEARCH_MODE}
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "summary": summary, "rows": rows}, f, indent=1, default=str)
    print(f"\nSaved every answer and verdict to {os.path.relpath(path, ROOT)}")

    if args.min_faithfulness is not None:
        got = summary["faithfulness_mean"]
        if got is None or got < args.min_faithfulness:
            sys.exit(f"FAIL: faithfulness {got} is below the required {args.min_faithfulness}")
        print(f"PASS: faithfulness {got} >= {args.min_faithfulness}")


if __name__ == "__main__":
    main()
