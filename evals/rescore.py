"""
Re-score a saved eval run with the current rule 1 and rule 4 code. No API calls.

    python -m evals.rescore evals/results/20261105-171351.json

Use it after fixing the scorer itself, so an old run and a new run are
measured with exactly the same rules. Judge verdicts (rules 2 and 3) are
kept as they were saved.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evals.citations import check_citations, is_abstention  # noqa: E402
from evals.run_evals import print_summary, summarize  # noqa: E402


def rescore(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        run = json.load(f)
    for r in run["rows"]:
        r["abstained"] = is_abstention(r["answer"])
        r["citations"] = check_citations(r["answer"], len(r["sources"]))
    run["summary"] = summarize(run["rows"])
    return run


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    run = rescore(sys.argv[1])
    print(f"Re-scored {sys.argv[1]} (meta: {run.get('meta')})")
    print_summary(run["summary"])
