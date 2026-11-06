"""
Rules 2 and 3 of the FastFind eval suite: an LLM judge.

A second, larger model from a different family grades each FastFind answer:

  Rule 2, faithfulness   Every claim (sentence) of the answer is checked
                         against the sources. Verdicts:
                           supported        stated or directly implied by the source it cites
                           wrong_citation   true per some other given source, but not the cited one
                           unsupported      not backed by any given source (hallucination)
                           not_a_claim      framing such as "The sources do not say ..."
                         faithfulness = (supported + wrong_citation) / real claims
                         citation precision = supported / (supported + wrong_citation)

  Rule 3, relevance      Does the answer respond to the question? yes / partial / no

Why a different model: a model grading its own answers tends to be lenient.
The default judge is Qwen3-235B; Nemotron writes the answers. Change it with
the JUDGE_MODEL environment variable.

The judge sees claims already split and numbered (same splitter as rule 1),
so every claim gets exactly one verdict and scores are comparable across runs.
"""

import json
import os
import re
import time
from typing import Callable

from evals.citations import split_claims

DEFAULT_JUDGE_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
VERDICTS = ("supported", "wrong_citation", "unsupported", "not_a_claim")
RELEVANCE = ("yes", "partial", "no")

JUDGE_SYSTEM = """You are a strict grader of answers produced by a retrieval system.
You get a question, numbered sources, and the answer split into numbered claims.
Claims cite sources with markers like [1].

For EACH claim give one verdict:
- "supported": the claim is stated in, or directly implied by, a source it cites.
  If the claim cites nothing, it is supported only if some source states it.
- "wrong_citation": the claim is true according to a DIFFERENT given source, but
  not according to the source(s) it cites.
- "unsupported": no given source backs the claim, or the claim adds numbers,
  names or conclusions that the sources do not contain. Use your own knowledge
  ONLY to read the sources, never to accept a claim the sources do not make.
- "not_a_claim": the sentence makes no factual statement about the topic
  (for example "The sources do not say how X works.").

Then judge the whole answer:
- "answers_question": "yes" if it directly answers the question, "partial" if it
  covers only part of it or answers a nearby question, "no" if it does not answer.

Return ONLY a JSON object, no other text:
{"claims": [{"id": 1, "verdict": "supported", "reason": "short reason"}],
 "answers_question": "yes", "reason": "one sentence"}"""

# A judge call takes chat messages and returns (text, usage dict).
JudgeCall = Callable[[list[dict]], tuple[str, dict]]


def build_messages(question: str, sources: list[str], claims: list[str]) -> list[dict]:
    src = "\n\n".join(f"[{i}] {s}" for i, s in enumerate(sources, start=1))
    cl = "\n".join(f"{i}. {c}" for i, c in enumerate(claims, start=1))
    user = f"Question: {question}\n\nSources:\n{src}\n\nClaims from the answer:\n{cl}"
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]


def parse_judge_output(text: str, num_claims: int) -> dict:
    """Pull the JSON object out of the judge's reply and validate it.

    Tolerates <think> blocks and ```json fences. Raises ValueError if the reply
    is unusable, so the caller can retry once.
    """
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("judge reply has no JSON object")
    data = json.loads(text[start : end + 1])

    by_id = {}
    for c in data.get("claims", []):
        cid, verdict = int(c.get("id", 0)), str(c.get("verdict", "")).strip().lower()
        if verdict not in VERDICTS:
            raise ValueError(f"unknown verdict {verdict!r}")
        by_id[cid] = {"id": cid, "verdict": verdict, "reason": str(c.get("reason", ""))}
    missing = [i for i in range(1, num_claims + 1) if i not in by_id]
    if missing:
        raise ValueError(f"judge skipped claims {missing}")

    rel = str(data.get("answers_question", "")).strip().lower()
    if rel not in RELEVANCE:
        raise ValueError(f"unknown answers_question {rel!r}")
    return {"claims": [by_id[i] for i in range(1, num_claims + 1)],
            "answers_question": rel, "reason": str(data.get("reason", ""))}


def score(parsed: dict) -> dict:
    """Turn per-claim verdicts into the rule 2 numbers for one answer."""
    counts = {v: 0 for v in VERDICTS}
    for c in parsed["claims"]:
        counts[c["verdict"]] += 1
    real = counts["supported"] + counts["wrong_citation"] + counts["unsupported"]
    backed = counts["supported"] + counts["wrong_citation"]
    return {
        "counts": counts,
        "faithfulness": round(backed / real, 3) if real else None,
        "citation_precision": round(counts["supported"] / backed, 3) if backed else None,
        "fully_faithful": real > 0 and counts["unsupported"] == 0,
    }


def nebius_judge_call(messages: list[dict]) -> tuple[str, dict]:
    """Default judge: a Token Factory model, temperature 0 for repeatable grades."""
    from backend.nebius_client import get_client

    resp = get_client().chat.completions.create(
        model=os.environ.get("JUDGE_MODEL", DEFAULT_JUDGE_MODEL),
        temperature=0,
        messages=messages,
    )
    u = resp.usage
    usage = {"prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
             "completion_tokens": getattr(u, "completion_tokens", 0) or 0}
    return resp.choices[0].message.content, usage


def judge(question: str, sources: list[str], answer: str, call: JudgeCall | None = None) -> dict:
    """Grade one answer. Retries once if the judge returns malformed JSON."""
    call = call or nebius_judge_call
    claims = split_claims(answer)
    if not claims:
        return {"error": "empty answer", "claims": [], "answers_question": "no"}

    messages = build_messages(question, sources, claims)
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    t0 = time.perf_counter()
    last_error = ""
    for _ in range(2):
        text, u = call(messages)
        usage = {k: usage[k] + u.get(k, 0) for k in usage}
        try:
            parsed = parse_judge_output(text, len(claims))
            break
        except (ValueError, json.JSONDecodeError) as e:
            last_error = str(e)
    else:
        return {"error": f"judge output unusable: {last_error}", "claims": [],
                "usage": usage, "judge_ms": round((time.perf_counter() - t0) * 1000, 1)}

    for c, claim_text in zip(parsed["claims"], claims):
        c["text"] = claim_text
    return {**parsed, **score(parsed), "usage": usage,
            "judge_ms": round((time.perf_counter() - t0) * 1000, 1)}
