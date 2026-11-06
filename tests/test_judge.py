"""Tests for the LLM judge (rules 2 and 3), using a fake judge so no API key is needed.

Run: python -m pytest -q tests/test_judge.py
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evals.judge import judge, parse_judge_output, score  # noqa: E402

ANSWER = "Raft elects a leader [1]. Logs are replicated to followers [1]. It runs on GPUs [2]."
SOURCES = ["Raft elects a leader and replicates its log to followers.", "A study of GPU kernels."]


def fake_call(reply: dict | str):
    """A judge stand-in that always returns the given reply."""
    text = reply if isinstance(reply, str) else json.dumps(reply)
    calls = []

    def call(messages):
        calls.append(messages)
        return text, {"prompt_tokens": 100, "completion_tokens": 20}

    call.calls = calls
    return call


GOOD = {"claims": [{"id": 1, "verdict": "supported", "reason": ""},
                   {"id": 2, "verdict": "supported", "reason": ""},
                   {"id": 3, "verdict": "unsupported", "reason": "source 2 never says Raft runs on GPUs"}],
        "answers_question": "partial", "reason": "one made-up claim"}


def test_scores_one_hallucinated_claim():
    r = judge("How does Raft work?", SOURCES, ANSWER, call=fake_call(GOOD))
    assert r["faithfulness"] == round(2 / 3, 3)
    assert r["fully_faithful"] is False
    assert r["answers_question"] == "partial"
    assert r["claims"][2]["text"] == "It runs on GPUs [2]."
    assert r["usage"] == {"prompt_tokens": 100, "completion_tokens": 20}


def test_wrong_citation_counts_as_faithful_but_lowers_precision():
    s = score({"claims": [{"verdict": "supported"}, {"verdict": "wrong_citation"}]})
    assert s["faithfulness"] == 1.0 and s["citation_precision"] == 0.5


def test_framing_sentences_do_not_count():
    s = score({"claims": [{"verdict": "supported"}, {"verdict": "not_a_claim"}]})
    assert s["faithfulness"] == 1.0 and s["fully_faithful"] is True


def test_parses_json_inside_fences_and_think_block():
    text = "<think>let me check</think>\n```json\n" + json.dumps(GOOD) + "\n```"
    assert len(parse_judge_output(text, 3)["claims"]) == 3


def test_skipped_claim_is_rejected():
    bad = {**GOOD, "claims": GOOD["claims"][:2]}
    with pytest.raises(ValueError):
        parse_judge_output(json.dumps(bad), 3)


def test_malformed_reply_is_retried_once_then_reported():
    call = fake_call("Sorry, I cannot grade this.")
    r = judge("q", SOURCES, ANSWER, call=call)
    assert "error" in r and len(call.calls) == 2
    assert r["usage"]["prompt_tokens"] == 200


def test_judge_sees_numbered_claims_and_sources():
    call = fake_call(GOOD)
    judge("How does Raft work?", SOURCES, ANSWER, call=call)
    user = call.calls[0][1]["content"]
    assert "[1] Raft elects a leader" in user and "3. It runs on GPUs [2]." in user
