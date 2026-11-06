"""Tests for the citation checker (rule 1). Run: python -m pytest -q tests/test_citations.py"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evals.citations import check_citations, split_claims  # noqa: E402


def test_fully_cited_answer_passes():
    a = "Paged KV caches cut memory waste [1]. Prefix caching reuses shared prompts [2]."
    r = check_citations(a, num_sources=2)
    assert r["passed"] and r["coverage"] == 1.0 and r["invalid"] == []


def test_citation_number_out_of_range_fails():
    a = "The system reaches 10x throughput [4]."
    r = check_citations(a, num_sources=3)
    assert not r["passed"] and r["invalid"] == [4]


def test_mostly_uncited_answer_fails():
    a = "GPUs are fast. Memory is the bottleneck. Tiling helps a lot [1]."
    r = check_citations(a, num_sources=1)
    assert not r["passed"] and r["coverage"] < 0.8
    assert "GPUs are fast." in r["uncited_claims"]


def test_citation_after_period_belongs_to_previous_claim():
    claims = split_claims("Raft elects a leader. [1] Logs are replicated. [2]")
    assert claims == ["Raft elects a leader. [1]", "Logs are replicated. [2]"]


def test_bullets_are_separate_claims():
    a = "- Uses MPI for halo exchange [1]\n- Reaches 5.55x speedup [2]"
    assert len(split_claims(a)) == 2
    assert check_citations(a, num_sources=2)["passed"]


def test_decimals_and_eg_do_not_split_sentences():
    claims = split_claims("Search is 3.7x faster, e.g. on a laptop CPU [1].")
    assert len(claims) == 1


def test_abstention_is_skipped_not_failed():
    r = check_citations("I could not find this in the indexed documents.", num_sources=0)
    assert r["abstained"] and r["passed"]


def test_empty_answer_fails():
    assert not check_citations("", num_sources=3)["passed"]


def test_decline_in_own_words_is_an_abstention():
    a = "The provided sources do not discuss how a hash map handles collisions in Python."
    assert check_citations(a, num_sources=3)["abstained"]


def test_partial_answer_with_citations_is_not_an_abstention():
    a = "The datasets used different scanners [1]. The sources do not list the exact combinations."
    r = check_citations(a, num_sources=1)
    assert not r["abstained"] and r["coverage"] == 1.0 and len(r["limit_statements"]) == 1
