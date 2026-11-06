"""
Rule 1 of the FastFind eval suite: valid citations.

FastFind's prompt tells the model to cite a source like [1] right after every
claim. This module checks, with plain Python (no API calls, no cost), whether
an answer actually follows that rule:

  coverage         share of sentences that carry at least one citation
  invalid          citation numbers that point to no source (e.g. [7] when
                   only 3 sources were given) -- a sign of hallucination
  unused sources   sources the model was given but never cited (info only)

An answer that declines ("I could not find this ...") has nothing to cite,
so it is marked as an abstention and skipped instead of failing.

Usage:
    from evals.citations import check_citations
    report = check_citations(answer_text, num_sources=3)
    report["passed"]   # True / False
"""

import re

CITATION = re.compile(r"\[(\d+)\]")

# Phrases FastFind uses when it does not answer. Matching is case-insensitive.
ABSTAIN_PHRASES = (
    "could not find this in the indexed documents",
    "do not contain",
    "does not contain",
    "do not answer",
    "does not answer",
    "not covered by the sources",
)

# A sentence ends at . ! or ? followed by a space and a new sentence.
# The lookbehinds avoid splitting "e.g. " and "i.e. "; decimals like 3.7
# never split because no space follows the dot.
_SENTENCE_END = re.compile(r"(?<!e\.g\.)(?<!i\.e\.)(?<=[.!?])\s+(?=[A-Z\[(\"'])")
_ONLY_CITATIONS = re.compile(r"^(\s*\[\d+\])+\s*[.,;]?\s*$")
_LEADING_CITATIONS = re.compile(r"^(\s*\[\d+\])+\s*[.,;]?")
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")


def split_claims(answer: str) -> list[str]:
    """Split an answer into claims: one per sentence, and one per bullet line."""
    claims: list[str] = []
    for line in answer.splitlines():
        line = _BULLET.sub("", line).strip()
        if not line:
            continue
        for part in _SENTENCE_END.split(line):
            part = part.strip()
            if not part:
                continue
            # "Claim. [1] Next." splits into "Claim." and "[1] Next.": give the
            # leading citation back to the claim it belongs to.
            lead = _LEADING_CITATIONS.match(part)
            if lead and claims:
                claims[-1] = claims[-1] + " " + lead.group(0).strip()
                part = part[lead.end():].strip()
            if part and not _ONLY_CITATIONS.match(part):
                claims.append(part)
    return claims


def is_abstention(answer: str) -> bool:
    text = answer.lower()
    return any(p in text for p in ABSTAIN_PHRASES)


def check_citations(answer: str, num_sources: int, min_coverage: float = 0.8) -> dict:
    """Score one answer against the citation rule.

    min_coverage: share of claims that must carry a citation to pass. 0.8
    allows a short framing sentence ("Two approaches are compared.") to go
    uncited while still catching answers that mostly cite nothing.
    """
    claims = split_claims(answer)
    if is_abstention(answer):
        return {"abstained": True, "passed": True, "num_claims": len(claims),
                "coverage": None, "invalid": [], "unused_sources": []}

    cited_per_claim = [[int(n) for n in CITATION.findall(c)] for c in claims]
    all_cited = [n for nums in cited_per_claim for n in nums]
    with_citation = sum(1 for nums in cited_per_claim if nums)

    coverage = with_citation / len(claims) if claims else 0.0
    invalid = sorted({n for n in all_cited if n < 1 or n > num_sources})
    unused = [i for i in range(1, num_sources + 1) if i not in set(all_cited)]
    uncited = [c for c, nums in zip(claims, cited_per_claim) if not nums]

    return {
        "abstained": False,
        "passed": bool(claims) and coverage >= min_coverage and not invalid,
        "num_claims": len(claims),
        "coverage": round(coverage, 3),
        "invalid": invalid,
        "unused_sources": unused,
        "uncited_claims": uncited,
    }
