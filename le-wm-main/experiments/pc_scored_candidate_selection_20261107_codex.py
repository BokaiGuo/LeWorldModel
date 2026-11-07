"""Selection over explicitly available, already scored complete chains.

The caller controls which candidates are eligible. This module does not score
chains, generate candidates, access evaluation truth, or merge query budgets.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ScoredCandidate:
    source: str
    index: int
    cost: float


def select_scored(candidates):
    """Choose the first lowest-cost candidate, rejecting ambiguous identities."""
    candidates = tuple(candidates)
    if not candidates:
        raise ValueError("no available scored candidates")
    seen = set()
    for candidate in candidates:
        if (not isinstance(candidate, ScoredCandidate) or
                not isinstance(candidate.source, str) or not candidate.source or
                type(candidate.index) is not int or candidate.index < 0 or
                not math.isfinite(candidate.cost)):
            raise ValueError("invalid scored candidate")
        identity = candidate.source, candidate.index
        if identity in seen:
            raise ValueError("duplicate candidate identity")
        seen.add(identity)
    return min(candidates, key=lambda candidate: candidate.cost)
