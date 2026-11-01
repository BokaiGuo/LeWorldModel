"""Pinned historical metadata overlap audit, not a reset or freshness certificate.

Reads only an existing bounded index. Its intervals are conditional hypotheses;
numeric overlaps do not establish actual reset attempts or common RNG semantics.
"""
import hashlib
import json
from pathlib import Path

from experiments.spectral_reachability import codex_historical_pose_collision_audit_20261030 as pose

NAME = "codex_p1_reset_envelope_overlap_20261030"
LIMIT = 16 * 1024 * 1024
KINDS = {"candidate_random_reset_seed_envelope", "reserved_not_observed_reset_envelope"}


def overlaps(envelopes):
    if type(envelopes) is not list or not envelopes:
        raise ValueError("nonempty envelope list required")
    seen = set()
    for row in envelopes:
        if type(row) is not dict:
            raise ValueError("invalid envelope")
        source, kind = row.get("source_id"), row.get("kind")
        lo, hi = row.get("start_inclusive"), row.get("end_inclusive")
        if (type(source) is not str or not source or kind not in KINDS
                or type(lo) is not int or type(hi) is not int or lo < 0 or hi < lo
                or row.get("certified") is not False or (source, kind) in seen):
            raise ValueError("invalid or certified envelope")
        seen.add((source, kind))
    pairs = []
    for i, a in enumerate(envelopes):
        for j in range(i + 1, len(envelopes)):
            b = envelopes[j]
            lo = max(a["start_inclusive"], b["start_inclusive"])
            hi = min(a["end_inclusive"], b["end_inclusive"])
            if lo <= hi:
                pairs.append(dict(left_index=i, right_index=j,
                                  left_source=a["source_id"], right_source=b["source_id"],
                                  left_kind=a["kind"], right_kind=b["kind"],
                                  overlap_inclusive=[lo, hi], integer_count=hi - lo + 1))
    return pairs


def audit(index):
    if (type(index) is not dict or index.get("schema") != "codex_d2p_historical_exclusion_index_20261029.v1"
            or index.get("history_complete") is not False or index.get("freshness_certified") is not False
            or index.get("source_run_authorized") is not False):
        raise ValueError("index schema or authority mismatch")
    envelopes = index["conditional_seed_envelopes"]
    if len(envelopes) != 11 or index["counts"]["conditional_envelopes"] != 11:
        raise ValueError("envelope denominator")
    pairs = overlaps(envelopes)
    summary = pose.audit(index)
    return dict(schema=NAME + ".v1", historical_index_sha256=pose.INDEX_SHA,
                envelope_count=len(envelopes), overlaps=pairs,
                prefix_pose_summary=summary, history_complete=False,
                reset_attempts_verified=False, common_seed_semantics_verified=False,
                training_disjoint="unknown", freshness_certified=False,
                source_run_authorized=False, P1_admission=False,
                interpretation="Integer interval intersections only; no actual reset, deduplication or freshness inference.")


def build():
    path = pose.INDEX
    if not path.is_absolute() or path.resolve() != path or not path.is_file() or path.is_symlink():
        raise ValueError("canonical index required")
    with path.open("rb") as f:
        raw = f.read(LIMIT + 1)
    if len(raw) > LIMIT or hashlib.sha256(raw).hexdigest() != pose.INDEX_SHA:
        raise ValueError("index digest or size")
    def unique(items):
        result = {}
        for k, v in items:
            if k in result:
                raise ValueError("duplicate JSON key")
            result[k] = v
        return result
    def invalid(_):
        raise ValueError("nonfinite JSON")
    return audit(json.loads(raw, object_pairs_hook=unique, parse_constant=invalid))


if __name__ == "__main__":
    print(json.dumps(build(), sort_keys=True, indent=2, allow_nan=False))
