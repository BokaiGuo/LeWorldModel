"""CPU-only, read-only comparison of fixed prospective CEM budget proposals.

Not a runner: no plan scoring, hardware use, physical execution or writes.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPECTED = (
    "7faf6dcc2f47ecc1e4805a3b3ff580f1c563603cd0308c029eeb26fe95eed236",
    "15829c899142de5a374fd0555dfe26c5416f70ea10b1964f87ee990307a44122",
)
BOUNDARIES = (0, 5, 10, 15)
HORIZONS = (3, 2, 1, 1)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def allocate(cap, horizon, iterations, elites):
    """Largest integer bank under forward cap, reserving one full replay."""
    require(all(type(v) is int and v > 0 for v in (cap, horizon, iterations, elites)),
            "positive integer dimensions and cap required")
    samples = (cap // horizon - 1) // iterations
    require(samples >= elites, "budget does not fit elite count")
    calls = (iterations * samples + 1) * horizon
    return {"samples": samples, "forward": calls, "unused": cap - calls}


def inspect(root):
    require(type(root) is int and root in (0, 1), "invalid root")
    base = ROOT / "results"
    path = base / f"codex_p1_full_cem_host_v2_20261101_exclusive_root{root}_PREP_MAIN01.json"
    require(sha(path) == EXPECTED[root], "v2 prepare SHA drift")
    cem = json.loads(path.read_text())
    ident = cem["cem_source_identity"]
    original = base / f"p1_first_feedback_carry_20261101_main_root{root}_PREP_MAIN02.json"
    require(str(original) == ident["prepare"] and sha(original) == ident["prepare_sha256"],
            "first-feedback prepare binding")
    prior = json.loads(original.read_text())
    cp, pp = cem["protocol"], prior["protocol"]
    require(cp["root_slot"] == pp["root_slot"] == root, "root slot")
    require(cp["goal"] == pp["goal"] == f"root{root}_previously_observed_cp10", "goal description")
    require(cp["boundaries"] == pp["boundaries"] == list(BOUNDARIES), "feedback opportunities")
    require(cp["feedback_horizons"] == list(HORIZONS), "CEM horizons")
    require([pp["feedback_horizons"][str(b)][0] for b in BOUNDARIES] == list(HORIZONS),
            "comparison horizons")
    require(cp["cem"]["iterations"] == 4 and cp["cem"]["elites"] == 4
            and cp["cem"]["samples_per_iteration"] == 16, "CEM configuration")
    costs = [cp["node_costs"][str(b)]["predict_calls"] for b in BOUNDARIES]
    require(costs == [195, 130, 65, 65], "CEM declared costs")
    structured = pp["node_costs"]["structured"]
    require(structured["predict_calls"] == structured["action_encode_calls"]
            == structured["predict_samples"] == 174 and structured["backward_calls"] == 240,
            "structured declared b0 cost drift")
    proposals = [allocate(structured["predict_calls"], h, 4, 4) for h in HORIZONS]
    return {
        "root": root, "cem_prepare_sha256": EXPECTED[root],
        "first_feedback_prepare_sha256": ident["prepare_sha256"],
        "identity_source_prepare_sha_verified": True,
        "identity_note": "source prepare link and goal labels only; realized root/goal must be certified",
        "feedback_horizons_match": True,
        "existing_cem_forward": costs,
        "structured_declared_forward": structured["predict_calls"],
        "structured_declared_backward": structured["backward_calls"],
        "structured_cost_scope": "static protocol per-arm baseline; later-boundary actual costs not verified",
        "proposal_samples": [p["samples"] for p in proposals],
        "proposal_forward": [p["forward"] for p in proposals],
        "proposal_unused_forward": [p["unused"] for p in proposals],
        "proposal_assumption": "same 174 forward cap each boundary; not certified as comparator per-boundary usage",
        "proposal_implemented": False, "actual_cost_verified": False,
        "ready_to_run_comparison": False, "equal_compute_claim": False,
        "blockers": [
            "new variable-bank runner and contracts not implemented or sealed",
            "comparator per-boundary cap and source/goal runtime identity not certified",
            "independent forward/backward/encoder coverage and runtime measurements missing",
            "outcome-dependent early stop requires full episode cost receipts",
        ],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=int, choices=(0, 1), required=True)
    args = p.parse_args()
    print(json.dumps(inspect(args.root), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
