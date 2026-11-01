"""Post-hoc two-batch CEM sample census against fixed GRASP delivered Q."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAME = "codex_grasp_cem_budget_prefix_20261030"
SOURCES = {
    "grasp": ("results/agent_p2_common_resource_grasp_cuda_20261029_v1_consume_MAIN01.json", "d3ee465f2fa6d31345f369f9514165c4817c51a29c3d00ffdd2d681e5829cc1f"),
    "cem": ("results/agent_lewm_bounded_native_cem_h3_20261029_v1_consume_MAIN01.json", "d0e215618c367049c09eed7e93accb90615973f4f8cb322fd19359fef1f7e7a4"),
    "prepare": ("results/agent_lewm_bounded_native_cem_h3_20261029_v1_prepare_MAIN01.json", "89b091af85bb36aa36a684a438d2179e7ab0c74c85de8303d3ac3c37e10d642b"),
}


def inputs():
    data = {}
    for name, (rel, sha) in SOURCES.items():
        raw = (ROOT / rel).read_bytes()
        if hashlib.sha256(raw).hexdigest() != sha:
            raise ValueError("source SHA mismatch: " + name)
        data[name] = json.loads(raw)
    return data


def census(scores, reference):
    if type(reference) not in (float, int) or not math.isfinite(reference):
        raise ValueError("invalid reference")
    if len(scores) != 300 or any(type(x) not in (float, int) or not math.isfinite(x) or x < 0 for x in scores):
        raise ValueError("invalid candidate batch")
    return {"n": len(scores), "below": sum(x < reference for x in scores),
            "equal": sum(x == reference for x in scores), "above": sum(x > reference for x in scores),
            "minimum": min(scores), "maximum": max(scores)}


def analyze(d):
    g, c, prep = d["grasp"], d["cem"], d["prepare"]
    contract = prep["contract"]
    if (contract["N"], contract["horizon"], contract["iterations"], contract["packet_raw_steps"]) != (300, 3, 30, 5):
        raise ValueError("CEM schedule mismatch")
    if c["same_horizon_as_P2_common_H3"] is not True or c["fair_ranking"] is not False:
        raise ValueError("CEM comparison boundary mismatch")
    if len(g["plans"]) != 6 or len(g["resources"]) != 6 or len(c["rows"]) != 2 or len(prep["slots"]) != 2:
        raise ValueError("fixed roster mismatch")
    order = g["contract"]["order"]
    results = []
    for root in (0, 1):
        matches = [i for i, p in enumerate(g["plans"]) if p["arm"] == "grasp" and p["root_slot"] == root]
        cr = [r for r in c["rows"] if r["root"] == root]
        ps = [s for s in prep["slots"] if s["root_slot"] == root]
        if len(matches) != 1 or len(cr) != 1 or len(ps) != 1:
            raise ValueError("root identity mismatch")
        i, row, slot = matches[0], cr[0], ps[0]
        plan = g["plans"][i]
        if order[i] != [root, "grasp"]:
            raise ValueError("resource row mismatch")
        for key in ("task_id", "task_sha256"):
            if plan[key] != row[key] or plan[key] != slot[key]:
                raise ValueError("task identity mismatch")
        if plan["goal"] != slot["goal_emb"] or plan["model_hash"] != prep["expected_model_hash"]:
            raise ValueError("goal or model mismatch")
        if slot["condition"] != "off" or plan["warm_raw10"] != slot["warm_raw10"]:
            raise ValueError("condition or history mismatch")
        resource = g["resources"][i]
        if resource["completed"]["predict_samples"] != 1902 or resource["attempted"]["predict_samples"] != 1902:
            raise ValueError("GRASP sample count mismatch")
        if len(row["planning_scores"]) != 30:
            raise ValueError("CEM iterations mismatch")
        batches = []
        for j in range(2):
            entry = row["planning_scores"][j]
            if entry["iteration"] != j or len(entry["mean192_float64"]) != 1:
                raise ValueError("batch order/shape mismatch")
            batches.append(census(entry["mean192_float64"][0], plan["model_Q"]))
        results.append({"root": root, "task_id": plan["task_id"], "task_sha256": plan["task_sha256"],
                        "grasp_delivered_model_q": plan["model_Q"], "cem_batches": batches,
                        "cem_prefix_resources": {"predictor_API": 6, "predictor_samples": 1800,
                                                 "action_encoder_API": 2, "backward": 0,
                                                 "final_replay": "not included"},
                        "grasp_resources": resource,
                        "remaining_predictor_samples_to_grasp": 102})
    return {"schema": NAME, "sources": {k: {"path": v[0], "sha256": v[1]} for k, v in SOURCES.items()},
            "results": results,
            "scope": "post-hoc model-only census; CEM second batch adapts to first; no candidate selected or deployed",
            "budget": "CEM prefix 1800 predictor samples versus delivered GRASP 1902, not matched complete policies",
            "limitations": ["CEM box and GRASP joint-ball supports differ", "planning reduction and batched API differ",
                            "GRASP backward and Adam work are additional", "CEM final mean uses all 30 iterations",
                            "no candidate physical outcomes or deployable prefix policy evaluated",
                            "FLOPs, comparable wall and inherited per-root encoding remain unknown",
                            "two known roots, not independent confirmation"],
            "trust": "pinned saved receipt and prepare; upstream fullstrict and neural calculations not rerun",
            "fair_ranking": False, "new_neural_calls": 0, "new_physics_calls": 0, "new_gpu_calls": 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "verify"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--sha256")
    args = parser.parse_args()
    out = Path(args.output)
    if not out.is_absolute() or out.parent != ROOT / "results" or not out.name.startswith(NAME + "_"):
        raise ValueError("invalid exclusive output")
    raw = (json.dumps(analyze(inputs()), sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    digest = hashlib.sha256(raw).hexdigest()
    if args.command == "build":
        with out.open("xb") as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
    elif args.sha256 != digest or out.read_bytes() != raw:
        raise ValueError("digest or reconstruction mismatch")
    print(json.dumps({"path": str(out), "sha256": digest, "command": args.command}))


if __name__ == "__main__":
    main()
