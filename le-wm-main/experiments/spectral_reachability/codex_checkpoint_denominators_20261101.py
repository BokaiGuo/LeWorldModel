"""CPU-only availability census of the sealed six-plan RGB summary."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/agent_p2_common_resource_rgb_scalar_summary_20261029_v1_MAIN01.json"
SHA = "b3c1493d92816e0956f6e952a885505147879fdd243528d10f6098a82608b353"
NAME = "codex_checkpoint_denominators_20261101"


def census(data):
    rows = data["rows"]
    if (data["fixed_denominator"], data["plan_denominator"], len(rows)) != (18, 6, 18):
        raise ValueError("unexpected fixed denominator")
    expected = {(root, arm, cp) for root in (0, 1)
                for arm in ("joint", "composition", "grasp") for cp in (5, 10, 15)}
    actual = [(r["root_slot"], r["arm"], r["checkpoint"]) for r in rows]
    if len(set(actual)) != 18 or set(actual) != expected:
        raise ValueError("roster mismatch")
    if [r["index"] for r in rows if r["missing"]] != data["missing_slots"]:
        raise ValueError("missing slot mismatch")
    groups = []
    for cp in (5, 10, 15):
        for arm in ("joint", "composition", "grasp"):
            selected = sorted((r for r in rows if r["checkpoint"] == cp and r["arm"] == arm),
                              key=lambda r: r["root_slot"])
            slots = []
            for r in selected:
                if type(r["missing"]) is not bool:
                    raise ValueError("missing flag must be boolean")
                if r["missing"]:
                    if r["real_Q"] is not None or not r["missing_reason"]:
                        raise ValueError("invalid missing slot")
                    success = None
                else:
                    success = r["native_metrics"]["native_success"]
                    if r["real_Q"] is None or type(success) is not bool or r["missing_reason"] is not None:
                        raise ValueError("invalid observed slot")
                slots.append({"root": r["root_slot"], "index": r["index"],
                              "observed": not r["missing"], "native_success_at_checkpoint": success,
                              "missing_reason": r["missing_reason"]})
            groups.append({"checkpoint": cp, "arm": arm, "planned": 2,
                           "observed": sum(x["observed"] for x in slots), "slots": slots})
    return {"schema": NAME, "source": str(SOURCE.relative_to(ROOT)), "source_sha256": SHA,
            "groups": groups, "planned_slots": 18, "observed_slots": sum(g["observed"] for g in groups),
            "interpretation": "checkpoint observations, not episode success rates; missing is not failure",
            "limitations": ["two known roots, not independent confirmation", "early stop is outcome-dependent",
                            "no author numbers, cross-method ranking, or equal-compute claim",
                            "saved summary census; upstream model and physical execution not rerun"],
            "new_gpu_calls": 0, "new_neural_calls": 0, "new_physics_calls": 0}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    raw = SOURCE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SHA:
        raise ValueError("source SHA mismatch")
    result = census(json.loads(raw))
    output = args.output.resolve()
    if output.parent != ROOT / "results" or not output.name.startswith(NAME + "_"):
        raise ValueError("exclusive output required")
    with output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    print(json.dumps({"output": str(output), "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
