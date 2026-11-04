"""Analyze fixed Reacher development interventions; never assign P1 admission."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def analyze(report):
    if report["status"] != "finished" or report["expected_branches"] != 20:
        raise ValueError("unexpected report status or denominator")
    if report["recorded_branches"] != 20 or report["complete_branches"] != 20:
        raise ValueError("incomplete branches cannot be treated as paired outcomes")
    if [r["seed"] for r in report["rows"]] != [20263160, 20263161]:
        raise ValueError("unexpected roots")
    results = []
    for root in report["rows"]:
        if root["status"] != "branches_recorded":
            raise ValueError("root incomplete")
        branches = root["branches"]
        expected = [(arm, t) for arm in ("selected", "parameter_zero") for t in range(5)]
        if [(b["arm"], b["tail_index"]) for b in branches] != expected:
            raise ValueError("unexpected or reordered branch roster")
        for b in branches:
            if b["status"] != "complete" or b["stop"] is not None or len(b["trajectory"]) != 16:
                raise ValueError("incomplete, stopped, or malformed trajectory")
            if b["attempted_steps"] != 15 or b["returned_steps"] != 15:
                raise ValueError("step count mismatch")
            for point in b["trajectory"]:
                if len(point["qpos"]) != 2 or len(point["target_qpos"]) != 2:
                    raise ValueError("expected two joint coordinates")
                delta = [((a - b + math.pi) % (2 * math.pi)) - math.pi
                         for a, b in zip(point["qpos"], point["target_qpos"])]
                dist = math.hypot(*delta)
                if not math.isfinite(dist) or not math.isclose(dist, point["qpos_distance"], rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError("distance mismatch")
        sel, zero = branches[:5], branches[5:]
        for s, z in zip(sel, zero):
            if s["raw_actions"][5:] != z["raw_actions"][5:]:
                raise ValueError("tail actions differ")
        for arm in (sel, zero):
            for b in arm[1:]:
                if b["trajectory"][:6] != arm[0]["trajectory"][:6]:
                    raise ValueError("same prefix yielded different saved trajectory")
        initial = sel[0]["trajectory"][0]["qpos_distance"]
        if zero[0]["trajectory"][0]["qpos_distance"] != initial:
            raise ValueError("initial distance mismatch")
        s5, z5 = (arm[0]["trajectory"][5]["qpos_distance"] for arm in (sel, zero))
        s0, z0 = (arm[0]["trajectory"][15]["qpos_distance"] for arm in (sel, zero))
        pairs = []
        for s, z in zip(sel, zero):
            se, ze = (b["trajectory"][15]["qpos_distance"] for b in (s, z))
            pairs.append({
                "tail_index": s["tail_index"],
                "selected_final": se, "parameter_zero_final": ze,
                "selected_minus_zero_final": se - ze,
                "selected_minus_zero_relative_to_tail0": (se - s0) - (ze - z0),
                "selected_final_reward": s["trajectory"][15]["reward"],
                "parameter_zero_final_reward": z["trajectory"][15]["reward"],
            })
        results.append({
            "seed": root["seed"], "initial_distance": initial,
            "selected_cp5": s5, "parameter_zero_cp5": z5,
            "selected_cp5_minus_initial": s5 - initial,
            "parameter_zero_cp5_minus_initial": z5 - initial,
            "selected_minus_zero_cp5": s5 - z5,
            "pairs": pairs,
        })
    return {"scope": "exploratory fixed common tails, exposed roots; no admission",
            "roots": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = args.report.read_bytes()
    output = analyze(json.loads(data))
    output["input_sha256"] = hashlib.sha256(data).hexdigest()
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
