"""Saved-only input/transition divergence audit; no model or physics execution."""
import argparse
import math
import os
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import p5_rgb_refresh_openloop_physical_compare_20261030_codex_v1 as comparison

bridge, old, new = comparison.bridge, comparison.old, comparison.new
NAME = "p5_rgb_recursive_contract_audit_20261030_codex_v1"
ROOT, RESULTS = bridge.ROOT, bridge.ROOT / "results"
COMPARISON_SHA = "9520c574157fc50aa24ad2e89a8554e31f3e19935c882fbe90bc79f561ed33ad"


def need(value, message):
    if not value:
        raise ValueError(message)


def vector(output):
    need(set(output) == {"values", "sha256"}, "output fields")
    values = np.asarray(output["values"], dtype="<f4")
    need(values.shape == (6,) and np.isfinite(values).all() and
         bridge.digest(values.tobytes()) == output["sha256"], "saved vector identity")
    return values.astype(np.float64)


def analyze(open_prepared, open_review, refresh_prepared, refresh_review):
    need(len(open_prepared["rows"]) == len(open_review["rows"]) == len(refresh_prepared["rows"])
         == len(refresh_review["rows"]) == 164 and
         open_prepared["normalization_from_prior_strict_load"] ==
         refresh_prepared["normalization_from_prior_strict_load"], "fixed rows and normalization")
    norm = open_prepared["normalization_from_prior_strict_load"]
    std = np.asarray(norm["std"], dtype=np.float64)
    need(std.shape == (6,) and np.isfinite(std).all() and (std > 0).all(), "normalization std")
    rows = []
    for i, (op, ov, rp, rv) in enumerate(zip(open_prepared["rows"], open_review["rows"],
                                             refresh_prepared["rows"], refresh_review["rows"], strict=True)):
        need(all(op[k] == rp[k] for k in ("index", "root_slot", "condition", "task_id", "selection_digest")) and
             all(ov[k] == rv[k] == op[k] for k in ("index", "root_slot", "condition")) and
             (i, i // 82, ("off", "on")[(i // 41) % 2]) == (op["index"], op["root_slot"], op["condition"]) and
             len(op["actions"]) == len(ov["steps"]) == len(rp["steps"]) == len(rv["steps"]) == 5,
             "row identity")
        actions = np.asarray(op["actions"], dtype="<f4")
        need(actions.shape == (5, 2) and np.isfinite(actions).all() and
             bridge.digest(actions.tobytes()) == op["actions_sha256"], "open actions")
        initial = vector(ov["encode"]["output"])
        need(rp["steps"][0]["previous_rgb"] == op["previous_rgb"] and
             rp["steps"][0]["current_rgb"] == op["current_rgb"], "first frames")
        steps = []
        for h, (oslot, rstep, rs) in enumerate(zip(ov["steps"], rp["steps"], rv["steps"], strict=True), 1):
            need(oslot["step"] == h and oslot["status"] == "returned" and rs["horizon"] == h and
                 (rstep["horizon"], rstep["previous_plan_step"], rstep["current_plan_step"],
                  rstep["target_plan_step"]) == (h, h + 3, h + 4, h + 5), "horizon identity")
            action = np.asarray(rstep["action"], dtype="<f4")
            need(action.shape == (2,) and action.tobytes() == actions[h - 1].tobytes() and
                 bridge.digest(action.tobytes()) == rstep["action_sha256"], "same action")
            observed = vector(rs["encode_state"])
            source = initial if h == 1 else vector(ov["steps"][h - 2]["output"])
            open_next = vector(oslot["output"])
            refresh_next = vector(rs["predict_next"])
            steps.append({"horizon": h, "plan_step": h + 5,
                          "action_l2": float(np.linalg.norm(action.astype(np.float64))),
                          "action_component_outside_unit_box": bool(np.any(np.abs(action) > 1.0)),
                          "input_difference_normalized": (source - observed).tolist(),
                          "output_difference_normalized": (open_next - refresh_next).tolist(),
                          "open_input": source.tolist(), "refresh_input": observed.tolist(),
                          "open_output": open_next.tolist(), "refresh_output": refresh_next.tolist()})
        rows.append({"index": i, "root_slot": op["root_slot"], "condition": op["condition"], "steps": steps})
    summary = {}
    for condition in ("off", "on"):
        subset = [r for r in rows if r["condition"] == condition]
        need(len(subset) == 82, "fixed condition denominator")
        horizons = []
        for h in range(5):
            steps = [r["steps"][h] for r in subset]
            def meanabs(key):
                return [math.fsum(abs(s[key][k]) for s in steps) / 82 for k in range(6)]
            def meanl2(key):
                return math.fsum(math.sqrt(math.fsum(x * x for x in s[key])) for s in steps) / 82
            horizons.append({"horizon": h + 1, "denominator": 82,
                             "mean_abs_input_difference_normalized": meanabs("input_difference_normalized"),
                             "mean_abs_output_difference_normalized": meanabs("output_difference_normalized"),
                             "mean_input_difference_l2_normalized": meanl2("input_difference_normalized"),
                             "mean_output_difference_l2_normalized": meanl2("output_difference_normalized"),
                             "input_exact_equal_count": sum(all(x == 0 for x in s["input_difference_normalized"]) for s in steps),
                             "output_exact_equal_count": sum(all(x == 0 for x in s["output_difference_normalized"]) for s in steps),
                             "action_l2_over_0_85_count": sum(s["action_l2"] > 0.85 for s in steps),
                             "action_component_outside_unit_box_count": sum(s["action_component_outside_unit_box"] for s in steps)})
        summary[condition] = {"denominator": 82, "horizons": horizons}
    return {"schema": NAME + ".result", "status": "descriptive_saved_only_not_causal",
            "source_sha256": {"open_accept": old.ACCEPT_SHA, "refresh_accept": comparison.NEW_ACCEPT_SHA,
                              "open_journal": open_review["journal_sha256"], "refresh_journal": refresh_review["journal_sha256"],
                              "open_prepared": old.runner.PREP_SHA, "refresh_prepared": new.PREP_SHA,
                              "model_source": bridge.FILES["model_source"][1], "checkpoint": bridge.FILES["checkpoint"][1]},
            "denominators": {"branches": 164, "off": 82, "on": 82, "known_roots": 2, "fresh_roots": 0},
            "coordinate_order": ["agent_x", "agent_y", "block_x", "block_y", "sin_angle", "cos_angle"],
            "summary": summary, "rows": rows, "P5_admission": False,
            "new_neural_calls": 0, "new_cuda_calls": 0, "new_physics_calls": 0,
            "boundary": "input/output difference is observational not a Jacobian; 0.85 is a source-default reference not proven historical support"}


def build():
    own = [Path(__file__), ROOT / "tests" / ("test_" + NAME + ".py"), ROOT / "innerdocs" / (NAME + ".md")]
    closure = {str(p.relative_to(ROOT)): bridge.digest(p.read_bytes()) for p in own}
    need(bridge.digest(Path(comparison.__file__).read_bytes()) == COMPARISON_SHA, "comparison source pin")
    result = analyze(*comparison.sealed_predictions())
    need(closure == {str(p.relative_to(ROOT)): bridge.digest(p.read_bytes()) for p in own}, "source closure")
    result["consumer_closure"] = closure
    return result


def target(name):
    need(Path(name).name == name and name.startswith(NAME + "_") and name.endswith(".json")
         and RESULTS.resolve() == RESULTS and not RESULTS.is_symlink(), "output name")
    return RESULTS / name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["write", "verify"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--sha256")
    args = ap.parse_args()
    path = target(args.name)
    data = bridge.canonical(build())
    if args.action == "write":
        need(args.sha256 is None, "unexpected SHA")
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        old.runner.sync_dir(path.parent)
    else:
        need(args.sha256 is not None and old.checked_file(path, args.sha256, 16 * 1024 * 1024) == data, "verify")
    print(bridge.digest(data))


if __name__ == "__main__":
    main()
