"""Bind the fixed Adam suffix results to the existing P1 intervention panel."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAME = "codex_p1_adam_paired_join_20261030_main"
SOURCES = {
    "view": ("results/codex_p1_guard_mechanism_join_20261030_main_RUN01.json", "b9eff7b3d612787407e6923fc4875102fe4709e45dc1118fda40e87e48cad313"),
    "prepare": ("results/codex_p1_adam_physical_20261030_main_RUN01/prepare.json", "ebd31d94a7bcb75623497893bc80adf43dceb2506f91c3a00633f0b457f5ac30"),
    "saved": ("results/codex_p1_adam_physical_20261030_main_SAVED_RUN01.json", "b2e38b2bb52fa8ac4dd8ebb90bf702b1477a4efe4df3606b8da42c607ee74881"),
    "plan": ("results/codex_p1_adam_cp5_20261030_main_RUN01.json", "455c93a37f8d1200ffe75290e30e939378ed536b9d81ede802da221f7b6c4253"),
}
IDS = ("index", "logical_pair", "root_slot", "condition", "arm", "role")
FIELDS = ("object_position_error", "object_angle_error_wrapped",
          "pusher_position_error", "native_joint_xy_error")


def need(condition, message):
    if not condition:
        raise ValueError(message)


def delta(a, b):
    if a is None or b is None:
        return None
    return {key: a[key] - b[key] for key in FIELDS}


def build(view, prepare, saved, plan):
    need(all(len(x) == 24 for x in (view["rows"], prepare["roster"], saved["rows"], plan["rows"])),
         "fixed denominator")
    need(saved["prepare_sha256"] == SOURCES["prepare"][1], "prepare binding")
    need(prepare["plan_sha256"] == SOURCES["plan"][1], "plan binding")
    rows = []
    for i, (v, p, s, q) in enumerate(zip(view["rows"], prepare["roster"], saved["rows"], plan["rows"], strict=True)):
        ident = v["identity"]
        need(ident["index"] == i and all(ident[k] == p[k] == s[k] for k in IDS), "identity mismatch")
        need(all(ident[k] == p[k] for k in ("task_id", "cp5_file_sha256", "cp5_pixel_sha256", "prefix5_float32_hex")), "prefix binding")
        need(all(q[k] == ident[k] for k in ("index", "logical_pair", "role")), "plan identity")
        need(q["selected_action_sha256"] == s["raw10_sha256"] == p["raw10_sha256"], "suffix binding")
        need(s["status"] in ("probe_completed", "native_stopped"), "unexpected status")
        need((s["endpoint_metrics"] is not None) == (s["status"] == "probe_completed"), "censor policy")
        zero = next(x for x in v["common_suffixes"] if x["probe"] == "zero")
        adam = {"status": s["status"], "accepted_steps": s["returned_suffix"],
                "horizon_metrics": s["endpoint_metrics"],
                "last_observed_metrics": s["last_accepted_state_metrics"],
                "endpoint_minus_own_zero_suffix": delta(s["endpoint_metrics"], zero["common_suffix_endpoint_native_metrics"]),
                "raw10_sha256": s["raw10_sha256"], "planning_calls_code_path": q["calls"],
                "planning_wall_seconds": q["planning_wall_seconds"], "physical_counts": s["counts"]}
        rows.append({**v, "adam_independent_suffix": adam})
    for i, row in enumerate(rows):
        z = rows[(i // 3) * 3 + 1]
        need(z["identity"]["role"] == "zero" and row["identity"]["logical_pair"] == z["identity"]["logical_pair"], "paired zero")
        row["adam_independent_suffix"]["endpoint_minus_zero_prefix_independent_endpoint"] = delta(
            row["adam_independent_suffix"]["horizon_metrics"], z["adam_independent_suffix"]["horizon_metrics"])
    return {"schema": NAME, "status": "descriptive_join", "rows": rows,
            "sources": SOURCES, "P1_admission": False, "independent_confirmation": False,
            "equal_total_budget_certified": False,
            "limitations": ["two known roots; duplicate trajectories; exploratory post-outcome optimizer",
                            "common suffix effect is not independent planning capability",
                            "native-stopped horizon endpoints remain censored even at step ten",
                            "call counts are code-path estimates, not complete independently audited total costs"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    out = Path(parser.parse_args().output)
    need(out.is_absolute() and out.parent == ROOT / "results" and out.name.startswith(NAME + "_"), "output scope")
    inputs = {}
    for name, (path, expected) in SOURCES.items():
        data = (ROOT / path).read_bytes()
        need(hashlib.sha256(data).hexdigest() == expected, f"source hash: {name}")
        inputs[name] = json.loads(data)
    result = build(**inputs)
    result["code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with out.open("x") as handle:
        json.dump(result, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(out), "rows": len(result["rows"]), "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
