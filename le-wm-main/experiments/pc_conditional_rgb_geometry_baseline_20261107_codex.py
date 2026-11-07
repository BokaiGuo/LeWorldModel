"""RGB-only geometric baseline on the exposed conditional-CEM panel."""
import json
import sys
from pathlib import Path

import h5py
import numpy as np

from experiments import pc_conditional_selected_alignment_20261107_codex as source
from experiments import pc_tworoom_geometric_stage_repair_20261107_codex as geometry
from experiments.pc_input_permissions import file_hash

ROOT = source.ROOT
PROTOCOL = ROOT.parent / "docs/PC_CONDITIONAL_RGB_GEOMETRY_BASELINE_PROTOCOL_20261107_CODEX.md"
BASE = ROOT / "results/pc_conditional_rgb_geometry_baseline_20261107_codex"
PLAN, ACTIONS, RESULT = [Path(str(BASE) + suffix) for suffix in ("_plan.json", "_actions.npz", "_result.json")]


def generate(current, goal):
    try:
        pool, _ = geometry.candidate_pool(current, goal)
    except ValueError as exc:
        return None, str(exc)
    chain = pool[1]
    if chain.shape != (25, 2) or not np.isfinite(chain).all() or np.abs(chain).max() > 1:
        raise ValueError("invalid generated action chain")
    return chain, None


def compare(rows, arm):
    wins = losses = ties = 0
    for row in rows:
        ours, other = row["success"], row["comparators"][arm]
        wins += int(ours and not other)
        losses += int(other and not ours)
        ties += int(ours == other)
    return {"wins": wins, "losses": losses, "ties": ties}


def create_plan():
    if PLAN.exists() or ACTIONS.exists():
        raise FileExistsError("plan artifacts already exist")
    plan, _, inputs_path = source.source_records()
    arrays, rows = {}, []
    with np.load(inputs_path, allow_pickle=False) as inputs:
        for task in plan["rows"]:
            root = task["root"]
            chain, error = generate(inputs["current"][root], inputs["goal"][root])
            if chain is not None:
                arrays[f"r{root}"] = chain
            rows.append({"root": root, "row": task["row"], "abstained": chain is None, "reason": error})
    with ACTIONS.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    with PLAN.open("x") as stream:
        json.dump({"rows": rows, "actions_sha256": file_hash(ACTIONS),
                   "source_plan_sha256": file_hash(source.src.PLAN),
                   "inputs_sha256": file_hash(inputs_path), "protocol_sha256": file_hash(PROTOCOL),
                   "runner_sha256": file_hash(__file__), "generator_sha256": file_hash(geometry.__file__)},
                  stream, indent=2, allow_nan=False)
    print(f"Sealed {len(rows)} tasks; abstentions: {sum(row['abstained'] for row in rows)}", flush=True)


def evaluate():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if RESULT.exists():
        raise FileExistsError(RESULT)
    old, old_truth, inputs_path = source.source_records()
    sealed = json.loads(PLAN.read_text())
    for field, path in (("actions_sha256", ACTIONS), ("source_plan_sha256", source.src.PLAN),
                        ("inputs_sha256", inputs_path), ("protocol_sha256", PROTOCOL),
                        ("runner_sha256", __file__), ("generator_sha256", geometry.__file__)):
        if sealed[field] != file_hash(path):
            raise ValueError(f"sealed input changed: {field}")
    common = source.src.old.base.source.block.prior.common
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    try:
        with (np.load(ACTIONS, allow_pickle=False) as actions,
              np.load(inputs_path, allow_pickle=False) as inputs,
              h5py.File(common.DATA, "r") as dataset):
            for item, task, prior in zip(sealed["rows"], old["rows"], old_truth["rows"], strict=True):
                root = item["root"]
                if root != task["root"] or root != prior["root"] or item["row"] != task["row"]:
                    raise ValueError("root identity mismatch")
                trace, pixel_error = [], None
                if not item["abstained"]:
                    env.reset(seed=42)
                    actor = env.unwrapped
                    actor._set_state(dataset["proprio"][task["row"]])
                    actor._set_goal_state(dataset["proprio"][task["row"] + 25])
                    rendered = actor.render()
                    reference = inputs["current"][root]
                    if rendered.shape != reference.shape:
                        raise ValueError("initial rendering shape mismatch")
                    pixel_error = float(np.abs(rendered.astype(float) - reference.astype(float)).mean())
                    chain = actions[f"r{root}"]
                    if chain.shape != (25, 2) or np.abs(chain).max() > 1 or not np.isfinite(chain).all():
                        raise ValueError("invalid sealed chain")
                    for action in chain:
                        _, _, terminated, truncated, info = actor.step(action.copy())
                        trace.append({"distance": float(info["distance_to_target"]),
                                      "terminated": bool(terminated), "truncated": bool(truncated)})
                rows.append({"root": root, "abstained": item["abstained"], "trace": trace,
                             "initial_pixel_mean_absolute_error": pixel_error,
                             "success": any(x["terminated"] for x in trace),
                             "final_success": bool(trace and trace[-1]["terminated"]),
                             "comparators": {name: bool(prior["outcomes"][name]["ever"])
                                             for name in source.src.METHODS}})
                print(f"root={root} success={rows[-1]['success']}", flush=True)
    finally:
        env.close()
    summary = {"tasks": len(rows), "abstentions": sum(x["abstained"] for x in rows),
               "geometry_success": sum(x["success"] for x in rows),
               "geometry_final_success": sum(x["final_success"] for x in rows),
               "comparators": {name: {"success": sum(x["comparators"][name] for x in rows),
                                      **compare(rows, name)} for name in source.src.METHODS}}
    with RESULT.open("x") as stream:
        json.dump({"scope": "exposed development; RGB geometry; physical truth evaluation only",
                   "plan_sha256": file_hash(PLAN), "source_physical_sha256": file_hash(source.src.PHYS),
                   "summary": summary, "rows": rows}, stream, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("plan", "evaluate"):
        raise SystemExit("expected plan|evaluate")
    {"plan": create_plan, "evaluate": evaluate}[sys.argv[1]]()
