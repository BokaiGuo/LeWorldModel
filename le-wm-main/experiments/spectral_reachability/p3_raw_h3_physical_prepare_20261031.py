"""Seal all H3 old/new 20-step action pairs without executing physics."""
import hashlib
import json
from pathlib import Path

import numpy as np

from . import p3_raw_h1_physical_20261031_main as h1

ROOT = h1.ROOT
PROTOCOL = ROOT.parent / "docs/D2P_P3_RAW_H3_PHYSICAL_PROTOCOL_20261031_MAIN.md"
NAME = "p3_raw_h3_physical_prepare_20261031"
require = h1.require


def build_plans(roster, result, physical):
    require(result["schema"] == h1.source.NAME and result["status"] == "complete_cuda_model_only"
            and len(roster) == len(result["rows"]) == 72, "model source roster")
    refs = {p["root_slot"]: p for p in physical["plans"]}
    plans = []
    for (index, slot, task, horizon, row), saved in zip(roster, result["rows"], strict=True):
        require((saved["decision_index"], saved["slot"], saved["root"], saved["policy"],
                 saved["horizon"], saved["candidate"]) ==
                (index, slot["slot"], slot["root"], slot["policy"], horizon, row["index"]),
                "model row identity")
        root = slot["root"]
        require(task["binding"]["root_sha256"] == refs[root]["root_sha256"]
                and task["binding"]["task_id"] == refs[root]["task_id"], "physical root identity")
        original = np.asarray(row["suffix_actions"], dtype="<f4")
        require(np.array_equal(original, np.asarray(saved["old_action"], dtype="<f4")),
                "old action binding")
        if horizon != 3:
            continue
        prefix = np.asarray(row["actions"], dtype="<f4")
        require(prefix.shape == (15, 2) and original.shape == (5, 2), "H3 packet shape")
        for variant in ("old", "new"):
            tail = np.asarray(saved[variant + "_action"], dtype="<f4")
            actions = np.concatenate((prefix, tail), axis=0)
            require(tail.shape == (5, 2) and np.isfinite(actions).all()
                    and (np.abs(actions) <= 1).all(), "action bounds")
            plan = {"root_slot": root, "root_sha256": refs[root]["root_sha256"],
                    "task_id": refs[root]["task_id"], "decision_index": index,
                    "policy": slot["policy"], "candidate": row["index"], "variant": variant,
                    "commands": actions.astype(float).tolist(),
                    "prefix_float32_sha256": hashlib.sha256(prefix.tobytes()).hexdigest(),
                    "raw_float32_sha256": hashlib.sha256(actions.tobytes()).hexdigest()}
            plan["selection_digest"] = h1.host.kernel.source.digest(plan)
            plans.append(plan)
    require(len(plans) == 72, "complete 4x9x2 H3 roster")
    return plans


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    require(output.is_absolute() and output.parent == ROOT / "results"
            and output.name.startswith(NAME + "_") and not output.exists()
            and not output.is_symlink(), "exclusive output")
    plans1, bindings, _, _ = h1.prepare()
    require(len(plans1) == 72 and h1.host.sha(h1.NONLINEAR) == h1.NONLINEAR_SHA,
            "source reconstruction")
    result = json.loads(h1.NONLINEAR.read_bytes())
    roster = h1.source.load_roster()
    physical, _ = h1.host.inputs()
    plans = build_plans(roster, result, physical)
    for path in (Path(__file__).resolve(), PROTOCOL.resolve()):
        bindings[str(path)] = h1.host.sha(path)
    require(all(h1.host.sha(path) == digest for path, digest in bindings.items()), "source drift")
    record = {"schema": NAME, "plans": plans, "bindings": bindings,
              "prefix_steps": 15, "suffix_steps": 5, "physical_execution": False,
              "independent_confirmation": False, "equal_budget_certified": False,
              "P1_admission": False, "P3_admission": False}
    raw = (json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with output.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"output": str(output), "plans": len(plans),
                      "sha256": hashlib.sha256(raw).hexdigest()}))


if __name__ == "__main__":
    main()
