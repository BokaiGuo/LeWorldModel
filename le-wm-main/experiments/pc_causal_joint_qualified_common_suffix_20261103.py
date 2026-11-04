"""Common-suffix intervention on a previously selected joint-search prefix."""
import json
from pathlib import Path
import time

import numpy as np

from experiments import pc_causal_history_h10_suffix_20261103 as h

SOURCE = h.ROOT / "results/pc_causal_joint_prefix_gradient_20261103_DEV01.json"
SOURCE_SHA = "931c7e09fd95617c9a3ce37549e561f7510df311b4b369acf66ad9ca3a21a123"
PROTOCOL = h.ROOT.parent / "docs/PC_CAUSAL_JOINT_QUALIFIED_COMMON_SUFFIX_PROTOCOL_20261103.md"
OUTPUT = h.ROOT / "results/pc_causal_joint_qualified_common_suffix_20261103_DEV01.json"
ARRAYS = OUTPUT.with_suffix(".npz")


def build_suffixes(plans, qualified, greedy):
    if plans.shape != (13, 10, 15, 2) or not np.isfinite(plans).all():
        raise ValueError("invalid joint-search plans")
    if len(qualified) != 2 or len(greedy) != 2:
        raise ValueError("invalid plan index")
    random = np.random.default_rng(2026110349).uniform(-.5, .5, (2, 10, 2)).astype(np.float32)
    result = np.stack((np.zeros((10, 2), np.float32),
                       plans[qualified[1], qualified[0], 5:],
                       plans[greedy[1], greedy[0], 5:], random[0], random[1]))
    if not np.isfinite(result).all() or np.any(np.abs(result) > 1):
        raise ValueError("invalid bounded suffixes")
    return result


def main():
    if OUTPUT.exists() or ARRAYS.exists():
        raise FileExistsError("refusing overwrite")
    if h.sha256_file(SOURCE) != SOURCE_SHA:
        raise ValueError("source changed")
    source = json.loads(SOURCE.read_text())
    if h.sha256_file(SOURCE.with_suffix(".npz")) != source["arrays_sha256"]:
        raise ValueError("source arrays changed")
    rows, arrays, starts, sequences = [], {}, [], []
    start = time.monotonic()
    print("Run fixed common suffixes on selected, zero and greedy prefixes at exposed root 8349.", flush=True)
    root = 8349
    root_index = list(h.ROOTS).index(root)
    row = source["rows"][root_index]
    if row["root"] != root or row["decisions"][0]["qualified"] is None:
        raise ValueError("source-qualified prefix missing")
    qualified = row["decisions"][0]["qualified"]
    greedy = [1, row["control_iterations"][1]]
    names = ("zero", "greedy", "raw_qualified")
    suffix_names = ("zero", "qualified_model", "greedy_model", "random0", "random1")
    with np.load(SOURCE.with_suffix(".npz"), allow_pickle=False) as saved:
        plans = saved[f"r{root}_plans"]
        suffixes = build_suffixes(plans, qualified, greedy)
        arrays["suffixes"] = suffixes
        offset = sum(len(r["branches"]) for r in source["rows"][:root_index])
        initial = saved["starts"][offset].astype(np.float64)
        for name in names:
            old_actions = saved[f"r{root}_{name}_actions"]
            old_history = saved[f"r{root}_{name}_history"]
            for s, suffix in enumerate(suffixes):
                actions = np.concatenate((old_actions[:7], suffix))
                with h.NativeRGBBridge(initial) as env:
                    history = [env.initial_rgb]
                    for action in actions:
                        history.append(env.step(action[None]))
                    if env.step_count != 17 or not np.array_equal(np.stack(history[:8]), old_history[:8]):
                        raise RuntimeError("prefix physical replay mismatch")
                key = f"{name}_{suffix_names[s]}"
                arrays[f"{key}_actions"], arrays[f"{key}_history"] = actions, np.stack(history)
                starts.append(initial)
                sequences.append(actions)
                rows.append({"prefix": name, "suffix": suffix_names[s]})
    starts, sequences = np.asarray(starts), np.asarray(sequences)
    cp2 = h.physical.replay_position_endpoints(starts, sequences[:, :2])
    cp7 = h.physical.replay_position_endpoints(starts, sequences[:, :7])
    cp17 = h.physical.replay_position_endpoints(starts, sequences)
    arrays["starts"], arrays["sequences"] = starts, sequences
    arrays["cp2_eval_only"], arrays["cp7_eval_only"], arrays["cp17_eval_only"] = cp2, cp7, cp17
    for i, entry in enumerate(rows):
        ref = (i // len(suffixes)) * len(suffixes)
        for tag, position in (("cp2", cp2[i]), ("cp7", cp7[i]), ("cp17", cp17[i])):
            entry[f"{tag}_distance"] = float(np.linalg.norm(position[:2] - h.GOAL[:2]))
        entry["physical_nonprogress_cp7"] = entry["cp7_distance"] >= entry["cp2_distance"]
        entry["cp17_distance_minus_own_zero"] = float(
            entry["cp17_distance"] - np.linalg.norm(cp17[ref, :2] - h.GOAL[:2]))
        entry["cp17_position_minus_own_zero"] = (cp17[i, :2] - cp17[ref, :2]).tolist()
    with ARRAYS.open("xb") as out:
        np.savez_compressed(out, **arrays)
    report = {"schema": "pc_causal_joint_qualified_common_suffix.v1", "root": root, "rows": rows,
              "scope": "outcome-exposed development; common suffix response only, no independent admission",
              "source_sha256": SOURCE_SHA, "arrays_sha256": h.sha256_file(ARRAYS),
              "protocol_sha256": h.sha256_file(PROTOCOL), "script_sha256": h.sha256_file(Path(__file__)),
              "counts": {"rgb_backend_steps": len(sequences)*17, "position_backend_replay_steps": len(sequences)*26,
                         "new_model_samples": 0},
              "inherited_upstream_cost": source["counts"], "wall_seconds": time.monotonic() - start}
    with OUTPUT.open("x") as out:
        json.dump(report, out, indent=2, sort_keys=True, allow_nan=False)
        out.write("\n")
    print(f"Saved {OUTPUT} sha256={h.sha256_file(OUTPUT)}", flush=True)


if __name__ == "__main__":
    main()
