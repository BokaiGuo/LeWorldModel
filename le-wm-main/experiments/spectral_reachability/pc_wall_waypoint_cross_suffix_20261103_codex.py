"""Post-outcome common-suffix replay of privileged waypoint winners."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .pc_wall_triprefix_cem_intervention_20261103_codex import ROLES, SEEDS
from .pc_wall_unicycle_atlas_cem_20261103 import HIGH, LOW, replay

REPO = Path(__file__).resolve().parents[3]
SOURCE = REPO / "le-wm-main/results/pc_wall_triprefix_waypoint_ceiling_20261103_codex_DEV01.json"
PROTOCOL = REPO / "docs/PC_WALL_WAYPOINT_CROSS_SUFFIX_PROTOCOL_20261103_CODEX.md"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_actions(actions, shape):
    value = np.asarray(actions, dtype=np.float64)
    if (value.shape != shape or not np.isfinite(value).all()
            or np.any(value < LOW) or np.any(value > HIGH)):
        raise ValueError("invalid action array")
    return value


def source_arms(record):
    arms = record["arms"]
    if set(arms) != set(ROLES):
        raise ValueError("unexpected arm roster")
    prefixes, suffixes = {}, {}
    for role in ROLES:
        row = arms[role]
        prefix = check_actions(row["prefix"], (4, 2))
        saved = row["replay"]
        actions = check_actions(saved["actions"], (16, 2))
        if not np.array_equal(prefix, actions[:4]):
            raise ValueError("source prefix/action mismatch")
        verified = replay(actions)
        if (verified != saved
                or not np.isclose(saved["terminal_error"], row["scores"][row["selected_index"]],
                                  atol=1e-12, rtol=0)):
            raise ValueError("source replay mismatch")
        prefixes[role], suffixes[role] = prefix, actions[4:]
    return prefixes, suffixes


def summarize(result, zero_distance):
    positions = np.asarray(result["trajectory"], dtype=float)
    after = np.flatnonzero(positions[5:, 0] > 0)
    return {
        "replay": result,
        "zero_tail_distance": float(zero_distance),
        "distance_change_vs_zero_tail": float(result["terminal_error"] - zero_distance),
        "first_positive_x_suffix_step": int(after[0] + 1) if len(after) else None,
        "blocked_suffix_count": int(sum(result["blocked"][4:])),
    }


def run_record(record):
    prefixes, suffixes = source_arms(record)
    zero = {role: replay(np.concatenate((prefixes[role], np.zeros((12, 2)))))
            for role in ROLES}
    cells = []
    for suffix_role in ROLES:
        for prefix_role in ROLES:
            result = replay(np.concatenate((prefixes[prefix_role], suffixes[suffix_role])))
            if suffix_role == prefix_role and result != record["arms"][prefix_role]["replay"]:
                raise ValueError("diagonal replay mismatch")
            cells.append({"prefix_role": prefix_role, "suffix_role": suffix_role,
                          **summarize(result, zero[prefix_role]["terminal_error"])})
    return {"seed": record["seed"], "zero_tail": zero, "cells": cells}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    sources = [Path(__file__), SOURCE, PROTOCOL,
               Path(__file__).with_name("pc_wall_triprefix_cem_intervention_20261103_codex.py"),
               Path(__file__).with_name("pc_wall_unicycle_atlas_cem_20261103.py"),
               Path(__file__).with_name("pc_exact_wall_kill.py")]
    hashes = {str(p.relative_to(REPO)): sha(p) for p in sources}
    source = json.loads(SOURCE.read_text())
    if [row["seed"] for row in source["rows"]] != list(SEEDS):
        raise ValueError("seed roster mismatch")
    print("Replaying all 27 privileged waypoint suffix/prefix combinations.", flush=True)
    rows = [run_record(row) for row in source["rows"]]
    for row in rows:
        print(f"seed {row['seed']}: " + ", ".join(
            f"{cell['suffix_role']}->{cell['prefix_role']}={cell['replay']['terminal_error']:.4f}"
            for cell in row["cells"]), flush=True)
    if hashes != {str(p.relative_to(REPO)): sha(p) for p in sources}:
        raise ValueError("input changed during replay")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump({"schema": "pc_wall_waypoint_cross_suffix.v1",
                   "scope": "privileged post-outcome one exposed exact task",
                   "source_sha256": hashes, "numpy_version": np.__version__,
                   "transition_counts": {"source_verification": 9 * 16,
                                         "zero_tail": 9 * 16,
                                         "cross_suffix": 27 * 16},
                   "rows": rows}, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
