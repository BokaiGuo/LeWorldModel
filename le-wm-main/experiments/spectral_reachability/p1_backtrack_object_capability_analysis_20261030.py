"""Separate object position and circular angle responses of fixed physical probes.

No normalization combines pixels and radians. This is exploratory saved-data
analysis, not a selector, significance test, or P1 admission.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import p1_backtrack_capability_consume_20261030 as saved

require = saved.require


def wrap(angle):
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def measure(anchor, endpoints):
    require(len(endpoints) == 21, "fixed 21 probes")
    zero = endpoints[0]
    if zero is None:
        return dict(anchor=anchor["index"], root_slot=anchor["root_slot"], condition=anchor["condition"],
                    arm=anchor["arm"], role=anchor["role"],
                    complete_endpoints=sum(x is not None for x in endpoints),
                    status="zero_endpoint_missing", position_jacobian=None, angle_jacobian=None)
    base = np.asarray(zero, dtype=np.float64)
    require(base.shape == (7,) and np.isfinite(base).all(), "zero physical endpoint")
    pos_columns, angle_columns = [], []
    for i in range(10):
        plus, minus = endpoints[1 + 2 * i:3 + 2 * i]
        if plus is None or minus is None:
            pos_columns.append(None)
            angle_columns.append(None)
            continue
        pos, neg = (np.asarray(x, dtype=np.float64) for x in (plus, minus))
        require(pos.shape == neg.shape == (7,) and np.isfinite(pos).all() and np.isfinite(neg).all(),
                "physical endpoint")
        # Wrapped displacements around the common zero prevent a branch-cut jump.
        pos_columns.append(((pos[2:4] - neg[2:4]) / .5).tolist())
        angle_columns.append((wrap(pos[4] - base[4]) - wrap(neg[4] - base[4])) / .5)
    complete = all(x is not None for x in pos_columns + angle_columns)
    root = np.asarray(anchor["root_state"], dtype=np.float64)
    goal = np.asarray(anchor["goal"], dtype=np.float64)
    require(root.shape == goal.shape == (7,) and np.isfinite(root).all() and np.isfinite(goal).all(),
            "fixed task root and goal")
    direction = goal[2:4] - root[2:4]
    direction_norm = float(np.linalg.norm(direction))
    direction = direction / direction_norm if direction_norm else None
    angle_direction = wrap(goal[4] - root[4])
    angle_sign = None if abs(angle_direction) < 1e-12 or abs(abs(angle_direction) - np.pi) < 1e-12 else float(np.sign(angle_direction))
    position_goal_columns = [None if x is None or direction is None else float(direction @ x)
                             for x in pos_columns]
    angle_goal_columns = [None if x is None or angle_sign is None else angle_sign * x
                          for x in angle_columns]
    zero_position_error = float(np.linalg.norm(base[2:4] - goal[2:4]))
    zero_angle_error = abs(wrap(base[4] - goal[4]))
    position_gains, angle_gains = [], []
    for endpoint in endpoints:
        if endpoint is None:
            position_gains.append(None)
            angle_gains.append(None)
            continue
        state = np.asarray(endpoint, dtype=np.float64)
        require(state.shape == (7,) and np.isfinite(state).all(), "probe endpoint")
        position_gains.append(zero_position_error - float(np.linalg.norm(state[2:4] - goal[2:4])))
        angle_gains.append(zero_angle_error - abs(wrap(state[4] - goal[4])))
    if complete:
        matrix = np.asarray(pos_columns).T
        singular = np.linalg.svd(matrix, compute_uv=False).tolist()
        angle_norm = float(np.linalg.norm(angle_columns))
    else:
        singular = angle_norm = None
    return dict(anchor=anchor["index"], root_slot=anchor["root_slot"], condition=anchor["condition"],
                arm=anchor["arm"], role=anchor["role"],
                complete_endpoints=sum(x is not None for x in endpoints),
                status="complete" if complete else "incomplete",
                position_jacobian=pos_columns, position_singular_values=singular,
                angle_jacobian=angle_columns, angle_response_l2=angle_norm,
                position_goal_direction=None if direction is None else direction.tolist(),
                position_goal_response=position_goal_columns,
                angle_goal_sign=angle_sign, angle_goal_response=angle_goal_columns,
                zero_relative_position_error_gains=position_gains,
                zero_relative_angle_error_gains=angle_gains,
                fixed_bank_best_position_gain=max(position_gains) if complete else None,
                fixed_bank_best_angle_gain=max(angle_gains) if complete else None,
                denominator="0.5 logit units; tanh maps logits to bounded raw commands")


def paired(rows):
    require(len(rows) == 16, "fixed 16 paired anchors")
    result = []
    for i in range(0, 16, 2):
        a, b = rows[i:i + 2]
        require(a["anchor"] == i and b["anchor"] == i + 1 and
                all(a[k] == b[k] for k in ("root_slot", "condition", "arm")) and
                (a["role"], b["role"]) == ("incumbent", "model_selected"), "paired anchor identity")
        record = dict(root_slot=a["root_slot"], condition=a["condition"], arm=a["arm"],
                      incumbent_anchor=i, selected_anchor=i + 1)
        for name in ("fixed_bank_best_position_gain", "fixed_bank_best_angle_gain"):
            record["selected_minus_incumbent_" + name] = (
                None if a.get(name) is None or b.get(name) is None else b[name] - a[name])
        for name in ("zero_relative_position_error_gains", "zero_relative_angle_error_gains"):
            x, y = a.get(name), b.get(name)
            record["selected_minus_incumbent_" + name] = (
                None if x is None or y is None else
                [None if u is None or v is None else v - u for u, v in zip(x, y, strict=True)])
        result.append(record)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--prepare-sha", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.prepared)
    require(saved.runner.physical.sha(source) == args.prepare_sha, "prepared SHA")
    prepared = saved.runner.physical.native.read(source, cap=32 * 1024**2)
    require(prepared == saved.preparation.build(), "source and input closure")
    directory = Path(args.directory)
    binding_file = directory / "run_binding.json"
    binding_sha = saved.runner.physical.sha(binding_file)
    binding = saved.runner.physical.native.read(binding_file, cap=32 * 1024**2)
    require(binding["prepare_path"] == str(source.resolve()) and
            binding["prepare_sha256"] == args.prepare_sha and binding["fixed_rows"] == 336 and
            all(binding["source_code_bindings"].get(k) == v for k, v in prepared["bindings"].items()),
            "run binding identity")
    analysis_files = {str(Path(file).resolve()): saved.runner.physical.sha(file)
                      for file in (__file__, saved.__file__)}
    output = Path(args.output)
    require(output.is_absolute() and output.parent == saved.runner.physical.ROOT / "results" and
            output.name.startswith("p1_backtrack_object_capability_analysis_20261030_") and
            not output.exists() and not output.is_symlink(), "exclusive output")
    rows, hashes = [], []
    for anchor in prepared["anchors"]:
        shard = directory / f"anchor_{anchor['index']:02d}"
        require(shard.is_dir(), "all fixed shards required")
        report = shard / "report.json"
        digest = saved.runner.physical.sha(report)
        endpoints = saved.check_shard(shard, anchor, prepared["probes"])
        require(saved.runner.physical.sha(report) == digest, "shard report drift")
        rows.append(measure(anchor, endpoints))
        hashes.append(dict(anchor=anchor["index"], report_sha256=digest))
    for path, digest in {**binding["source_code_bindings"], **analysis_files}.items():
        require(saved.runner.physical.sha(path) == digest, "source or analysis code drift")
    require(saved.runner.physical.sha(binding_file) == binding_sha, "run binding drift")
    result = dict(schema="p1_backtrack_object_capability_analysis_20261030", rows=rows, pairs=paired(rows),
                  shard_hashes=hashes,
                  prepared=str(source.resolve()), prepare_sha256=args.prepare_sha, fixed_branches=336,
                  run_binding_sha256=binding_sha, analysis_code_bindings=analysis_files,
                  known_roots=2, independent_confirmation=False, P1_admission=False,
                  limitation="Finite secants only; no rank or causal capability conclusion.")
    data = saved.runner.physical.native.raw(result)
    with output.open("xb") as stream:
        require(stream.write(data) == len(data), "complete result write")
        stream.flush()
        import os
        os.fsync(stream.fileno())
    saved.runner.physical.native.paired.fsync_dir(output.parent)
    print(json.dumps(dict(path=str(output), sha256=saved.runner.physical.sha(output))))


if __name__ == "__main__":
    main()
