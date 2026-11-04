"""Common-suffix position intervention on previously selected development prefixes."""
import json
from pathlib import Path

import numpy as np
import torch

from experiments.pc_pointmaze_affine_feedback_cem_20261103 import (
    ORIGINAL, ORIGINAL_SHA, OUTPUT as RANDOM,
)
from experiments.pc_pointmaze_affine_spectral_cem_20261103 import (
    LENGTHS_SHA, OUTPUT as SPECTRAL, REFERENCE_SHA,
)
from experiments.spectral_reachability import pointmaze_physical_evaluator as physical
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import ROOT, DEFAULT_DATASET, sha256_file

SPECTRAL_SHA = "2606f1b1e6cbd70a263ed8593881df8355f2de20f15483ed0caa8db4f3e50694"
SCORE_AUDIT = ROOT / "results/pc_pointmaze_affine_score_replay_20261103.json"
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_AFFINE_COMMON_SUFFIX_PROTOCOL_20261103.md"
OUTPUT = ROOT / "results/pc_pointmaze_affine_common_suffix_20261103.json"
ARRAYS = OUTPUT.with_suffix(".npz")
ROOTS = tuple(range(1888, 1896))
PREFIXES = ("zero", "random_cem", "spectral_cem")
SUFFIXES = ("zero", "xp", "xm", "yp", "ym", "random_tail", "spectral_tail")


def validate_action(action):
    action = np.asarray(action, np.float64)
    if action.shape != (2,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
        raise ValueError("expected finite bounded two-dimensional action")
    return action


def main():
    if OUTPUT.exists() or ARRAYS.exists():
        raise FileExistsError("refusing overwrite")
    for path, expected in ((RANDOM, REFERENCE_SHA), (SPECTRAL, SPECTRAL_SHA),
                           (ORIGINAL, ORIGINAL_SHA), (DEFAULT_DATASET / "seq_lengths.pth", LENGTHS_SHA)):
        if sha256_file(path) != expected:
            raise ValueError(f"changed input: {path}")
    random, spectral, original, audit = (json.loads(path.read_text()) for path in
                                        (RANDOM, SPECTRAL, ORIGINAL, SCORE_AUDIT))
    if not all(audit[key] for key in ("all_scores_close", "all_orders_same", "all_elites_same")):
        raise ValueError("previous fixed candidate score audit failed")
    if audit["input_sha256"][str(RANDOM.relative_to(ROOT))] != REFERENCE_SHA or audit["input_sha256"][str(SPECTRAL.relative_to(ROOT))] != SPECTRAL_SHA:
        raise ValueError("score audit source mismatch")
    for path, digest in ((RANDOM.with_suffix(".npz"), random["arrays_sha256"]),
                         (SPECTRAL.with_suffix(".npz"), spectral["arrays_sha256"]),
                         (DEFAULT_DATASET / "states.pth", original["states_sha256"]),
                         (DEFAULT_DATASET / "actions.pth", original["actions_sha256"])):
        if sha256_file(path) != digest:
            raise ValueError(f"changed input: {path}")
    if any([r["root"] for r in report["rows"]] != list(ROOTS) for report in (random, spectral)):
        raise ValueError("root roster mismatch")
    # Match PointMazeDataset's float32 conversion before extracting states.
    states = torch.load(DEFAULT_DATASET / "states.pth", map_location="cpu", weights_only=True).float().numpy()
    source_rows = {kind: {r["root"]: r for r in report["rows"]}
                   for kind, report in (("random_cem", random), ("spectral_cem", spectral))}
    starts, sequences, roster = [], [], []
    for root in ROOTS:
        initial = np.asarray(states[root, 0], np.float64)
        if initial.shape != (4,) or not np.isfinite(initial).all():
            raise ValueError("invalid root state")
        prefix = {"zero": np.zeros(2), **{
            kind: validate_action(source_rows[kind][root]["initial_plan"][0])
            for kind in ("random_cem", "spectral_cem")}}
        suffix = {"zero": np.zeros(2), "xp": np.array([.5, 0]), "xm": np.array([-.5, 0]),
                  "yp": np.array([0, .5]), "ym": np.array([0, -.5]),
                  "random_tail": validate_action(source_rows["random_cem"][root]["tail"][0]),
                  "spectral_tail": validate_action(source_rows["spectral_cem"][root]["tail"][0])}
        for p in PREFIXES:
            for s in SUFFIXES:
                starts.append(initial)
                sequences.append(np.concatenate((np.repeat(prefix[p][None], 5, axis=0),
                                                  np.repeat(suffix[s][None], 10, axis=0))))
                roster.append((root, p, s))
    starts, sequences = np.stack(starts), np.stack(sequences)
    receipt = physical.position_backend_receipt()
    print(f"Execute {len(roster)} paired common-suffix trajectories at cp5, cp10 and cp15.", flush=True)
    cp5 = physical.replay_position_endpoints(starts, sequences[:, :5])
    cp10 = physical.replay_position_endpoints(starts, sequences[:, :10])
    cp15 = physical.replay_position_endpoints(starts, sequences)
    # The actual previous action chain is a mandatory cross-protocol anchor.
    for index, (root, prefix, suffix) in enumerate(roster):
        if (prefix, suffix) in (("random_cem", "random_tail"), ("spectral_cem", "spectral_tail")):
            expected = np.asarray(source_rows[prefix][root]["endpoint_eval_only"])
            if not np.array_equal(cp10[index, :2], expected[:2]):
                raise RuntimeError(f"historical physical endpoint mismatch: {root} {prefix}")
    rows = []
    index_by_key = {key: i for i, key in enumerate(roster)}
    for index, (root, prefix, suffix) in enumerate(roster):
        target = np.asarray(states[root, 10, :2], np.float64)
        start_error = float(np.linalg.norm(starts[index, :2] - target))
        prefix_error = float(np.linalg.norm(cp5[index, :2] - target))
        zero_prefix_error = float(np.linalg.norm(cp5[index_by_key[(root, "zero", suffix)], :2] - target))
        passive_index = index_by_key[(root, prefix, "zero")]
        horizons = {}
        for name, endpoints in (("5", cp10), ("10", cp15)):
            state = endpoints[index]
            passive = endpoints[passive_index]
            error = float(np.linalg.norm(state[:2] - target))
            passive_error = float(np.linalg.norm(passive[:2] - target))
            horizons[name] = {"state": state.tolist(), "goal_error": error,
                              "delta_from_own_passive_goal_error": error - passive_error,
                              "position_response_from_own_passive": (state[:2] - passive[:2]).tolist()}
        rows.append({"root": root, "prefix": prefix, "suffix": suffix, "start_state": starts[index].tolist(),
                     "goal_eval_only": target.tolist(), "prefix_action": sequences[index, 0].tolist(),
                     "suffix_action": sequences[index, 5].tolist(), "prefix_state": cp5[index].tolist(),
                     "start_goal_error": start_error, "prefix_goal_error": prefix_error,
                     "prefix_delta_from_start": prefix_error - start_error,
                     "prefix_delta_from_zero_prefix": prefix_error - zero_prefix_error,
                     "non_immediate_position_progress": bool(prefix_error - start_error >= -1e-8),
                     "horizons": horizons})
    with ARRAYS.open("xb") as stream:
        np.savez_compressed(stream, starts=starts, actions=sequences, cp5=cp5, cp10=cp10, cp15=cp15)
    report = {"schema": "pc_pointmaze_affine_common_suffix.v1", "protocol_sha256": sha256_file(PROTOCOL),
              "script_sha256": sha256_file(Path(__file__)), "arrays_sha256": sha256_file(ARRAYS),
              "random_sha256": REFERENCE_SHA, "spectral_sha256": SPECTRAL_SHA,
              "score_audit_sha256": sha256_file(SCORE_AUDIT), "states_sha256": original["states_sha256"],
              "seq_lengths_sha256": LENGTHS_SHA, "physical_backend": receipt,
              "root_count": len(ROOTS), "trajectory_count": len(rows), "rows": rows,
              "scope": "exposed roots and common suffixes; not individually replanned or confirmatory"}
    with OUTPUT.open("x") as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Saved {OUTPUT} sha256={sha256_file(OUTPUT)}", flush=True)


if __name__ == "__main__":
    main()
