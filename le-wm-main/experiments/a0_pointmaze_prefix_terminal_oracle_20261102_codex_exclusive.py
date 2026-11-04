"""Offline PointMaze fixed-bank prefix/terminal oracle using physical replay.

This uses privileged saved states. It is neither an RGB planner nor closed-loop
control. The two-step ceiling is exhaustive only over the supplied 32x32 bank.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import pointmaze_physical_evaluator as physical

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/pointmaze_rgb_closed_loop_strict_20261021_cuda_robust_goal_fresh32/live_report.json"
SOURCE_SHA = "04fbd14ed4eabc3a98f19281afa42d75325b1502e5380b445ed7fbcb22be29e7"
ROOTS = tuple(range(1900, 1908))
NAME = "a0_pointmaze_prefix_terminal_oracle_20261102_codex_exclusive"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract_inputs(report):
    """Recover the shared first-decision state, waypoint, and RGB-derived bank."""
    methods = {"visual_dynamics", "effect_adapter_rgb", "effect_adapter_rgb_htsc",
               "effect_adapter_rgb_htsc_consistency", "ground_truth_candidate_oracle",
               "perfect_dynamics_mpc"}
    require(set(report["protocol"]["methods"]) == methods, "source method roster")
    result = []
    for root in ROOTS:
        rows = [r for r in report["rows"] if r["episode_id"] == root]
        require(len(rows) == len(methods) and {r["method"] for r in rows} == methods,
                "complete distinct root arms")
        reference = next(r for r in rows if r["method"] == "visual_dynamics")
        trace = reference["trace"][0]
        bank = np.asarray(reference["policy_contract"]["candidate_actions"], dtype=np.float64)
        state = np.asarray(trace["runtime_state_for_evaluation"], dtype=np.float64)
        target = np.asarray(trace["active_waypoint"], dtype=np.float64)
        require(bank.shape == (32, 2) and state.shape == (4,) and target.shape == (2,)
                and np.isfinite(bank).all() and np.isfinite(state).all()
                and np.isfinite(target).all() and np.max(np.abs(bank)) <= 1, "finite legal inputs")
        bank_sha = hashlib.sha256(np.ascontiguousarray(bank).tobytes()).hexdigest()
        state_matches = {}
        for row in rows:
            t, contract = row["trace"][0], row["policy_contract"]
            require(t["block"] == 0 and t["candidate_bank_digest"] == bank_sha
                    and contract["candidate_bank_digest"] == bank_sha
                    and np.array_equal(np.asarray(contract["candidate_actions"]), bank),
                    "first-decision cross-arm bank identity")
            state_matches[row["method"]] = bool(
                np.array_equal(np.asarray(t["runtime_state_for_evaluation"]), state)
                and np.array_equal(np.asarray(t["active_waypoint"]), target))
        result.append({"root": root, "source_state": state, "target": target,
                       "bank": bank, "bank_sha256": bank_sha, "state_matches": state_matches})
    return result


def evaluate(inputs, replay):
    """Issue the 1- and 2-action batches and retain every physical position."""
    require(inputs and callable(replay), "inputs and replay")
    n = len(inputs[0]["bank"])
    require(n > 0 and all(len(r["bank"]) == n for r in inputs), "equal bank sizes")
    one_sources = np.concatenate([np.repeat(r["source_state"][None, :], n, axis=0) for r in inputs])
    one_actions = np.concatenate([r["bank"][:, None, :] for r in inputs])
    two_sources = np.concatenate([np.repeat(r["source_state"][None, :], n*n, axis=0) for r in inputs])
    two_actions = np.concatenate([
        np.stack([np.repeat(r["bank"], n, axis=0), np.tile(r["bank"], (n, 1))], axis=1)
        for r in inputs
    ])
    prefixes = np.asarray(replay(one_sources, one_actions), dtype=np.float64)
    terminals = np.asarray(replay(two_sources, two_actions), dtype=np.float64)
    require(prefixes.shape == one_sources.shape and terminals.shape == two_sources.shape
            and np.isfinite(prefixes).all() and np.isfinite(terminals).all(), "physical endpoint shape")
    rows = []
    for index, item in enumerate(inputs):
        p = prefixes[index*n:(index+1)*n, :2]
        t = terminals[index*n*n:(index+1)*n*n, :2].reshape(n, n, 2)
        pc = np.linalg.norm(p - item["target"], axis=1)
        tc = np.linalg.norm(t - item["target"], axis=2)
        prefix_index = int(np.argmin(pc))
        first, second = np.unravel_index(int(np.argmin(tc)), tc.shape)
        prefix_tail = int(np.argmin(tc[prefix_index]))
        rows.append({
            "root": item["root"], "source_state_privileged": item["source_state"].tolist(),
            "active_waypoint": item["target"].tolist(), "candidate_bank": item["bank"].tolist(),
            "candidate_bank_sha256": item["bank_sha256"],
            "first_decision_state_and_waypoint_match_visual_by_arm": item["state_matches"],
            "prefix_positions": p.tolist(), "terminal_positions": t.tolist(),
            "prefix_selected_index": prefix_index,
            "terminal_selected_indices": [int(first), int(second)],
            "prefix_selected_best_terminal_tail_index": prefix_tail,
            "prefix_cost_at_prefix_choice": float(pc[prefix_index]),
            "prefix_cost_at_terminal_choice": float(pc[first]),
            "terminal_cost_at_terminal_choice": float(tc[first, second]),
            "terminal_cost_with_prefix_first_best_tail": float(tc[prefix_index, prefix_tail]),
            "first_action_differs": bool(first != prefix_index),
        })
    return rows


def build():
    require(digest(SOURCE) == SOURCE_SHA, "sealed source report digest")
    inputs = extract_inputs(json.loads(SOURCE.read_bytes()))
    receipt = physical.position_backend_receipt()
    rows = evaluate(inputs, physical.replay_position_endpoints)
    return {
        "schema": NAME + ".v1", "source_report": str(SOURCE), "source_sha256": SOURCE_SHA,
        "backend_position_receipt": receipt,
        "protocol": {
            "roots": list(ROOTS), "source": "previously opened strict-panel first decision",
            "source_arm": "visual_dynamics; cross-arm first-decision equality not assumed",
            "action_bank": "saved root-fixed RGB-derived 32 actions, unchanged at both depths",
            "input_boundary": "privileged saved simulator state used only for offline oracle evaluation",
            "target": "saved first active waypoint, not global terminal task goal",
            "prefix": "distance after one archive action",
            "terminal": "distance after two archive actions",
            "rollout": "each archive action clips velocity then executes 5 MuJoCo ticks",
            "selection": "exact argmin, lexicographic tie-break, 32 + 1024 sequences per root",
            "scope": "single decision; no action executed in an episode and no re-observation",
            "oracle": True, "rgb_only_planner": False, "training": False,
            "fresh_root_evidence": False, "equal_compute_claim": False,
            "full_state_equivalence": "unknown; inherited contract is position-only",
            "contract_horizon_note": "backend receipt tests recorded horizons 5 and 10 archive actions; this audit uses 1 and 2, not separately certified",
        },
        "rows": rows,
        "summary": {"roots": len(rows), "different_first_action": sum(r["first_action_differs"] for r in rows),
                    "success_rate": None, "closed_loop_gain": None},
        "P1_gate_changed": False, "scientific_admission": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(args.output.is_absolute() and args.output.parent == ROOT / "results"
            and args.output.name.startswith(NAME + "_") and args.output.suffix == ".json",
            "exclusive result path")
    print("Evaluating fixed-bank 1/2-action physical oracle; no RGB planner or GPU.")
    result = build()
    raw = (json.dumps(result, sort_keys=True, allow_nan=False) + "\n").encode()
    with args.output.open("xb") as stream:
        stream.write(raw)
        stream.flush()
    print(json.dumps({"output": str(args.output), "sha256": hashlib.sha256(raw).hexdigest(),
                      "summary": result["summary"]}))


if __name__ == "__main__":
    main()
