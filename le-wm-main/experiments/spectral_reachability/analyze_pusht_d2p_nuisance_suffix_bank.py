"""Fail-closed saved-array audit; no torch, inference, selector or P1 admission."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

MODES = ("identity_gn4", "nuisance_gn4")
GATES = {"cuda_only", "all64_pairs", "all128_branches", "predict_accounting",
         "parameters_unchanged", "buffers_unchanged", "all_eval", "source_unchanged",
         "inputs_unchanged", "weights_unchanged", "config_unchanged"}
SOURCES = {"runner", "solver", "metric", "GN", "dynamics", "context", "physical",
           "base", "loader", "collector"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def array(value, shape):
    result = np.asarray(value, dtype=float)
    require(result.shape == shape and np.isfinite(result).all(), "finite array shape contract")
    return result


def close(actual, expected):
    # CUDA float32 receipt roundoff, not a scientific decision threshold.
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-5)


def joined(rows):
    keys = [(r["root_id"], r["proposal_index"]) for r in rows]
    require(len(keys) == 64 and len(set(keys)) == 64 and
            set(keys) == {(r, p) for r in range(4) for p in range(16)}, "full unique 4x16 join")
    return dict(zip(keys, rows, strict=True))


def projection(goals):
    goals = array(goals, (5, 192))
    directions = (goals[1:] - goals[0]).T
    # Independent least-squares orthogonal complement, not runner compile_metric.
    coefficients, _, rank, singular = np.linalg.lstsq(directions, np.eye(192), rcond=None)
    return np.eye(192) - directions @ coefficients, rank, singular


def numerical(branch, goal, q):
    action = array(branch["suffix_action"], (1, 2))
    require(np.all(np.abs(action) <= 1), "bounded tanh action required")
    forecast = branch["forecast"]
    require(forecast["metric_same_for_spectrum_and_GN"] is True and
            forecast["physical_labels_used"] is False and
            forecast["linear_legacy_raw_cost_fields_not_used_for_selection"] is True,
            "same-Q causal receipt")
    predicted = array(forecast["predicted_terminal_vector"], (192,))
    actual = array(branch["final_vector"], (192,))
    observed = array(branch["observed_vector"], (192,))
    cost = lambda x, metric: float(np.mean((metric @ (x-goal))**2))
    model_cost = cost(predicted, q)
    close(model_cost, forecast["predicted_metric_cost"])
    close(cost(predicted, np.eye(192)), forecast["raw_predicted_goal_cost"])
    gn = forecast["nonlinear"]
    require(gn["damping"] == .01 and gn["metric"] == "full_latent_SSE_plus_damping_parameter_energy",
            "fixed GN objective")
    close(array(gn["predicted_terminal_vector"], (192,)), q @ predicted)
    close(gn["predicted_goal_cost"], model_cost)
    # Float32 tanh can round to +/-1; original parameter energy is then lost.
    if np.all(np.abs(action) < 1):
        close(gn["final_objective"], 192*model_cost + .01*np.sum(np.arctanh(action)**2))
    else:
        require(gn["final_objective"] >= 192*model_cost-3e-5*max(1, 192*model_cost),
                "nonnegative parameter energy")
    current = min(gn["initial_objective"], gn["zero_objective"])
    close(current, gn["retained_incumbent_objective"])
    require(len(gn["iterations"]) == 4, "exact four GN steps")
    for i, step in enumerate(gn["iterations"]):
        require(step["iteration"] == i, "ordered GN iterations")
        candidates = array(step["line_search_costs"], (4,))
        close(step["before"], current)
        current = min(current, *candidates)
        close(step["after"], current)
    close(current, gn["final_objective"])
    linear = forecast["linear"]
    singular = array(linear["singular_values"], (2,))
    require(np.all(singular >= 0) and singular[0] >= singular[1] and
            np.isfinite(linear["value"]) and linear["value"] >= 0, "finite spectral receipt")
    close(np.linalg.norm(singular), linear["jacobian_frobenius_norm"])
    start = array(branch["successor_state"], (7,))
    end = array(branch["final_state"], (7,))
    for label in ("prefix_pose", "final_pose"):
        pose = branch[label]
        values = array([pose[k] for k in ("xy_distance", "angle_distance", "pose_cost")], (3,))
        require(np.all(values >= 0) and values[1] <= np.pi, "physical label range")
        close(values[2], values[0]+20*values[1])
    return {"predicted_own_metric_cost": model_cost,
            "actual_own_metric_cost": cost(actual, q),
            "predicted_identity_cost": cost(predicted, np.eye(192)),
            "actual_identity_cost": cost(actual, np.eye(192)),
            "actual_common_nuisance_cost": None,
            "saturated_action_coordinates": int(np.sum(np.abs(action) == 1)),
            "actual_own_metric_progress": cost(observed, q)-cost(actual, q),
            "pose_cost": branch["final_pose"]["pose_cost"],
            "pose_progress": branch["prefix_pose"]["pose_cost"]-branch["final_pose"]["pose_cost"],
            "xy_progress": branch["prefix_pose"]["xy_distance"]-branch["final_pose"]["xy_distance"],
            "angle_progress": branch["prefix_pose"]["angle_distance"]-branch["final_pose"]["angle_distance"],
            "object_xy_movement": float(np.linalg.norm(end[2:4]-start[2:4])),
            "object_angle_movement": float(abs(np.arctan2(np.sin(end[4]-start[4]), np.cos(end[4]-start[4]))))}


def analyze(path):
    path = Path(path)
    report, protocol = read(path), read(path.parent/"protocol.json")
    require(report["schema"] == "pusht_d2p_nuisance_suffix_bank.v1" and
            report["status"] == "completed" and report["protocol"] == protocol,
            "completed immutable protocol required")
    require(set(report["gates"]) == GATES and all(v is True for v in report["gates"].values()), "all 11 gates")
    require(report["P1_admission"] is False and protocol["P1_admission"] is False, "not P1")
    require(protocol["schema"] == "pusht_d2p_nuisance_suffix_bank_protocol.v1" and
            protocol["modes"] == list(MODES) and protocol["proposal_indices"] == list(range(16)) and
            len(protocol["root_ids"]) == len(set(protocol["root_ids"])) == 4 and
            protocol["iterations"] == 4 and protocol["damping"] == .01, "frozen experiment design")
    for flag in ("all_prefixes_fixed_unchanged", "goal_set_only_metric_construction", "no_outcome_selection"):
        require(protocol[flag] is True, flag)
    require(report["actual_predict_samples"] == protocol["expected_predict_samples"] == 3968 and
            report["physical_future_branches"] == protocol["expected_real_branches"] == 128, "3968/128 accounting")
    require(set(protocol["implementation_sha256"]) == SOURCES, "all source archives")
    for name, sha in protocol["implementation_sha256"].items():
        require(digest(path.parent/"implementation"/(name+".py")) == sha, "source archive hash")
    require(len(protocol["bindings"]) == 4, "four upstream bindings")
    bound = {}
    for filename, sha in protocol["bindings"].items():
        require(digest(filename) == sha, "upstream binding hash")
        bound[filename] = read(filename)
    bank = bound[protocol["bank_path"]]
    interventions = [r for r in bound.values() if r["schema"] == "pusht_d2p_goal_agent_intervention.v1"]
    require(len(interventions) == 1, "unique goal-only intervention")
    intervention = interventions[0]
    require(intervention["protocol"]["bank_path"] == protocol["bank_path"] and
            intervention["protocol"]["root_ids"] == protocol["root_ids"] == bank["protocol"]["roots"], "root lineage")
    originals, rows = joined(bank["rows"]), joined(report["rows"])
    goals_by_root = {r["root_id"]: r for r in intervention["rows"]}
    require(len(intervention["rows"]) == len(goals_by_root) == 4, "four unique goal roots")
    output = []
    for root in range(4):
        plan_path = path.parent/"stages"/f"root_{root}.plan.json"
        plan = read(plan_path)
        goalrow = goals_by_root[root]
        require(plan["root_id"] == root and plan["root_sha256"] == goalrow["root_sha256"] ==
                protocol["root_ids"][root] and plan["physical_future_labels_used"] is False, "root plan causal identity")
        goals = array([v["goal_vector"] for v in goalrow["variants"]], (5, 192))
        q, rank, singular = projection(goals)
        np.testing.assert_allclose(plan["projection"], q, atol=1e-12, rtol=1e-12)
        require(plan["metric_receipt"]["rank"] == rank and
                plan["metric_receipt"]["physical_outcomes_used"] is False and
                plan["metric_receipt"]["additional_training"] is False, "goal-only metric receipt")
        close(plan["metric_receipt"]["singular_values"], singular)
        np.testing.assert_array_equal(plan["goal_vector"], goals[0])
        for proposal in range(16):
            row, old = rows[root, proposal], originals[root, proposal]
            require(old["root_sha256"] == protocol["root_ids"][root], "historical root join")
            np.testing.assert_array_equal(plan["prefixes"][proposal], old["prefix_action"])
            require(set(row["branches"]) == set(MODES), "exact two branches")
            metrics = {}
            baseline = old["branches"]["full192_gn4"]
            for mode in MODES:
                branch = row["branches"][mode]
                stage_path = path.parent/"stages"/f"root_{root}.candidate_{proposal}.{mode}.json"
                require(Path(branch["stage_path"]).resolve() == stage_path.resolve() and
                        digest(stage_path) == branch["stage_sha256"], "stage path/hash")
                stage = read(stage_path)
                require(stage["root_id"] == root and stage["proposal_index"] == proposal and
                        stage["mode"] == mode and stage["root_plan_sha256"] == digest(plan_path) and
                        stage["future_suffix_observations_used"] is False, "pre-execution stage identity")
                for field in ("suffix_action", "forecast", "observed_vector"):
                    require(stage[field] == branch[field], "stage/report receipt mismatch")
                for field in ("successor_state", "prefix_pose"):
                    require(branch[field] == baseline[field], "fixed prefix replay")
                require(branch["observed_vector"] == baseline["observed_prefix_vector"], "observed context replay")
                if mode == MODES[0]:
                    for field in ("suffix_action", "final_rgb_sha256", "final_state", "final_vector", "final_pose"):
                        require(branch[field] == baseline[field], "exact historical baseline replay")
                metrics[mode] = numerical(branch, goals[0], np.eye(192) if mode == MODES[0] else q)
                final = array(branch["final_vector"], (192,))
                metrics[mode]["actual_common_nuisance_cost"] = float(np.mean((q@(final-goals[0]))**2))
            output.append({"root_id": root, "proposal_index": proposal, "branches": metrics,
                           "nuisance_minus_identity": {k: metrics[MODES[1]][k]-metrics[MODES[0]][k]
                                                       for k in metrics[MODES[0]]}})
    summary = {}
    for key in output[0]["nuisance_minus_identity"]:
        values = np.asarray([r["nuisance_minus_identity"][key] for r in output])
        summary[key] = {"mean": float(values.mean()), "median": float(np.median(values)),
                        "negative": int(sum(values < 0)), "zero": int(sum(values == 0)),
                        "positive": int(sum(values > 0)),
                        "root_means": [float(values[i*16:(i+1)*16].mean()) for i in range(4)]}
    return {"schema": "pusht_d2p_nuisance_suffix_bank_analysis.v1", "report_sha256": digest(path),
            "protocol_sha256": digest(path.parent/"protocol.json"), "verified_pairs": 64,
            "verified_branches": 128, "predict_samples_receipt": 3968,
            "new_inference": 0, "P1_admission": False, "summary": summary, "rows": output,
            "scope": "All fixed prefixes, no outcome-selected selector. Paired diagnostic on four known calibration roots; 64 pairs are not 64 independent roots. Both arms GN4/same predict budget, but nuisance arm uses extra goal-set input; own-metric costs are not common-objective efficacy comparisons.",
            "verification_limits": ["No saved Jacobian, SVD vectors, zero-terminal vector, GN intermediate vectors or parameters: spectral value/DLS/GN directions cannot be independently recomputed without inference; singular norm and saved objective/line-search contracts only.",
                                    "Float32 tanh saturation loses original parameters: final regularized objective is reconstructed only for unsaturated actions; saturated actions permit only nonnegative-energy checks.",
                                    "No saved physical goal state: absolute xy/angle distances are receipts, not independently reconstructed; pose algebra, changes and object movement are recomputed.",
                                    "Frozen parameters/buffers, CUDA and compute counts are runner receipts plus hashed archives, not an independent runtime trace."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "immutable new output required")
    result = analyze(args.report)
    result["analyzer_sha256"] = digest(Path(__file__))
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result["summary"], allow_nan=False))


if __name__ == "__main__":
    main()
