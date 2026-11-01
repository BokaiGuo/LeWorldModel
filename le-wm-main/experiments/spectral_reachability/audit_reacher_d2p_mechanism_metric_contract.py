"""Own-state vs common-residual and same-space vs mixed-space mechanism audit.

Saved host-CUDA observations only. This diagnoses evaluator/proposal semantics;
it neither changes the historic P1 gates nor selects programs using outcomes.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import analyze_reacher_d2p_prefix_capability as base
from experiments.spectral_reachability import reacher_d2p_capability_reference as source
from experiments.spectral_reachability.analyze_reacher_d2p_goal_metric_confirmation import verify_goal_receipt
from experiments.spectral_reachability.audit_reacher_d2p_capability_rejections import indexed

SPACES = ("raw_rgb", "goal_Q_rgb", "joint")
STAGE_MARGIN = 1e-6
VALUE_MARGIN = 1e-10


def compare_role(space, roles, role):
    """Compare both residual contracts without treating them as interchangeable."""
    index, zero, greedy = (str(roles[k]) for k in (role, "zero", "greedy"))
    cell, z, g = (space[k] for k in (index, zero, greedy))
    values = base.finite([cell["stage"], z["stage"], cell["terminal"], g["terminal"]], (4,))
    if np.any(values < 0):
        raise ValueError("nonnegative same-space physical costs required")
    if any(len(x["fits"]) != 2 for x in (cell, z, g)):
        raise ValueError("both fixed probe scales required")
    comparisons = []
    for k, scale in enumerate(base.SCALES):
        if any(x["fits"][k]["scale"] != scale for x in (cell, z, g)):
            raise ValueError("fixed ordered scales required")
        own = [cell["fits"][k]["own_objective_mean"]-c["fits"][k]["own_objective_mean"] for c in (z, g)]
        gain = [cell["fits"][k]["common_gain_fraction"]-c["fits"][k]["common_gain_fraction"] for c in (z, g)]
        total = [cell["stage"]-c["stage"]+v for c, v in zip((z, g), own)]
        base.finite(own+gain+total, (6,))
        comparisons.append({"scale":scale,"own_value_minus_zero_greedy":own,
            "common_gain_minus_zero_greedy":gain,"stage_plus_own_value_minus_zero_greedy":total})
    checks = {"same_space_stage_regression_vs_zero":cell["stage"] > z["stage"]+STAGE_MARGIN,
        "same_space_terminal_better_than_greedy":cell["terminal"] < g["terminal"]-STAGE_MARGIN,
        "own_state_value_better_at_both_scales":all(v < -VALUE_MARGIN for c in comparisons for v in c["own_value_minus_zero_greedy"]),
        "common_gain_better_at_both_scales":all(v > VALUE_MARGIN for c in comparisons for v in c["common_gain_minus_zero_greedy"]),
        "stage_plus_own_value_better_at_both_scales":all(v < -VALUE_MARGIN for c in comparisons for v in c["stage_plus_own_value_minus_zero_greedy"])}
    own_conjunction = all(checks[k] for k in ("same_space_stage_regression_vs_zero",
        "same_space_terminal_better_than_greedy", "own_state_value_better_at_both_scales"))
    return {"checks":checks,"comparisons":comparisons,
        "own_state_same_space_conjunction":own_conjunction,
        "stage_plus_own_reach_conjunction":own_conjunction and checks["stage_plus_own_value_better_at_both_scales"],
        "initial_state_stage_regression":(cell["stage"] > cell["initial_stage"]+STAGE_MARGIN if cell["initial_stage"] is not None else None),
        "zero_prefix_stage_minus_initial":(z["stage"]-z["initial_stage"] if z["initial_stage"] is not None else None)}


def audit(path):
    historical = base.verify(path)
    report = json.loads(path.read_text())
    protocol = report["sealed_protocol"]
    if protocol.get("source_contract") != source.FULL:
        raise ValueError("explicit complete full-Q compiler capability source required")
    references = [Path(p) for p in protocol["input_bindings"] if not p.endswith(".predictions.json")]
    if len(references) != 1:
        raise ValueError("one original full compiler report required")
    reference = references[0]
    pred_path = reference.with_suffix(".predictions.json")
    plans = indexed(json.loads(pred_path.read_text())["rows"])
    measured, old_checks = indexed(report["rows"]), indexed(historical["rows"])
    if len(plans) != 8 or set(plans) != set(measured) or set(plans) != set(old_checks):
        raise ValueError("complete eight-root identity join required")
    original_physical = indexed(json.loads(reference.read_text())["rows"])
    rows, fits = [], 0
    for seed in sorted(plans):
        plan, observation = plans[seed], measured[seed]
        if any(plan[k] != observation[k] for k in ("source_episode", "source_start_row", "goal_vector", "goal_admission")):
            raise ValueError("complete source/task identity required")
        target = base.finite(plan["goal_vector"], (192,))
        admission = plan["goal_admission"]
        target_qpos = base.finite(admission["feasible_target_qpos"], (2,))
        limited = admission["limited_joints"]
        transform = verify_goal_receipt(plan["goal_metric"])
        initial_joint_cost = float(np.mean(base.delta(target_qpos, plan["initial_snapshot"]["qpos"], limited)**2))
        tasks = {}
        for task in base.TASKS:
            measured_task = observation["tasks"][task]
            roles, cells = measured_task["roles"], measured_task["cells"]
            model_plan = source.task_plan(plan, task, source.FULL)
            if roles != source.full_roles(model_plan):
                raise ValueError("all original model-sealed roles required")
            zero = cells["0"]["center"]
            common_residual = {"raw_rgb":target-base.finite(zero["observed"], (192,)),
                "joint":base.delta(target_qpos, zero["qpos"], limited)}
            common_residual["goal_Q_rgb"] = transform @ common_residual["raw_rgb"]
            spaces = {name:{} for name in SPACES}
            for i, cell in cells.items():
                branch = source.physical_branch(original_physical[seed], task, int(i), source.FULL)
                prefix, terminal = branch["packets"][0], branch["packets"][-1]
                residuals = {"raw_rgb":target-base.finite(cell["center"]["observed"], (192,)),
                    "joint":base.delta(target_qpos, cell["center"]["qpos"], limited)}
                residuals["goal_Q_rgb"] = transform @ residuals["raw_rgb"]
                prefix_residual = {"raw_rgb":target-base.finite(prefix["observed_vector"], (192,)),
                    "joint":base.delta(target_qpos, prefix["qpos"], limited)}
                prefix_residual["goal_Q_rgb"] = transform @ prefix_residual["raw_rgb"]
                terminal_residual = {"raw_rgb":target-base.finite(terminal["observed_vector"], (192,)),
                    "joint":base.delta(target_qpos, terminal["qpos"], limited)}
                terminal_residual["goal_Q_rgb"] = transform @ terminal_residual["raw_rgb"]
                for name in SPACES:
                    spaces[name][i] = {"stage":float(np.mean(prefix_residual[name]**2)),
                        "terminal":float(np.mean(terminal_residual[name]**2)),
                        "initial_stage":initial_joint_cost if name == "joint" else None,"fits":[]}
                for probe in base.response_matrices(cell, task, limited):
                    matrices = {"raw_rgb":probe["matrices"]["physical_rgb_fd"],
                        "goal_Q_rgb":transform @ probe["matrices"]["physical_rgb_fd"],
                        "joint":probe["matrices"]["physical_qpos_fd"]}
                    for name in SPACES:
                        common = base.geometry(matrices[name], common_residual[name])
                        own = base.geometry(matrices[name], residuals[name])
                        fits += 2
                        spaces[name][i]["fits"].append({"scale":probe["scale"],
                            "common_gain_fraction":common["gain_fraction"],
                            "own_objective_mean":own["regularized_objective"]/len(residuals[name]),
                            "common":common,"own":own})
            role_diagnostics = {}
            for role, decision in (("native_candidate", model_plan["decision"]),
                    ("common_candidate", model_plan["historical_common_rule_decision"])):
                role_diagnostics[role] = {"index":roles[role],
                    "model_witness":decision["reconfiguration_witness_found"],
                    "deployment_objective":"own-state" if role == "native_candidate" else "common-residual-gain",
                    "historical_mixed_space_diagnostic":old_checks[seed]["tasks"][task]["mechanism_diagnostics"][role],
                    "spaces":{name:compare_role(spaces[name], roles, role) for name in SPACES}}
            tasks[task] = {"roles":roles,"role_diagnostics":role_diagnostics,"spaces":spaces}
        rows.append({"seed":seed,"source_episode":plan["source_episode"],"tasks":tasks})
    if fits != protocol["cells"]*12:
        raise ValueError("all three spaces/two scales/two residual contracts required")
    aggregate = {}
    for task in base.TASKS:
        aggregate[task] = {}
        for role in ("native_candidate", "common_candidate"):
            role_rows = [r["tasks"][task]["role_diagnostics"][role] for r in rows]
            aggregate[task][role] = {"model_witnesses":sum(r["model_witness"] for r in role_rows),
                "spaces":{name:{
                    "own_state_same_space_conjunction":sum(r["spaces"][name]["own_state_same_space_conjunction"] for r in role_rows),
                    "stage_plus_own_reach_conjunction":sum(r["spaces"][name]["stage_plus_own_reach_conjunction"] for r in role_rows),
                    "with_model_witness":sum(r["model_witness"] and r["spaces"][name]["stage_plus_own_reach_conjunction"] for r in role_rows),
                    "check_counts_among_model_witnesses":{key:sum(r["model_witness"] and r["spaces"][name]["checks"][key] for r in role_rows)
                        for key in role_rows[0]["spaces"][name]["checks"]}}
                    for name in SPACES}}
    return {"schema":"reacher_d2p_mechanism_metric_contract.v1","rows":rows,"aggregate":aggregate,
        "independent_new_fits":fits,"original_verified_fits":historical["independent_bounded_fits"],
        "bindings":{str(p.resolve()):base.digest(p) for p in (path,reference,pred_path)},
        "P1_admission":False,"historic_gates_changed":False,"selection_changed":False,
        "code_defect_confirmed":False,"neural_inference_performed":False,
        "scope":"Posthoc evaluator-contract diagnosis, all8 roots/all3 tasks/all presealed native/common/greedy/zero roles. Common gain and own-state value are different estimands; same-space conjunctions are not retroactive P1 admission, causal controllability proof, oracle deployment or general theory conclusions. Initial RGB anchor is missing, not silently replaced by coasting zero-prefix."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capability", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    archive = args.output.with_suffix(".implementation")
    if args.output.exists() or archive.exists():
        raise FileExistsError("new immutable metric/reference contract output required")
    result = audit(args.capability)
    modules = {"analyzer":Path(__file__),"physical_geometry":Path(base.__file__),
        "source_contract":Path(source.__file__),"goal_metric_verifier":Path(verify_goal_receipt.__code__.co_filename),
        "root_identity":Path(indexed.__code__.co_filename)}
    archive.mkdir()
    result["implementation_sha256"] = {k:base.digest(p) for k,p in modules.items()}
    for key,path in modules.items():
        shutil.copyfile(path, archive/(key+".py"))
        if base.digest(archive/(key+".py")) != result["implementation_sha256"][key]:
            raise RuntimeError("archived audit helper changed")
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({"fits":result["independent_new_fits"],"aggregate":result["aggregate"]}, indent=2))


if __name__ == "__main__":
    main()
