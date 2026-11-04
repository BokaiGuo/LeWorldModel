"""Fixed four-prefix by five-suffix intervention for the new spectrum bank."""
import argparse
import inspect
import json
from pathlib import Path

import numpy as np

from . import p1_dev303_residual_spectral_physical_20261102_main as physical
from . import p1_residual_cp5_physical_20261102 as feedback
from .p1_dev303_dynamics_saved_analysis_20261102_main import measures

base, require, ROOT = physical.base, physical.require, physical.ROOT
AUTHORITY = "DEV303_RESIDUAL_COMMON_TAIL_PHYSICS"


def build():
    require(base.sha(physical.REPORT) == physical.REPORT_SHA, "model report SHA")
    model = base.read(physical.REPORT)
    source = physical.build()
    by_label = {row["label"]: row for row in model["rows"]}
    chosen = {"selected": model["rows"][model["selections"]["latent"]],
              "zero": by_label["zero"], "greedy": by_label["task_greedy_direct"],
              "random": by_label["random_control"]}
    plans, _, roles, _ = feedback.source_data()
    require(plans["status"] == "model_only_complete", "sealed feedback plans")
    by_index = {row["index"]: row for row in roles if row["metric"] == "latent"}
    require(len(by_index) == 4, "complete feedback plans")
    tails = {role: by_index[row["index"]]["tail"] for role, row in chosen.items()}
    tails["zero_suffix"] = [[0., 0.]] * 10
    specs = [{"arm": prefix_name + "__" + tail_name, "prefix_role": prefix_name,
              "suffix_role": tail_name, "prefix": row["prefix"], "tail": tail}
             for prefix_name, row in chosen.items() for tail_name, tail in tails.items()]
    return {"schema": "dev303_residual_common_tail.protocol.v1", "specs": specs,
            "source": source, "source_sha256": base.sha(__file__),
            "feedback_plan_sha256": feedback.MODEL_REPORT_SHA,
            "classification": "posthoc_development_mechanism_panel",
            "suffix_rule": "all four latent feedback-planned tails plus zero; no physical success filtering",
            "feedback": "common_suffix_intervention_not_independent_suffix_optimization",
            "P1_admission": False, "independent_confirmation": False}


def execute(contract_path, sha, output, authority):
    require(authority == AUTHORITY and base.sha(contract_path) == sha, "authority/contract SHA")
    contract = base.read(contract_path)
    require(contract == build(), "source closure")
    arrays, receipt, reset = physical.prior.source_data()
    from stable_worldmodel.envs.pusht.env import PushT
    require(base.sha(inspect.getsourcefile(PushT)) == base.native.ENV_SHA, "native source")
    store = base.Store(output)
    store.json("execution_binding.json", {"contract": str(Path(contract_path).resolve()), "sha256": sha})
    rows, errors = [], []
    store.emit("started", prepared_sha256=sha)
    for spec in contract["specs"]:
        row = {**spec, "status": "running", "transitions": [], "observations": [], "error": None}
        env = None
        store.emit("branch_started", arm=spec["arm"])
        try:
            env = PushT(render_mode="rgb_array", resolution=224, with_target=False)
            physical.branch(env, store, spec, receipt["history"], reset, arrays["goal_rgb"], row)
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = {"type": type(exc).__name__, "message": str(exc)}
            errors.append({"arm": spec["arm"], **row["error"]})
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception as exc:
                    errors.append({"arm": spec["arm"], "stage": "close", "message": str(exc)})
        rows.append(row)
        store.json(spec["arm"] + ".json", row)
        store.emit("branch_finished", arm=spec["arm"], status=row["status"])
    require(contract == build(), "postrun closure")
    store.emit("finished", errors=errors)
    report = {"schema": "dev303_residual_common_tail.report.v1", "rows": rows, "errors": errors,
              "contract_sha256": sha, "journal_sha256": base.sha(store.path / "journal.jsonl"),
              "goal_state_evaluation_only": receipt["history"]["goal"],
              "status": "partial_or_failed" if errors else "recorded_development",
              "P1_admission": False, "independent_confirmation": False}
    store.json("report.json", report)
    return {"report_sha256": base.sha(store.path / "report.json"), "status": report["status"]}


def analyze(directory, sha, output):
    directory = Path(directory)
    require(base.sha(directory / "report.json") == sha, "report SHA")
    report = base.read(directory / "report.json")
    require(not report["errors"] and len(report["rows"]) == 20, "complete panel")
    goal = report["goal_state_evaluation_only"]
    baselines = {row["prefix_role"]: row for row in report["rows"] if row["suffix_role"] == "zero_suffix"}
    analysis = []
    for row in report["rows"]:
        require(row["status"] == "fixed_horizon_completed" and len(row["transitions"]) == 25,
                "no censoring in paired analysis")
        require(base.read(directory / (row["arm"] + ".json")) == row, "row shard")
        root, prefix, final = [row["transitions"][i]["state"] for i in (9, 14, 24)]
        base_end = baselines[row["prefix_role"]]["transitions"][-1]["state"]
        initial, end, null = [measures(x, goal) for x in (prefix, final, base_end)]
        analysis.append({"arm": row["arm"], "prefix_role": row["prefix_role"],
                         "suffix_role": row["suffix_role"], "prefix": initial, "terminal": end,
                         "relative_own_zero_suffix": {key: end[key] - null[key] for key in end},
                         "prefix_object_translation": float(np.linalg.norm(np.asarray(prefix)[2:4]
                                                                             - np.asarray(root)[2:4])),
                         "native_terminated": any(t["terminated"] for t in row["transitions"]),
                         "terminal_reward": row["transitions"][-1]["reward"]})
    result = {"schema": "dev303_residual_common_tail.analysis.v1", "physical_report_sha256": sha,
              "rows": analysis, "classification": "posthoc_development",
              "independent_suffix_planning": False, "P1_admission": False}
    return {"analysis_sha256": physical.prior.write_new(output, result)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run", "analyze"))
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--sha")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--authority")
    args = parser.parse_args()
    if args.command == "prepare":
        result = {"sha256": physical.prior.write_new(args.contract, build())}
    elif args.command == "run":
        result = execute(args.contract, args.sha, args.output, args.authority)
    else:
        result = analyze(args.directory, args.sha, args.output)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
