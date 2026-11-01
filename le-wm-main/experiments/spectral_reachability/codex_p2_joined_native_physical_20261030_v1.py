"""Known-root eight-arm PushT replay; CPU physics only, no neural execution."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import codex_p2_knownroot_physical_join_20261030_v1 as joiner
from experiments.spectral_reachability import p2_composition_predict_frontier_physical_20261027_v1 as native

NAME = "codex_p2_joined_native_physical_20261030_v1"
ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "results/codex_p2_knownroot_host_cuda_20261030_v1_run01"
PREP = ROOT / "results/codex_p2_knownroot_host_cuda_20261030_v1_prepared01"
REF = ROOT / "results/p3_horizon_residual_physical_20261027_v4_source_prepared01"
JOIN = ROOT / "results/codex_p2_knownroot_physical_join_20261030_v1_CPU01.json"
RUN_SHA = "199647e1a7c449053ad8049cca40301f8d89de21b70e8640503a6191f0fbc842"
PREP_SHA = "455038931b3db1a89f34cd5e95f9e40662b01291b84bfd4d4bd523d54892605e"
JOIN_SHA = "68fadebe7cce9fca1b3af33dec1813de11534c447b24d6f35044e07746e34a1e"
need = native.require


def prepare():
    """Reconstruct all eight plans from validated saved planner and physical source."""
    need(joiner.host.frontier.file_sha(JOIN) == JOIN_SHA, "join SHA")
    saved = json.loads(JOIN.read_text())
    rebuilt = joiner.join(run_dir=RUN, run_sha=RUN_SHA, prepared_dir=PREP,
                          prepared_sha=PREP_SHA, reference_dir=REF)
    need(saved == rebuilt, "joined plans changed")
    _, protocol, _ = joiner.host.validate_prepare(PREP, PREP_SHA)
    refs = joiner.physical_source.validate_prepared(REF, joiner.REFERENCE_SHA)
    cases = {c["root_sha256"]: c for c in protocol["context"]["cases"]}
    compiled = {r["root_sha256"]: r["compiled"] for r in
                joiner.host.frontier.read(protocol["context"]["compilation"])["rows"]}
    plans = []
    for row in saved["plans"]:
        case = cases[row["root_sha256"]]
        c = compiled[row["root_sha256"]]
        root_report = joiner.host.frontier.read(RUN / ("root_" + str(row["root_index"])) / "report.json")
        selected = next(r for r in root_report["results"] if r["arm"] == row["arm"])
        receipt = selected["winner"]["witness"]["receipt"]
        need(receipt["actions"] == row["commands"], "saved witness/action mismatch")
        p = dict(schema=native.NAME + ".plan", root_index=row["root_index"],
                 root_sha256=row["root_sha256"], method=row["arm"],
                 metric_label="source_Q", schedule="predict_frontier",
                 active_metric_id=case["metric_sha256"], planned_horizon_tokens=2,
                 planned_commands=10, corrected_search_horizon=2,
                 commands=deepcopy(row["commands"]), goal=deepcopy(c["goal_vectors"][0]),
                 predictor_context=deepcopy(case["root"]["context"]),
                 selected_predicted_context=deepcopy(receipt["terminal_context"]),
                 upstream_selection_sha256=row["selection_sha256"])
        p["selection_sha256"] = native.digest(p)
        native.validate_plan(p)
        plans.append(p)
    need(len(plans) == 8 and sum(p["planned_commands"] for p in plans) == 80,
         "eight fixed raw10 plans")
    execution_sources = {str(Path(path).resolve()): joiner.host.frontier.file_sha(path)
                         for path in (__file__, native.__file__, joiner.__file__)}
    return dict(plans=plans, references=refs["references"], root_seed=refs["root_seed"],
                sources_sha256=refs["sources_sha256"], execution_sources=execution_sources,
                join_sha256=JOIN_SHA)


def execute(output):
    """Native factory is fixed here; no injected callback or fixture entry."""
    p = prepare()
    output = Path(output)
    need(output.is_absolute() and output.parent.resolve() == output.parent and
         output.name.startswith(NAME + "_") and not os.path.lexists(output),
         "new absolute exclusive output")
    native.disk_check(output.parent)
    output.mkdir(mode=0o700)
    native.write(output / "binding.json", dict(schema=NAME + ".binding",
                 plans=p["plans"], references=p["references"], execution_sources=p["execution_sources"],
                 join_sha256=JOIN_SHA,
                 P1_admission=False, P2_admission=False))
    rows, failures, env, meter = [], [], None, None
    budget_path = output / "physical_budget.events.jsonl"
    try:
        with budget_path.open("x") as stream:
            def event(value):
                stream.write(native.io.canonical(value) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            try:
                from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
                from experiments.spectral_reachability.run_pusht_sampling_comparison_20261027_v2 import (
                    verify_rgb_context, official_checked)
                event(dict(kind="constructor", completed=False))
                env = base.gym.make("swm/PushT-v1", render_mode="rgb_array")
                event(dict(kind="constructor", completed=True))
                evaluator = Path(env.unwrapped.eval_state.__code__.co_filename).resolve()
                need(str(evaluator) in p["sources_sha256"] and
                     native.io.sha(evaluator.read_bytes()) == p["sources_sha256"][str(evaluator)],
                     "sealed official evaluator")
                need(tuple(env.unwrapped.variation_space["agent"]["color"].value.tolist()) ==
                     (65, 105, 225), "RoyalBlue source guard")
                meter = native.PhysicalMeter(env, event)
                roots = base.collect_nonterminal_boundary_contact_roots(
                    meter, count=4, seed=int(p["root_seed"]), oversample_factor=4)
                rootmap = native.helpers.validate_roots(roots, p["references"], base.root_fingerprint)
                identities = {}
                for plan in p["plans"]:
                    root = rootmap[plan["root_sha256"]]
                    meter.bind(plan)
                    def reset():
                        meter.phase = "warmup"
                        need(base.root_fingerprint(root) == plan["root_sha256"], "root fingerprint")
                        prior = deepcopy(meter.steps["warmup"])
                        frames, history, goal = base.physical._reset_and_warmup(meter, root, history_len=3)
                        need(all(meter.steps["warmup"][k] - prior[k] == 15
                                 for k in ("attempts", "returns")), "warmup exactly 15")
                        identity = verify_rgb_context(frames, goal, p["references"][plan["root_sha256"]])
                        identity.update(native.helpers.validate_reset_history(
                            history, plan["predictor_context"]["action_history"]),
                            state=base.p0.packets.state_of(env).tolist(),
                            goal=np.asarray(root["goal"]).tolist())
                        if plan["root_sha256"] in identities:
                            need(identity == identities[plan["root_sha256"]], "paired root mismatch")
                        identities[plan["root_sha256"]] = deepcopy(identity)
                        meter.phase = "selected"
                        return identity
                    def evaluate(state):
                        return dict(official=official_checked(env, state, root["goal"]),
                                    pose=base.physical._pose_label(state, root["goal"]))
                    row = native.physical_branch(
                        plan=plan, output=output, reset=reset, env=native.DiskGuardEnv(meter, output),
                        state_of=lambda proxy: base.p0.packets.state_of(proxy.env), evaluate=evaluate)
                    rows.append(row)
                    native.verify_branch(plan, row)
                    need(row["status"] in ("completed", "observation_unavailable",
                                           "environment_terminated"), "branch failure")
                need(meter.steps["selected"]["attempts"] ==
                     sum(r["physical_env_step_attempts"] for r in rows) and
                     meter.steps["selected"]["returns"] ==
                     sum(r["physical_env_step_returns"] for r in rows), "step census")
            except BaseException as exc:
                failures.append(native.durable.failure(exc, "physics"))
            finally:
                if env is not None:
                    try:
                        event(dict(kind="close", completed=False))
                        env.close()
                        event(dict(kind="close", completed=True))
                    except BaseException as exc:
                        failures.append(native.durable.failure(exc, "env_close"))
    except BaseException as exc:
        failures.append(native.durable.failure(exc, "budget_journal"))
    if meter is not None:
        failures.extend(meter.journal_errors)
    try:
        need(prepare() == p, "source or input drift")
    except BaseException as exc:
        failures.append(native.durable.failure(exc, "final_source"))
    native.materialize_missing(p["plans"], rows)
    calls = meter.receipt() if meter is not None else {"unknown": True}
    report = dict(schema=NAME + ".report", status="partial_or_failed",
                  join_sha256=JOIN_SHA, rows=rows, failures=failures,
                  physical_call_receipt=calls,
                  budget_journal_sha256=native.io.sha(budget_path.read_bytes()) if budget_path.exists() else None,
                  binding_sha256=native.io.sha((output / "binding.json").read_bytes()),
                  P1_admission=False, P2_admission=False, new_neural_calls=0,
                  total_resource_matched=False)
    if not failures and len(rows) == 8 and all(r["full_horizon_completed"] and
                                               r["endpoint_fresh"] and not r["failures"] for r in rows):
        report["status"] = "eight_branches_recorded_not_admitted"
    native.write(output / "report.json", report)
    return report


def consume(output, report_sha):
    """Re-read action journals and every saved RGB through historical consumer."""
    output = Path(output).resolve()
    need(joiner.host.frontier.file_sha(output / "report.json") == report_sha, "report SHA")
    report = json.loads((output / "report.json").read_text())
    p = prepare()
    binding_path = output / "binding.json"
    need(joiner.host.frontier.file_sha(binding_path) == report["binding_sha256"], "binding SHA")
    binding = json.loads(binding_path.read_text())
    need(binding["plans"] == p["plans"] and binding["references"] == p["references"] and
         binding["execution_sources"] == p["execution_sources"] and
         binding["join_sha256"] == JOIN_SHA and report["join_sha256"] == JOIN_SHA, "binding")
    need(joiner.host.frontier.file_sha(output / "physical_budget.events.jsonl") ==
         report["budget_journal_sha256"], "budget journal SHA")
    need(report["P1_admission"] is False and report["P2_admission"] is False and
         report["new_neural_calls"] == 0 and len(report["rows"]) == 8, "report boundary")
    for plan, row in zip(p["plans"], report["rows"], strict=True):
        if row["status"] != "blocked":
            native.verify_branch(plan, row)
        else:
            need(row["root_sha256"] == plan["root_sha256"] and row["method"] == plan["method"] and
                 row["selection_sha256"] == plan["selection_sha256"], "blocked identity")
    return dict(status=report["status"], rows=len(report["rows"]), failures=report["failures"],
                P1_admission=False, P2_admission=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("verify-inputs", "execute", "consume"))
    parser.add_argument("--output")
    parser.add_argument("--report-sha256")
    args = parser.parse_args()
    if args.mode == "verify-inputs":
        result = {"status": "inputs_checked_not_executed", "plans": len(prepare()["plans"])}
    elif args.mode == "execute":
        need(args.output and not args.report_sha256, "execute arguments")
        result = execute(args.output)
        result = {"status": result["status"], "report_sha256":
                  joiner.host.frontier.file_sha(Path(args.output) / "report.json")}
    else:
        need(args.output and args.report_sha256, "consume arguments")
        result = consume(args.output, args.report_sha256)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
