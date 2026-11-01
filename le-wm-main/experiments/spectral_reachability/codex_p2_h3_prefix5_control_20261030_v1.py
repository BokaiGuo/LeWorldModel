"""Fixed joint-prefix plus zero-suffix control; known-root CPU physics."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path

import numpy as np
from experiments.spectral_reachability import codex_p2_h3_rechart_cuda_20261030_v1 as chart
from experiments.spectral_reachability import codex_p2_joined_native_physical_20261030_v1 as prior
from experiments.spectral_reachability import codex_p2_h3_rechart_physical_20261030_v1 as original

NAME = "codex_p2_h3_prefix5_control_20261030_v1"
ROOT = chart.prior.root
CHART = ROOT/"results/codex_p2_h3_rechart_cuda_20261030_v1_RUN01"
CHART_SHA = "840c66efdcfe752f4177138cae9d26bb9e25cb3ccccf2136677d73aaab6c0a30"
need, native, physical = prior.need, prior.native, chart.prior.physical


def build():
    source = original.build()
    plans = []
    for source_plan in source["plans"]:
        if source_plan["method"] != "joint": continue
        p = deepcopy(source_plan)
        p.pop("selection_sha256")
        commands = np.asarray(p["commands"], dtype="<f4").copy()
        commands[5:] = 0
        p.update(method="joint_prefix5_zero10", commands=commands.tolist(),
                 raw_float32_hex=commands.tobytes().hex(),
                 action_provenance="recomputed_joint_prefix5_fixed_zero_suffix10", original_raw_bits_proved=False)
        p["selection_sha256"] = physical.planning.digest(p)
        physical.validate_plan(p)
        plans.append(p)
    need(len(plans) == 4 and [p["root_index"] for p in plans] == [0,1,2,3], "four fixed prefix plans")
    sources = {str(Path(x).resolve()): chart.prior.checks.run.digest_file(x)
               for x in (__file__, original.__file__, chart.__file__, physical.__file__, native.__file__, prior.__file__,
                         ROOT/"tests/test_codex_p2_h3_prefix5_control_20261030_v1.py")}
    return dict(schema=NAME+".binding", plans=plans, references=source["references"],
                root_seed=source["root_seed"], sources_sha256=source["sources_sha256"],
                execution_sources=sources, chart_report_sha256=CHART_SHA,
                budgets=dict(root_collection=0, warmup=60, selected=60, per_arm_selected=15),
                action_provenance="recomputed_joint_prefix5_fixed_zero_suffix10", original_raw_bits_proved=False,
                after_prior_results_known=True, preregistered=False,
                new_neural_calls=0, P1_admission=False, P2_admission=False)


class Meter(native.PhysicalMeter):
    def call(self, name, args, kwargs):
        need(self.phase in self.steps, "explicit physical phase")
        c = (self.steps if name == "step" else self.resets)[self.phase]
        key = (self.binding.get("root_sha256"), self.binding.get("method"), self.phase, name)
        local = self.branches.setdefault(key, dict(attempts=0, returns=0))
        if name == "step":
            cap = dict(root_collection=0, warmup=60, selected=60)[self.phase]
            per = dict(root_collection=0, warmup=15, selected=self.selected_limit)[self.phase]
            need(per is not None and c["attempts"] < cap and local["attempts"] < per, "H3 step cap")
        event = dict(kind="env_"+name, phase=self.phase, index=c["attempts"], completed=False, **self.binding)
        if name == "step": event["command"] = np.asarray(args[0]).tolist()
        self.sink(deepcopy(event))
        c["attempts"] += 1; local["attempts"] += 1
        error = None
        try:
            value = getattr(self.env, name)(*args, **kwargs)
            c["returns"] += 1; local["returns"] += 1; event["completed"] = True
            return value
        except BaseException as exc:
            error = exc; event["error"] = type(exc).__name__+": "+str(exc)
            raise
        finally:
            try: self.sink(deepcopy(event))
            except BaseException as exc:
                self.journal_errors.append(native.durable.failure(exc, "physical_meter_journal"))
                if error is None: raise


def execute(output, prepared, prepared_sha):
    binding = build()
    need(chart.prior.checks.run.digest_file(prepared) == prepared_sha and
         native.io.decode(Path(prepared).read_bytes()) == binding, "fixed prepared control")
    output = Path(output)
    need(output.is_absolute() and output.parent.resolve() == output.parent and
         output.name.startswith(NAME+"_") and not os.path.lexists(output), "exclusive output")
    native.disk_check(output.parent)
    output.mkdir(mode=0o700)
    native.write(output/"binding.json", binding)
    rows, failures, env, meter = [], [], None, None
    budget_path = output/"physical_budget.events.jsonl"
    try:
        with budget_path.open("x") as stream:
            def event(value):
                stream.write(native.io.canonical(value)+"\n"); stream.flush(); os.fsync(stream.fileno())
            try:
                from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
                from experiments.spectral_reachability.run_pusht_sampling_comparison_20261027_v2 import verify_rgb_context, official_checked
                event(dict(kind="constructor", completed=False))
                env = base.gym.make("swm/PushT-v1", render_mode="rgb_array")
                event(dict(kind="constructor", completed=True))
                evaluator = Path(env.unwrapped.eval_state.__code__.co_filename).resolve()
                need(str(evaluator) in binding["sources_sha256"] and
                     native.io.sha(evaluator.read_bytes()) == binding["sources_sha256"][str(evaluator)], "official evaluator")
                need(tuple(env.unwrapped.variation_space["agent"]["color"].value.tolist()) == (65,105,225), "RGB source guard")
                meter = Meter(env, event)
                roots = base.collect_nonterminal_boundary_contact_roots(meter, count=4, seed=binding["root_seed"], oversample_factor=4)
                rootmap = native.helpers.validate_roots(roots, binding["references"], base.root_fingerprint)
                identities = {}
                for plan in binding["plans"]:
                    root = rootmap[plan["root_sha256"]]
                    meter.bind(plan)
                    def reset():
                        meter.phase = "warmup"
                        need(base.root_fingerprint(root) == plan["root_sha256"], "root fingerprint")
                        before = deepcopy(meter.steps["warmup"])
                        frames, history, goal = base.physical._reset_and_warmup(meter, root, history_len=3)
                        need(all(meter.steps["warmup"][k]-before[k] == 15 for k in ("attempts","returns")), "warmup15")
                        identity = verify_rgb_context(frames, goal, binding["references"][plan["root_sha256"]])
                        identity.update(native.helpers.validate_reset_history(history, plan["predictor_context"]["action_history"]),
                                        state=base.p0.packets.state_of(env).tolist(), goal=np.asarray(root["goal"]).tolist())
                        if plan["root_sha256"] in identities: need(identity == identities[plan["root_sha256"]], "paired reset")
                        identities[plan["root_sha256"]] = deepcopy(identity)
                        meter.phase = "selected"
                        return identity
                    def evaluate(state):
                        return dict(official=official_checked(env, state, root["goal"]),
                                    pose=base.physical._pose_label(state, root["goal"]))
                    row = physical.physical_branch(plan=plan, output=output, reset=reset,
                        env=native.DiskGuardEnv(meter, output),
                        state_of=lambda proxy: base.p0.packets.state_of(proxy.env), evaluate=evaluate)
                    rows.append(row)
                    physical.verify_branch(plan, row)
                    need(row["status"] in ("completed", "observation_unavailable", "environment_terminated"), "branch failure")
                need(meter.steps["selected"]["attempts"] == sum(r["physical_env_step_attempts"] for r in rows) and
                     meter.steps["selected"]["returns"] == sum(r["physical_env_step_returns"] for r in rows), "selected census")
            except BaseException as exc:
                failures.append(native.durable.failure(exc, "physics"))
            finally:
                if env is not None:
                    try:
                        event(dict(kind="close", completed=False)); env.close()
                        event(dict(kind="close", completed=True))
                    except BaseException as exc: failures.append(native.durable.failure(exc, "env_close"))
    except BaseException as exc: failures.append(native.durable.failure(exc, "budget_journal"))
    if meter is not None: failures.extend(meter.journal_errors)
    try: need(build() == binding, "source drift")
    except BaseException as exc: failures.append(native.durable.failure(exc, "final_source"))
    physical.materialize_missing(binding["plans"], rows, output)
    report = dict(schema=NAME+".report", status="partial_or_failed", rows=rows, failures=failures,
                  physical_call_receipt=meter.receipt() if meter is not None else {"unknown":True},
                  budget_journal_sha256=chart.prior.checks.run.digest_file(budget_path) if budget_path.exists() else None,
                  binding_sha256=chart.prior.checks.run.digest_file(output/"binding.json"),
                  new_neural_calls=0, original_raw_bits_proved=False, total_resource_matched=False,
                  P1_admission=False, P2_admission=False)
    if not failures and len(rows) == 4 and all(r["status"] in ("completed", "environment_terminated") for r in rows):
        report["status"] = "four_prefix5_controls_recorded_not_admitted"
    native.write(output/"report.json", report)
    return report


def consume(output, sha):
    output = Path(output)
    need(chart.prior.checks.run.digest_file(output/"report.json") == sha, "external report SHA")
    report = native.io.decode((output/"report.json").read_bytes())
    binding = build()
    need(chart.prior.checks.run.digest_file(output/"binding.json") == report["binding_sha256"] and
         native.io.decode((output/"binding.json").read_bytes()) == binding, "binding")
    need(chart.prior.checks.run.digest_file(output/"physical_budget.events.jsonl") == report["budget_journal_sha256"],
         "budget journal")
    need(len(report["rows"]) == 4 and report["new_neural_calls"] == 0, "four rows/no neural")
    for plan, row in zip(binding["plans"], report["rows"], strict=True):
        if row["status"] != "blocked": physical.verify_branch(plan, row)
        else: need(row["selection_sha256"] == plan["selection_sha256"], "blocked identity")
        obs = row.get("trace", [])
        need(all(not (x["terminated"] or x["truncated"]) for x in obs[:-1]), "no actions after terminal")
        if obs and (obs[-1]["terminated"] or obs[-1]["truncated"]):
            need(row["status"] == "environment_terminated", "native early stop")
    return dict(status="saved_branch_checks", producer_status=report["status"], rows=len(report["rows"]),
                failures=report["failures"], original_raw_bits_proved=False,
                total_resource_matched=False, P1_admission=False, P2_admission=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["verify-inputs", "prepare", "execute", "consume"])
    p.add_argument("--output")
    p.add_argument("--sha")
    p.add_argument("--prepared")
    p.add_argument("--prepared-sha")
    a = p.parse_args()
    if a.mode == "verify-inputs": result = dict(status="inputs_checked", plans=len(build()["plans"]))
    elif a.mode == "prepare":
        out = Path(a.output)
        need(out.is_absolute() and out.name.startswith(NAME+"_") and not os.path.lexists(out), "exclusive prepare")
        native.write(out,build())
        result = dict(status="prepared_after_prior_results",sha256=chart.prior.checks.run.digest_file(out))
    elif a.mode == "execute":
        r = execute(a.output,a.prepared,a.prepared_sha); result = dict(status=r["status"], sha256=chart.prior.checks.run.digest_file(Path(a.output)/"report.json"))
    else: result = consume(a.output, a.sha)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__": main()
