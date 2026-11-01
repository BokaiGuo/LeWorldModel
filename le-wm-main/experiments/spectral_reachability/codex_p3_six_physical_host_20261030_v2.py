"""Six-slot known-root physical host; no CUDA, no scientific admission."""
import argparse
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import time

from experiments.spectral_reachability import codex_p3_noshadow_physical_bind_20261030_v1 as bind
from experiments.spectral_reachability import codex_p3_six_physical_contract_20261030_v1 as spec
from experiments.spectral_reachability import codex_p3_six_physical_adapter_20261030_v1 as adapter
from experiments.spectral_reachability import p1_moving_extended_baseline_physical_20261027_v1 as helper

NAME = "codex_p3_six_physical_host_20261030_v2"
ROOT = bind.host.ROOT
BIND = ROOT / "results/codex_p3_noshadow_physical_bind_20261030_v1_01.json"
BIND_SHA = "64d565d18fe7ebe85d65022a7bbc7322b87bb64d648b72337300893f5d719413"
kernel = helper.kernel
require, sha = helper.require, helper.sha


def inputs():
    require(sha(BIND) == BIND_SHA, "pinned saved binding")
    b = bind.host.prior.prior.read(BIND)
    spec.validate_plan(b["rows"])
    require(b["producer_report_sha256"] == bind.REPORT_SHA and b["producer_prepare_sha256"] == bind.PREP_SHA,
            "producer identities")
    require(sha(bind.RUN / "report.json") == bind.REPORT_SHA and
            sha(bind.RUN / "prepare.json") == bind.PREP_SHA, "producer bytes")
    subprocess.run([sys.executable, "-B", "-m",
                    "experiments.spectral_reachability.codex_p3_noshadow_host_20261030_v1",
                    "consume", "--run", str(bind.RUN), "--sha256", bind.REPORT_SHA],
                   check=True, stdout=subprocess.DEVNULL)
    report = bind.host.prior.prior.read(bind.RUN / "report.json")
    events = bind.journal.read(bind.RUN / "events.jsonl", report["journal_tail"], bind.host.CAP)
    data = kernel.load_source(bind.m.SOURCEPACK, bind.m.SOURCEPACK_SHA,
                              bind.m.SOURCE_PREPARE, bind.m.SOURCE_PREPARE_SHA)
    slots = bind.host.prior.prior.read(bind.host.PREVIOUS)["slots"]
    bindings = dict(data["bindings"])
    for root, slot in enumerate(slots):
        bind.physical.task_bridge(slot, data["references"][root], bindings)
    require(bind.summarize(events, slots, data["references"]) == b["rows"], "producer winner reconstruction")
    for path in [Path(__file__).resolve(), Path(adapter.__file__).resolve(), Path(spec.__file__).resolve(),
                 Path(bind.__file__).resolve(), BIND, bind.RUN / "report.json",
                 bind.RUN / "prepare.json", bind.RUN / "events.jsonl",
                 ROOT / "experiments/spectral_reachability/codex_p3_six_physical_saved_20261030_v1.py",
                 ROOT / "experiments/spectral_reachability/codex_p3_six_physical_batch_saved_20261030_v1.py",
                 ROOT / "experiments/spectral_reachability/codex_p3_six_physical_batch_base_20261030_v2.py",
                 ROOT / "experiments/spectral_reachability/codex_p3_six_physical_batch_saved_20261030_v3.py",
                 *helper.physical_files()]:
        path = Path(path).resolve()
        value = sha(path)
        require(str(path) not in bindings or bindings[str(path)] == value, "source hash conflict")
        bindings[str(path)] = value
    native, h = helper.native_constructor_contract()
    bindings[native] = h
    for path, h in bindings.items():
        require(sha(path) == h, "source drift")
    plans = []
    for row in b["rows"]:
        p = dict(root_slot=row["root"], arm=row["policy"], root_sha256=row["root_sha256"],
                 task_id=row["task_id"], commands=deepcopy(row["commands"]),
                 source_candidate_sha256=row["candidate_sha256"], raw_float32_sha256=row["raw_float32_sha256"])
        p["selection_digest"] = kernel.source.digest(p)
        plans.append(p)
    contract = spec.contract()
    contract["wall_seconds"] = 600
    prepare = dict(schema=NAME + ".prepare", source_bind_sha256=BIND_SHA, contract=contract,
                   plans=plans, bindings=bindings, physical_execution=False, scientific_pass=False)
    return prepare, data


def verify(path, expected):
    require(sha(path) == expected, "external prepare SHA")
    saved = helper.read(path)
    actual, data = inputs()
    require(saved == actual, "prepare reconstruction")
    return saved, data


def execute(p, data, store, factory):
    rows = [dict(branch_index=i, root_slot=x["root_slot"], arm=x["arm"],
                 selection_digest=x["selection_digest"], selected_plan=deepcopy(x),
                 status="not_started", first_error=None, secondary_errors=[], warmup_returned=0,
                 own_attempted=0, own_returned=0, initial_exact=False, full_horizon=False,
                 native_success=False, terminated=False, truncated=False, endpoint_scope="not_started",
                 PNG=0, checkpoints=[], constructor_internal_work="not_attempted") for i, x in enumerate(p["plans"])]
    ops = kernel.Ops(p, store)
    errors = []
    try:
        store.emit("run_started", prepare=p)
        for i, (plan, row) in enumerate(zip(p["plans"], rows, strict=True)):
            env, holder = None, []
            ops.where = dict(branch=i, root_slot=plan["root_slot"], arm=plan["arm"],
                             selection_digest=plan["selection_digest"])
            def construct():
                row["constructor_internal_work"] = "unknown"
                value = factory()
                holder.append(value)
                return value
            try:
                env = ops.call("environment_construct", construct)
                kernel.moving.validate_env(env.unwrapped)
                require(all(getattr(env.unwrapped, k, None) is None for k in ("space", "render_buffer", "latest_action")),
                        "pristine constructor")
                require(env.unwrapped.render_action is False and env.unwrapped.render_size == 224 and
                        env.unwrapped.render_mode == "rgb_array", "native RGB interface")
                row["constructor_internal_work"] = "sealed_native_pristine_no_physics"
                row["status"] = "running"
                with kernel.native_hooks(env, ops, row):
                    adapter.run_branch(env, data["references"][plan["root_slot"]], data["plan"],
                                       plan, row, ops, store, i)
            except BaseException as e:
                kernel.retain(row, e, "setup_or_branch")
            finally:
                if env is None and holder:
                    env = holder[0]
                if env is not None:
                    helper.close_env(env, row, ops, store)
                if row["status"] == "failed":
                    row["endpoint_scope"] = "failed_partial"
                store.emit("branch_closed", receipt=row)
    except BaseException as e:
        errors.append({"stage": "execution", "type": type(e).__name__, "message": str(e)})
    try:
        for path, h in p["bindings"].items():
            require(sha(path) == h, "final source drift")
    except BaseException as e:
        errors.append({"stage": "source", "type": type(e).__name__, "message": str(e)})
    complete = not errors and all(r["status"] in ("horizon_completed", "native_success", "native_stopped") and
                                  r["first_error"] is None and not r["secondary_errors"] for r in rows)
    if complete:
        try:
            c = ops.counts
            steps = sum(r["warmup_returned"] + r["own_returned"] for r in rows)
            require(all(v["attempted"] == v["completed"] for v in c.values()) and
                    c["environment_construct"]["completed"] == c["environment_close"]["completed"] == 6 and
                    c["reset"]["completed"] == 12 and c["set_state"]["completed"] == 24 and
                    c["goal_render"]["completed"] == 12 and c["goal_bind"]["completed"] == 6 and
                    c["public_step"]["completed"] == steps and
                    c["physics_step"]["completed"] == 24 + 10 * steps and
                    c["explicit_render"]["completed"] == sum(r["PNG"] for r in rows),
                    "actual resource census")
        except BaseException as e:
            errors.append({"stage": "resource_census", "type": type(e).__name__, "message": str(e)})
            complete = False
    result = dict(schema=NAME + ".report", status="recorded_not_admitted" if complete else "partial_or_failed",
                  rows=rows, errors=errors, counts=ops.counts, contract=p["contract"],
                  elapsed_seconds=time.monotonic()-ops.started, fixed_denominator=6,
                  new_neural_calls=0, scientific_pass=False)
    try:
        store.emit("run_finished", status=result["status"], counts=ops.counts, errors=errors)
        result["journal_sha256"] = sha(store.path / "journal.jsonl")
        result["journal_chain_tail"] = store.previous
    except BaseException as e:
        result["status"] = "partial_or_failed"
        result["errors"].append({"stage": "journal", "type": type(e).__name__, "message": str(e)})
    store.put("report.json", result, final=True)
    return result


def exclusive(path, prefix):
    p = Path(path)
    require(p.is_absolute() and p.parent == ROOT / "results" and p.name.startswith(NAME + prefix)
            and not p.exists() and not p.is_symlink(), "exclusive output")
    return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--prepared")
    parser.add_argument("--sha256")
    args = parser.parse_args()
    if args.action == "prepare":
        path = exclusive(args.output, "_PREP_")
        p, _ = inputs()
        print(bind.host.write_new(path, p))
    else:
        p, data = verify(args.prepared, args.sha256)
        output = exclusive(args.output, "_RUN_")
        store = helper.Store(output)
        store.put("prepare.json", p)
        def factory():
            import gymnasium as gym
            import stable_worldmodel
            return gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224,
                            with_target=False, render_action=False)
        result = execute(p, data, store, factory)
        print(result["status"])
        require(result["status"] == "recorded_not_admitted", "partial preserved, no retry")


if __name__ == "__main__":
    main()
