"""Strict saved consumer for the exclusive bridge-aware H15 plan.

Three independently anchored chains plus a plan sidecar are checked. This
never executes a model, physical environment, or CUDA, and does not prove
unhooked resource coverage, timing origin, or budget fairness.
"""
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import agent_p3_parent_native_cuda_20261029_v1 as native
from experiments.spectral_reachability import agent_p3_runtime_contract_20261028_v1 as core
from experiments.spectral_reachability import p2_durable_hook_driver_20261030_codex as durable
from experiments.spectral_reachability import p2_h15_budgeted_plan_20261030_codex as budgeted
from experiments.spectral_reachability import p2_h15_cuda_ledger_runner_20261030_codex as runner
from experiments.spectral_reachability import p2_h15_neural_ledger_adapter_20261030_codex as adapter
from experiments.spectral_reachability import p2_h15_observable_budget_bridge_20261030_codex as bridge
from experiments.spectral_reachability import p2_h15_sra_prepare_20261029 as prep
from experiments.spectral_reachability import p2_h15_strict_saved_consumer_20261030_codex as old
from experiments.spectral_reachability import p2_preregistered_resource_vector_20261030_codex as spec

NAME = "p2_h15_budgeted_saved_consumer_20261030_codex"


def consume(directory, budget_directory, *, native_sha256, ledger_sha256, budget_sha256, plan_sha256,
            contract_sha256, expected_task, expected_task_sha256, expected_model_hashes,
            expected_manifest, expected_manifest_sha256, expected_plan_source_sha256,
            expected_bridge_source_sha256, arm, run_id):
    task = core.validate_task(expected_task)
    core.require(spec.hash_string(expected_task_sha256) and core.digest(task) == expected_task_sha256,
                 "external task snapshot digest")
    core.require(type(expected_manifest) is dict and set(expected_manifest) == set(old.MODULES)
                 and all(spec.hash_string(v) for v in expected_manifest.values())
                 and spec.hash_string(expected_manifest_sha256)
                 and core.digest(expected_manifest) == expected_manifest_sha256
                 and old.source_manifest() == expected_manifest, "external named source manifest")
    core.require(spec.hash_string(expected_plan_source_sha256) and budgeted.source_hash() == expected_plan_source_sha256
                 and spec.hash_string(expected_bridge_source_sha256)
                 and bridge.source_hash() == expected_bridge_source_sha256, "external bridge/plan source pins")
    core.require(type(expected_model_hashes) is dict
                 and set(expected_model_hashes) == {"original_state_dict", "all_buffers"}
                 and all(spec.hash_string(v) for v in expected_model_hashes.values())
                 and expected_model_hashes["original_state_dict"] == task["binding"]["model_sha256"],
                 "external model state and buffer hashes")
    core.require(type(arm) is str and bool(arm) and type(run_id) is str and bool(run_id), "external arm/run")
    directory, budget_directory = Path(directory), Path(budget_directory)
    core.require(all(p.is_absolute() and p.is_dir() and not p.is_symlink()
                     for p in (directory, budget_directory)), "absolute run and budget directories")
    io_path, ledger_path = directory / "native_io.jsonl", directory / "events.jsonl"
    budget_path, plan_path = budget_directory / "budget_events.jsonl", directory / "budgeted_plan.jsonl"
    io = old._read(io_path, native_sha256, runner.IO_LIMIT)
    ledger = old._read(ledger_path, ledger_sha256, durable.MAX_JOURNAL)
    budget_rows = old._read(budget_path, budget_sha256, bridge.LIMIT)
    plan_rows = old._read(plan_path, plan_sha256, budgeted.LIMIT)
    verified = durable.verify(ledger_path, ledger_sha256, contract_sha256)
    budget_check = bridge.verify(budget_path, budget_sha256, ledger_path, ledger_sha256,
                                 contract_sha256, arm=arm, run_id=run_id)
    core.require(old._read(ledger_path, ledger_sha256, durable.MAX_JOURNAL) == ledger, "ledger stable")
    core.require(len(io) >= 2 and io[0]["kind"] == "header" and io[-1]["kind"] == "terminal"
                 and all(row["kind"] == "native" for row in io[1:-1]), "native lifecycle")
    h, end, lh, lt = io[0]["payload"], io[-1]["payload"], ledger[0]["payload"], ledger[-1]["payload"]
    core.keys(h, ("schema", "contract_sha256", "arm", "run_id", "task", "task_sha256",
                  "model_before", "source_sha256", "runner_sha256"))
    core.keys(end, ("ledger_sha256", "status", "model_after"))
    core.require(h["schema"] == runner.NAME and h["contract_sha256"] == contract_sha256
                 and h["arm"] == lh["arm"] == arm and h["run_id"] == lh["run_id"] == run_id
                 and h["task"] == task and h["task_sha256"] == expected_task_sha256
                 and h["model_before"] == end["model_after"] == expected_model_hashes
                 and end["ledger_sha256"] == ledger_sha256
                 and end["status"] == verified["status"] == budget_check["status"] == "complete",
                 "task/model/run/terminal binding")
    core.require(all(lh["contract"]["identity"][a] == task["binding"][b]
                     for a, b in adapter.IDENTITY_MAP.items()), "contract task binding")
    core.require(h["source_sha256"] == adapter.sources() and h["runner_sha256"] == expected_manifest["runner"]
                 and lh["source_sha256"] == durable.source_hashes(), "recorded named source binding")
    core.require(len(plan_rows) == 3 and [x["kind"] for x in plan_rows] == ["header", "planned", "terminal"],
                 "plan sidecar lifecycle")
    ph, planned, pt = (x["payload"] for x in plan_rows)
    core.keys(ph, ("schema", "source_sha256", "bridge_sha256", "bridge_attach_sha256",
                   "contract_sha256", "arm", "run_id", "task_sha256"))
    core.keys(planned, ("plan_sha256", "winner", "wall_ns", "replay_count"))
    core.keys(pt, ("native_sha256", "ledger_sha256", "budget_sha256"))
    core.require(ph == dict(schema=budgeted.NAME, source_sha256=expected_plan_source_sha256,
                            bridge_sha256=expected_bridge_source_sha256, bridge_attach_sha256=budget_rows[1]["sha256"],
                            contract_sha256=contract_sha256, arm=arm, run_id=run_id, task_sha256=expected_task_sha256)
                 and pt == dict(native_sha256=native_sha256, ledger_sha256=ledger_sha256, budget_sha256=budget_sha256)
                 and planned["wall_ns"] == budget_check["planning_wall_ns"]
                 and type(planned["replay_count"]) is int and planned["replay_count"] == 42
                 and budget_check["counts"]["replay_calls"] == dict(attempted=42, completed=42),
                 "plan/bridge/ledger cross-binding")

    events = [row["payload"] for row in io[1:-1]]
    resource = [row["payload"] for row in ledger[1:-1]]
    receipt = lt["ledger"]
    native.resource_census(receipt, resource + [e for e in events if "operation_index" in e])
    calls, active, plan = [], None, None
    for event in events:
        kind = event.get("kind")
        if kind == "native_call_attempt":
            core.require(active is None and plan is None and event["task"] == task, "native attempt sequence/task")
            active = [event]
        elif kind == "native_call_return":
            core.require(active is not None, "orphan native return")
            active.append(event)
            calls.append(active)
            active = None
        elif kind == "plan_return":
            core.require(active is None and plan is None and len(calls) == 43, "unique final plan")
            core.keys(event, ("kind", "result"))
            plan = event["result"]
        else:
            core.require(active is not None and kind in native.CALL_ARRAYS | {"direction"},
                         "unexpected native event or failure")
            active.append(event)
    core.require(active is None and plan is not None and len(calls) == 43, "complete 43 call roster")
    native.event_census([dict(kind="arm_event", slot=0, call=0, payload=e) for call in calls for e in call])
    neural_source = dict(h["source_sha256"], task_sha256=expected_task_sha256)
    neural_scopes = [dict(stage="signal", index=0, neural_adapter=neural_source)] + [
        dict(stage="replay", index=i, segmented=(i == 41), neural_adapter=neural_source) for i in range(42)]
    replay_scopes = [dict(stage="observable_replay", index=i, bridge_sha256=expected_bridge_source_sha256)
                     for i in range(42)]
    for call, scope in zip(calls, neural_scopes, strict=True):
        selected = [e for e in resource if e["entry"]["scope"] == scope]
        core.require(bool(selected) and all(e.get("scope") == scope for e in call[1:-1]), "native resource scope")
        core.require(native.audit_native(call[:-1] + selected + call[-1:]) is True, "native call audit")

    cpu_sha = expected_manifest["cpu"]
    expected_cpu = [
        ("program_compile", dict(phase="compile", work="bank", adapter_sha256=cpu_sha)),
        ("solve", dict(phase="compile", work="ridge", adapter_sha256=cpu_sha)),
        ("array_svd", dict(phase="compile", work="ridge_svd", adapter_sha256=cpu_sha)),
        ("array_svd", dict(phase="compile", work="mode_svd", adapter_sha256=cpu_sha)),
    ] + [("metric", dict(phase="selection", candidate=i, adapter_sha256=cpu_sha)) for i in range(40)]
    ops = receipt["events"]
    cpu_ops = [op for op in ops if op["scope"] not in neural_scopes and op["scope"] not in replay_scopes]
    core.require([(op["resource"], op["scope"]) for op in cpu_ops] == expected_cpu, "exact CPU operations")
    outer_ops = [op for op in ops if op["scope"] in replay_scopes]
    core.require([(op["resource"], op["scope"], op["status"]) for op in outer_ops] ==
                 [("replay", scope, "returned") for scope in replay_scopes], "42 replay operations")
    scopes = []
    for op in ops:
        if not scopes or op["scope"] != scopes[-1]:
            scopes.append(op["scope"])
    expected_order = [neural_scopes[0]] + [scope for _, scope in expected_cpu[:4]]
    for i in range(40):
        expected_order.extend((replay_scopes[i], neural_scopes[i+1], expected_cpu[i+4][1]))
    for i in (40, 41):
        expected_order.extend((replay_scopes[i], neural_scopes[i+1]))
    core.require(scopes == expected_order, "plan ledger scope order")
    stack = []
    for item in resource:
        op, kind = item["entry"], item["kind"]
        if kind == "resource_attempt":
            scope = op["scope"]
            if scope in neural_scopes[1:]:
                i = scope["index"]
                core.require(any(x["scope"] == replay_scopes[i] for x in stack), "native replay nested in matching outer replay")
            elif scope in replay_scopes:
                core.require(not stack, "no nested outer replay")
            else:
                core.require(not any(x["scope"] in replay_scopes for x in stack), "no foreign work inside replay")
            stack.append(op)
        elif kind in ("resource_return", "resource_failure"):
            core.require(stack and stack.pop()["index"] == op["index"], "resource nesting")
        else:
            raise ValueError("blocked resource in complete plan")
    core.require(not stack, "closed resource stack")

    req = core.request(task, dict(context=task["root_context"]), operation="compile_finite_horizon_J", horizon=3)
    core.require(calls[0][0]["method"] == "signal" and calls[0][0]["request"] == req, "H15 signal request")
    response = calls[0][-1]["result"]
    bank = prep.spectral_bank(response["jacobian"], response["nominal"], task["goal"])
    core.keys(plan, ("bank", "candidates", "winner", "replays"))
    core.require(core.canonical(bank) == core.canonical(plan["bank"]), "independent exact bank recomputation")
    scored = []
    for i in range(40):
        call, actions = calls[i+1], bank["raw15"][i]
        core.require(call[0]["method"] == "replay" and call[0]["request"] ==
                     dict(parent_context=task["root_context"], actions=actions, segmented=False), "candidate request")
        path = call[-1]["result"]
        core.checked_path(task, task["root_context"], actions, path)
        delta = np.asarray(path[-1]["latent_history"][-1], np.float64) - np.asarray(task["goal"], np.float64)
        score = float(delta @ delta / 192)
        core.require(np.isfinite(score), "finite candidate score")
        scored.append(dict(index=i, score=score, path=path))
    winner = min(range(40), key=lambda i: (scored[i]["score"], i))
    core.require(plan["candidates"] == scored and type(plan["winner"]) is int and plan["winner"] == winner
                 and planned["winner"] == winner and planned["plan_sha256"] == core.digest(plan),
                 "candidate scores/winner/plan binding")
    core.require(type(plan["replays"]) is list and len(plan["replays"]) == 2, "two replay outputs")
    for i, call in enumerate(calls[41:]):
        core.require(call[0]["method"] == "replay" and call[0]["request"] ==
                     dict(parent_context=task["root_context"], actions=bank["raw15"][winner], segmented=bool(i))
                     and call[-1]["result"] == scored[winner]["path"] == plan["replays"][i], "selected replay")
    return dict(schema=NAME + ".verification.v1", winner=winner, native_calls=43,
                resources=verified, budget=budget_check, plan_sha256=plan_sha256,
                runtime_model_source_closure_verified=False, precision_verified=False,
                cross_journal_chronology_verified=False, coverage_verified=False,
                preregistration_timing_verified=False, budget_fairness=False,
                P1_admission=False, P2_admission=False, physical_release=False)
