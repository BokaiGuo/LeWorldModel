"""Strict saved-only H15 consumer. Never loads a model or dispatches a device.

Independent caller-supplied digests are mandatory. The source manifest covers
the named code surface only: the runner does not record model class sources,
runtime precision, or a shared cross-journal event clock.
"""
import hashlib
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import agent_p3_c3_k17_physical_transport_20261029_v1 as transport
from experiments.spectral_reachability import agent_p3_local_signal_cuda_20261028_v1 as signal
from experiments.spectral_reachability import agent_p3_parent_native_cuda_20261029_v1 as native
from experiments.spectral_reachability import agent_p3_runtime_contract_20261028_v1 as core
from experiments.spectral_reachability import p2_durable_hook_driver_20261030_codex as durable
from experiments.spectral_reachability import p2_h15_cpu_metered_backend_20261030_codex as cpu
from experiments.spectral_reachability import p2_h15_cuda_ledger_runner_20261030_codex as runner
from experiments.spectral_reachability import p2_h15_neural_ledger_adapter_20261030_codex as adapter
from experiments.spectral_reachability import p2_h15_sra_prepare_20261029 as prep
from experiments.spectral_reachability import p2_hook_contract_bridge_20261030_codex as bridge
from experiments.spectral_reachability import p2_preregistered_resource_vector_20261030_codex as spec

NAME = "p2_h15_strict_saved_consumer_20261030_codex"
MODULES = {
    "runner": runner, "adapter": adapter, "signal": signal, "tree": native,
    "cpu": cpu, "driver": durable, "bridge": bridge, "core": core,
    "journal": transport, "contract": spec, "bank_reference": prep,
    "trajectory_audit": native.live, "parent_adapter": native.parent,
    "native_program": native.native,
}


def source_manifest():
    """Build a manifest to pin independently; not a complete dependency graph."""
    return {name: spec.sha(Path(module.__file__).read_bytes()) for name, module in MODULES.items()}


def _read(path, digest, limit):
    path = Path(path)
    core.require(spec.hash_string(digest) and path.is_file() and not path.is_symlink()
                 and path.stat().st_size <= limit, "bounded externally anchored journal")
    data = path.read_bytes()
    core.require(len(data) <= limit and hashlib.sha256(data).hexdigest() == digest, "journal digest")
    return durable.read_chain(data)


def consume(directory, *, native_sha256, ledger_sha256, contract_sha256,
            expected_task, expected_task_sha256, expected_model_hashes,
            expected_manifest, expected_manifest_sha256, arm, run_id):
    """Validate a complete saved plan, refusing partial runs and caller stubs.

    Digests are external anchors, not proof of when commitments were made.
    No result from this function grants admission or establishes fairness.
    """
    task = core.validate_task(expected_task)
    core.require(spec.hash_string(expected_task_sha256) and core.digest(task) == expected_task_sha256,
                 "external task snapshot digest")
    core.require(type(expected_manifest) is dict and set(expected_manifest) == set(MODULES)
                 and all(spec.hash_string(v) for v in expected_manifest.values())
                 and spec.hash_string(expected_manifest_sha256)
                 and core.digest(expected_manifest) == expected_manifest_sha256
                 and source_manifest() == expected_manifest, "external named source manifest")
    core.require(type(expected_model_hashes) is dict
                 and set(expected_model_hashes) == {"original_state_dict", "all_buffers"}
                 and all(spec.hash_string(v) for v in expected_model_hashes.values())
                 and expected_model_hashes["original_state_dict"] == task["binding"]["model_sha256"],
                 "external model state and buffer hashes")
    core.require(type(arm) is str and arm and type(run_id) is str and run_id, "external arm/run")
    directory = Path(directory)
    core.require(directory.is_absolute() and directory.is_dir() and not directory.is_symlink(), "run directory")
    io_path, ledger_path = directory / "native_io.jsonl", directory / "events.jsonl"
    io = _read(io_path, native_sha256, runner.IO_LIMIT)
    ledger = _read(ledger_path, ledger_sha256, durable.MAX_JOURNAL)
    core.require(len(io) >= 2 and io[0]["kind"] == "header" and io[-1]["kind"] == "terminal"
                 and all(row["kind"] == "native" for row in io[1:-1]), "native lifecycle")
    verified = durable.verify(ledger_path, ledger_sha256, contract_sha256)
    # Ensure the bytes used here and by durable.verify have the same external anchor.
    core.require(_read(ledger_path, ledger_sha256, durable.MAX_JOURNAL) == ledger, "ledger stable")
    h, end, lh, lt = io[0]["payload"], io[-1]["payload"], ledger[0]["payload"], ledger[-1]["payload"]
    core.keys(h, ("schema", "contract_sha256", "arm", "run_id", "task", "task_sha256",
                  "model_before", "source_sha256", "runner_sha256"))
    core.keys(end, ("ledger_sha256", "status", "model_after"))
    core.require(h["schema"] == runner.NAME and h["contract_sha256"] == contract_sha256
                 and h["arm"] == lh["arm"] == arm and h["run_id"] == lh["run_id"] == run_id
                 and h["task"] == task and h["task_sha256"] == expected_task_sha256
                 and h["model_before"] == end["model_after"] == expected_model_hashes
                 and end["ledger_sha256"] == ledger_sha256 and end["status"] == verified["status"] == "complete",
                 "task/model/run/terminal binding")
    core.require(all(lh["contract"]["identity"][a] == task["binding"][b]
                     for a, b in adapter.IDENTITY_MAP.items()), "contract task binding")
    core.require(h["source_sha256"] == adapter.sources() and h["runner_sha256"] == expected_manifest["runner"]
                 and lh["source_sha256"] == durable.source_hashes(), "recorded named source binding")

    events = [row["payload"] for row in io[1:-1]]
    resource = [row["payload"] for row in ledger[1:-1]]
    receipt = lt["ledger"]
    # This checks every I/O buffer against its exact ledger operation index and
    # scope; the two journals do not establish cross-chain chronological order.
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
    native.event_census([dict(kind="arm_event", slot=0, call=0, payload=e)
                         for call in calls for e in call])

    neural_source = dict(h["source_sha256"], task_sha256=expected_task_sha256)
    ops = receipt["events"]
    expected_scopes = [dict(stage="signal", index=0, neural_adapter=neural_source)] + [
        dict(stage="replay", index=i, segmented=(i == 41), neural_adapter=neural_source)
        for i in range(42)]
    for call, scope in zip(calls, expected_scopes, strict=True):
        selected = [e for e in resource if e["entry"]["scope"] == scope]
        core.require(bool(selected), "missing per-call resource scope")
        core.require(all(e.get("scope") == scope for e in call[1:-1]), "native event scope mismatch")
        core.require(native.audit_native(call[:-1] + selected + call[-1:]) is True, "native call audit")
    allowed_cpu = []
    for op in ops:
        scope = op["scope"]
        if "neural_adapter" in scope:
            core.require(scope in expected_scopes, "unknown neural scope")
        else:
            allowed_cpu.append(op)
    cpu_source = expected_manifest["cpu"]
    expected_cpu = [
        ("program_compile", dict(phase="compile", work="bank", adapter_sha256=cpu_source)),
        ("solve", dict(phase="compile", work="ridge", adapter_sha256=cpu_source)),
        ("array_svd", dict(phase="compile", work="ridge_svd", adapter_sha256=cpu_source)),
        ("array_svd", dict(phase="compile", work="mode_svd", adapter_sha256=cpu_source)),
    ] + [("metric", dict(phase="selection", candidate=i, adapter_sha256=cpu_source)) for i in range(40)]
    core.require([(e["resource"], e["scope"]) for e in allowed_cpu] == expected_cpu,
                 "exact CPU operation sequence and scopes")
    scopes = []
    for op in ops:
        if not scopes or op["scope"] != scopes[-1]:
            scopes.append(op["scope"])
    expected_order = [expected_scopes[0]] + [scope for _, scope in expected_cpu[:4]]
    for i in range(40):
        expected_order.extend((expected_scopes[i+1], expected_cpu[i+4][1]))
    expected_order.extend(expected_scopes[41:43])
    core.require(scopes == expected_order, "signal/compile/candidate/score/replay ledger order")

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
                     dict(parent_context=task["root_context"], actions=actions, segmented=False),
                     "candidate request and fixed ordering")
        path = call[-1]["result"]
        core.checked_path(task, task["root_context"], actions, path)
        d = np.asarray(path[-1]["latent_history"][-1], np.float64) - np.asarray(task["goal"], np.float64)
        score = float(d @ d / 192)
        core.require(np.isfinite(score), "finite candidate score")
        scored.append(dict(index=i, score=score, path=path))
    winner = min(range(40), key=lambda i: (scored[i]["score"], i))
    core.require(plan["candidates"] == scored and type(plan["winner"]) is int and plan["winner"] == winner,
                 "all candidate scores and winner")
    core.require(type(plan["replays"]) is list and len(plan["replays"]) == 2, "two replay outputs")
    for i, call in enumerate(calls[41:]):
        core.require(call[0]["method"] == "replay" and call[0]["request"] ==
                     dict(parent_context=task["root_context"], actions=bank["raw15"][winner], segmented=bool(i))
                     and call[-1]["result"] == scored[winner]["path"] == plan["replays"][i],
                     "selected whole/segmented replay")
    return dict(schema=NAME + ".verification.v1", task_sha256=expected_task_sha256,
                source_manifest_sha256=expected_manifest_sha256, native_sha256=native_sha256,
                ledger_sha256=ledger_sha256, winner=winner, native_calls=43,
                resources=verified, named_source_surface_verified=True,
                runtime_model_source_closure_verified=False, precision_verified=False,
                cross_journal_chronology_verified=False, coverage_verified=False,
                preregistration_timing_verified=False, budget_fairness=False,
                P1_admission=False, P2_admission=False, physical_release=False)
