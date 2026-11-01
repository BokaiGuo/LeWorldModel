"""Exploratory bridge: frozen CUDA planner to separately reviewed physical plans.

No call is made at import. No admission, independent preregistration, or
physical authorization is implied. The physical dispatcher is intentionally
blocked until an independently reviewed root/warmup authority is integrated.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time

from experiments.spectral_reachability import codex_p2_frontier_shared_cap_gate_20261030_v1 as gated

frontier = gated.frontier
ARMS = gated.ARMS


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, data):
    raw = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    with Path(path).open("xb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())


def check_sources(bindings):
    need(type(bindings) is dict and bindings, "externally pinned source inventory required")
    for filename, expected in bindings.items():
        path = Path(filename)
        need(path.is_absolute() and path.is_file() and not path.is_symlink() and
             path.resolve() == path and sha(path) == expected, "source or input changed")


def validated_plans(report, bank, ready, compiled, provenances):
    """Validate the actual saved frontier candidates before extracting raw10."""
    need(report["status"] == "exploratory_complete" and report["fixture"] is False and
         report["P1_admission"] is False and report["P2_admission"] is False and
         report["physical_authorized"] is False and report["equal_total_compute"] is False,
         "complete exploratory planner report required")
    need(set(provenances) == set(ARMS) and len(report["results"]) == 2, "both arms required")
    by_arm = {r["arm"]: r for r in report["results"]}
    need(set(by_arm) == set(ARMS), "arm census")
    plans = []
    for arm in ARMS:
        expected_bank = frontier.digest({"bank": bank, "arm": arm, "contract": frontier.contract()})
        need(provenances[arm].get("candidate_bank_sha256") == expected_bank,
             "runtime bank not bound to externally supplied provenance")
        best = frontier.validate_result(by_arm[arm], arm=arm, bank=bank, ready=ready,
                                        prov=provenances[arm], compiled=compiled)
        witness = best["witness"]
        raw = deepcopy(best["actions"])
        need(len(raw) == 10 and witness["receipt"]["actions"] == raw, "exact raw10 witness")
        plan = {"root_sha256": ready["root_sha256"], "root_index": ready["root_index"],
                "arm": arm, "candidate_id": best["candidate_id"], "commands": raw,
                "witness": witness, "provenance": deepcopy(provenances[arm]),
                "root": deepcopy(ready["root"]), "action_domain": deepcopy(ready["action_domain"]),
                "selection_uses_physical_outcome": False}
        plan["selection_sha256"] = frontier.digest(plan)
        plans.append(plan)
    return plans


def host_plan(*, model, device, ready, compiled, provenance_by_arm, caps, output,
              source_bindings, expected_model_state, authorization):
    """Host CUDA only; model and all source authorities must be supplied externally."""
    need(authorization == "EXPLORATORY_P2_HOST_CUDA_COORDINATED", "explicit host coordination")
    need(getattr(device, "type", None) == "cuda", "host CUDA only")
    check_sources(source_bindings)
    need(all(str(Path(p).resolve()) in source_bindings for p in
             (__file__, gated.__file__, frontier.__file__)), "bridge and frontier source bindings required")
    output = Path(output)
    need(output.is_absolute() and output.parent.resolve() == output.parent and
         not os.path.lexists(output), "exclusive output")
    output.mkdir(mode=0o700)
    start = time.monotonic_ns()
    record = {"schema": "codex_p2_host_physical_bridge_v1", "status": "partial_or_failed",
              "P1_admission": False, "P2_admission": False, "physical_authorized": False,
              "equal_total_compute": False, "flops": None}
    try:
        import torch
        from experiments.spectral_reachability import run_pusht_d2p_budget_guard_cuda as checked
        need(torch.cuda.is_available(), "CUDA unavailable")
        def frozen_cuda():
            params = list(model.parameters())
            buffers = list(model.buffers())
            return (bool(params) and all(not m.training for m in model.modules()) and
                    all(not p.requires_grad and p.device == device for p in params) and
                    all(t.device == device for t in buffers))
        need(frozen_cuda(), "frozen eval CUDA model required")
        torch.cuda.synchronize(device)
        before = checked.tensor_state_hash(model)
        need(before == expected_model_state, "externally pinned model state mismatch")
        record.update(model_before=before, source_bindings=deepcopy(source_bindings),
                      ready_sha256=frontier.digest(ready), compiled_sha256=frontier.digest(compiled),
                      provenance_sha256=frontier.digest(provenance_by_arm), caps=deepcopy(caps))
        write(output / "binding.json", record)
        # Uses the unchanged opt-in frontier entry. It does not select physics.
        result = gated.run_native(model=model, device=device, ready=ready, compiled=compiled,
                                  provenance_by_arm=provenance_by_arm,
                                  output=output / "planning", caps=caps)
        torch.cuda.synchronize(device)
        after = checked.tensor_state_hash(model)
        need(after == before and frozen_cuda(), "model drift")
        check_sources(source_bindings)
        bank = frontier.read(output / "planning" / "bank.json")["expanded"]
        plans = validated_plans(result, bank, ready, compiled, provenance_by_arm)
        record.update(status="plans_checked_physics_blocked", model_after=after,
                      planner_report_sha256=sha(output / "planning" / "report.json"),
                      bank_sha256=sha(output / "planning" / "bank.json"),
                      plans=plans, resource_attempts=result["budget"]["attempted"],
                      device_peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
                      device_peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)))
    except BaseException as exc:
        record["failure"] = {"type": type(exc).__name__, "message": str(exc)[:1500]}
        record["wall_ns"] = time.monotonic_ns() - start
        try:
            write(output / "bridge_report.json", record)
        except BaseException:
            pass
        raise
    record["wall_ns"] = time.monotonic_ns() - start
    write(output / "bridge_report.json", record)
    return record


def physical_dispatch(*args, **kwargs):
    """Fail closed: existing physical consumer requires a different sealed report."""
    raise RuntimeError("BLOCKED: new-plan root/warmup/source authority and physical claim not integrated")
