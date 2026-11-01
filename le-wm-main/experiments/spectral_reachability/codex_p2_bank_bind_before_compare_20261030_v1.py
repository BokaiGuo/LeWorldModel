"""Durable bank/provenance binding before native P2 arm evaluation.

This is an opt-in execution component, not a scientific or physical authority.
Production callers must independently seal every input and source.
"""
import json
import os
from pathlib import Path
from copy import deepcopy

from experiments.spectral_reachability import codex_p2_frontier_shared_cap_gate_20261030_v1 as gate_module

frontier = gate_module.frontier
ARMS = gate_module.ARMS


def need(ok, message):
    if not ok:
        raise ValueError(message)


def _write(path, value):
    # Existing writer creates exclusively and fsyncs file AND parent directory.
    frontier.write(path, value)
    return frontier.file_sha(path)


def bind_then_compare(*, directory, ready, compile_bank, make_provenance, run_arm, verify_arm, emit):
    """Orchestration seam: work callbacks supplied by native entry or synthetic tests."""
    directory = Path(directory)
    need(directory.is_dir() and not directory.is_symlink(), "existing exclusive attempt directory required")
    base = compile_bank()
    bank = frontier.candidate_bank(base, ready["root_sha256"])
    bank_path = directory / "bank.json"
    bank_sha = _write(bank_path, {"base": base, "expanded": bank})
    provenances = {arm: make_provenance(bank, arm) for arm in ARMS}
    for arm in ARMS:
        need(provenances[arm]["candidate_bank_sha256"] == frontier.digest(
            {"bank": bank, "arm": arm, "contract": frontier.contract()}), "actual bank provenance")
    binding = {"schema": "codex_p2_runtime_bank_binding_v1",
               "root_sha256": ready["root_sha256"], "bank_file_sha256": bank_sha,
               "bank_digest": frontier.digest(bank), "provenances": provenances,
               "P1_admission": False, "P2_admission": False}
    binding_path = directory / "bank_binding.json"
    binding_sha = _write(binding_path, binding)
    emit({"kind": "bank_bound_before_compare", "bank_file_sha256": bank_sha,
          "binding_sha256": binding_sha})
    results = []
    for arm in ARMS:
        need(frontier.file_sha(bank_path) == bank_sha and
             frontier.file_sha(binding_path) == binding_sha, "bank/binding drift before arm")
        emit({"kind": "arm_started", "arm": arm, "binding_sha256": binding_sha})
        result = run_arm(arm, bank, deepcopy(provenances[arm]))
        need(result["complete"] is True, "incomplete arm")
        verify_arm(result, arm, bank, provenances[arm])
        need(frontier.file_sha(bank_path) == bank_sha and
             frontier.file_sha(binding_path) == binding_sha and
             frontier.digest(bank) == binding["bank_digest"], "bank/binding drift after arm")
        emit({"kind": "arm_verified", "arm": arm, "binding_sha256": binding_sha})
        results.append(result)
    return {"bank_file_sha256": bank_sha, "binding_sha256": binding_sha,
            "provenances": provenances, "results": results}


def execute_native(*, model, device, ready, compiled, protocol, prepared, caps, output):
    """Run real frontier under hooks; caller must verify frozen source/model authority."""
    need(getattr(device, "type", None) == "cuda", "host CUDA only")
    output = Path(output)
    need(output.is_absolute() and output.parent.resolve() == output.parent and
         not os.path.lexists(output), "new absolute exclusive output")
    output.mkdir(mode=0o700)
    (output / "arrays").mkdir(mode=0o700)
    faults, operations, arrays, stages = frontier.Faults(), [], [], []
    try:
        with (output / "events.jsonl").open("x", encoding="utf-8") as stream:
            def sink(event):
                if event.get("kind") in ("jacobian", "suffix_jacobian", "bounded_ridge_svd",
                                         "mode_svd", "suffix_solve"):
                    operations.append(deepcopy(event))
                stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())

            meter = gate_module.SharedCapGate(caps, sink)
            old = frontier.legacy()
            replay = old.full20.RootAnchoredCudaReplay(model, device)
            binding = {"root_sha256": ready["root_sha256"],
                       "protocol_sha256": frontier.digest(protocol),
                       "prepared_sha256": frontier.digest(prepared)}

            def save_shared(stage, value):
                return frontier.persist_array(output, "shared_" + stage, value, binding, arrays, meter.event)

            def compile_bank():
                return frontier.compile_cuda(model, device, ready, compiled, replay,
                                             meter.event, save_shared, faults, stages)

            def suffix(context, prefix, index):
                local = dict(binding, index=index, input_context_sha256=frontier.digest(context),
                             prefix_logits_sha256=frontier.digest(prefix.tolist()))
                def save(stage, value):
                    return frontier.persist_array(output, f"suffix_{index}_{stage}", value, local, arrays, meter.event)
                return frontier.suffix_cuda(model, device, replay, context, prefix, compiled,
                                            index, meter.event, save, faults, stages)

            def run_arm(arm, bank, provenance):
                return frontier.run_arm(arm=arm, bank=bank, root=ready["root"],
                    provenance=provenance, action_domain=ready["action_domain"],
                    target_metric=old.full20.metric_for(compiled), replay=replay,
                    suffix=suffix, journal=meter.event, faults=faults)

            def verify(result, arm, bank, provenance):
                return frontier.validate_result(result, arm=arm, bank=bank, ready=ready,
                                                prov=provenance, compiled=compiled)

            with gate_module.guarded_entries(model, replay, meter, faults) as entries:
                paired = bind_then_compare(
                    directory=output, ready=ready, compile_bank=compile_bank,
                    make_provenance=lambda bank, arm: frontier.provenance(protocol, prepared, ready, bank, arm),
                    run_arm=run_arm, verify_arm=verify, emit=meter.event)
            frontier.validate_actual(entries)
            frontier.validate_operations(operations, paired["results"])
            need(faults.primary is None and not meter.failed, "native execution faults")
            report = {"schema": "codex_p2_bank_bound_frontier_v1", "status": "exploratory_complete",
                      "fixture": False, "binding": binding, "bank_binding_sha256": paired["binding_sha256"],
                      "bank_file_sha256": paired["bank_file_sha256"],
                      "provenances": paired["provenances"], "results": paired["results"],
                      "events_sha256": frontier.file_sha(output / "events.jsonl"),
                      "budget": meter.snapshot(), "P1_admission": False, "P2_admission": False,
                      "equal_total_compute": False, "physical_authorized": False}
            _write(output / "report.json", report)
            return report
    except BaseException as exc:
        try:
            _write(output / "failure.json", {"status": "partial_or_failed",
                   "error_type": type(exc).__name__, "error": str(exc)[:1500],
                   "P2_admission": False, "physical_authorized": False})
        except BaseException:
            pass
        raise
