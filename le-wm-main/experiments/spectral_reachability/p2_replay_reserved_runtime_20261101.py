"""Opt-in P2 frontier runner with pre-dispatch replay budget reservation.

Common caps are not equal actual work, equal FLOPs, or scientific admission.
Input seals, device coordination and resources outside the hooks remain caller
responsibilities. This module does not provide an authorization to run CUDA.
"""
from copy import deepcopy
import json
import os
from pathlib import Path

from experiments.spectral_reachability import codex_p2_frontier_shared_cap_gate_20261030_v1 as base

frontier = base.frontier
REPLAY = {"predict": 4, "action_encoder": 4}


def resolve_provenance(source, bank):
    value = source(bank) if callable(source) else source
    base.require(type(value) is dict and set(value) == set(base.ARMS),
                 "both bank-bound arm provenance required")
    return value


class ReplayReservedGate(base.SharedCapGate):
    """Hold each arm's four packet replay entries until their actual dispatch."""

    def __init__(self, caps, journal):
        super().__init__(caps, journal)
        base.require(all(caps[k] >= v for k, v in REPLAY.items()), "replay reserve exceeds cap")
        self.remaining = {arm: dict(REPLAY) for arm in base.ARMS}
        try:
            self.journal({"kind": "replay_hold", "remaining": deepcopy(self.remaining)})
        except BaseException:
            self.failed = True
            raise

    def event(self, event):
        base.require(not self.failed, "gate poisoned")
        try:
            charges = self.charges(event)
            remaining = deepcopy(self.remaining)
            entry = event.get("entry", {}) if event.get("kind") == "entry_attempt" else {}
            scope = entry.get("scope", {})
            if scope.get("phase") == "replay":
                arm, axis = scope.get("arm"), entry.get("kind")
                base.require(arm in base.ARMS and axis in REPLAY and
                             remaining[arm][axis] > 0, "unexpected replay entry")
                remaining[arm][axis] -= 1
            for arm, amounts in charges.items():
                for axis, amount in amounts.items():
                    base.require(self.attempted[arm][axis] + amount +
                                 remaining[arm].get(axis, 0) <= self.caps[axis],
                                 "common cap exhausted after replay hold: " + axis)
            # SharedCapGate durably records the charge before dispatch.
            super().event(event)
            self.remaining = remaining
        except BaseException:
            self.failed = True
            raise

    def snapshot(self):
        result = super().snapshot()
        result["replay_hold_remaining"] = deepcopy(self.remaining)
        result["equal_flops"] = None
        return result


def run_native(*, model, device, ready, compiled, provenance_by_arm, output, caps):
    """Wire existing native compiler and arm loops through replay-reserved caps."""
    base.require(getattr(device, "type", None) == "cuda", "host CUDA only")
    base.require(callable(provenance_by_arm) or
                 (type(provenance_by_arm) is dict and set(provenance_by_arm) == set(base.ARMS)),
                 "both arm provenance or bank-bound factory required")
    out = Path(output)
    base.require(out.is_absolute() and out.parent.is_dir() and out.parent.resolve() == out.parent
                 and not os.path.lexists(out), "new absolute exclusive output required")
    out.mkdir(mode=0o700)
    (out / "arrays").mkdir(mode=0o700)
    faults = frontier.Faults()
    operations, arrays, stages, results = [], [], [], []
    with (out / "events.jsonl").open("x", encoding="utf-8") as stream:
        def sink(event):
            if event.get("kind") in ("jacobian", "suffix_jacobian", "bounded_ridge_svd",
                                     "mode_svd", "suffix_solve"):
                operations.append(deepcopy(event))
            stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

        gate = ReplayReservedGate(caps, sink)
        old = frontier.legacy()
        replay = old.full20.RootAnchoredCudaReplay(model, device)
        binding = {"root_sha256": ready["root_sha256"], "experiment": "exploratory_replay_reserved"}

        def save_shared(stage, value):
            return frontier.persist_array(out, "shared_" + stage, value, binding, arrays, gate.event)

        def suffix(context, prefix, index):
            local = dict(binding, index=index, input_context_sha256=old.digest(context),
                         prefix_logits_sha256=old.digest(prefix.tolist()))

            def save(stage, value):
                return frontier.persist_array(out, f"suffix_{index}_{stage}", value, local, arrays, gate.event)

            return frontier.suffix_cuda(model, device, replay, context, prefix, compiled, index,
                                        gate.event, save, faults, stages)

        try:
            with frontier.guarded_entries(model, replay, gate.event, faults) as entries:
                compiled_bank = frontier.compile_cuda(model, device, ready, compiled, replay,
                                                       gate.event, save_shared, faults, stages)
                bank = frontier.candidate_bank(compiled_bank, ready["root_sha256"])
                frontier.write(out / "bank.json", {"base": compiled_bank, "expanded": bank})
                provenance = resolve_provenance(provenance_by_arm, bank)
                for arm in base.ARMS:
                    result = frontier.run_arm(
                        arm=arm, bank=bank, root=ready["root"], provenance=provenance[arm],
                        action_domain=ready["action_domain"], target_metric=old.full20.metric_for(compiled),
                        replay=replay, suffix=suffix, journal=gate.event, faults=faults)
                    results.append(result)
                    base.require(result["complete"] and not gate.failed, "incomplete frontier arm")
                    base.require(not any(gate.remaining[arm].values()), "incomplete reserved replay")
            frontier.validate_actual(entries)
            frontier.validate_operations(operations, results)
            base.require(faults.primary is None, "frontier faults")
            report = {"status": "exploratory_complete", "results": results,
                      "budget": gate.snapshot(), "arrays": arrays, "fixture": False,
                      "P1_admission": False, "P2_admission": False,
                      "physical_authorized": False, "equal_total_compute": False,
                      "equal_flops": None}
            frontier.write(out / "report.json", report)
            return report
        except BaseException as exc:
            try:
                frontier.write(out / "failure.json", {
                    "status": "partial_or_failed", "error_type": type(exc).__name__,
                    "error": str(exc), "budget": gate.snapshot(), "P2_admission": False})
            except BaseException:
                pass
            raise
