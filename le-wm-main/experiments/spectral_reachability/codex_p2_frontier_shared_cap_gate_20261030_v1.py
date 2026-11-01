"""Predispatch common-cap gate for the existing P2 frontier execution.

This is an opt-in integration primitive, not a replacement for the frozen
frontier runner, an authority seal, or a complete compute meter.
"""
from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path

from experiments.spectral_reachability import p2_composition_predict_frontier_cuda_20261027_v2 as frontier

ARMS = frontier.ARMS
AXES = ("predict", "action_encoder", "jacobian", "vjp_directions", "svd", "solve")
SHARED_ARM = frontier.legacy().sra.ARM


def require(condition, message):
    if not condition:
        raise ValueError(message)


class SharedCapGate:
    """Charge shared compilation to both standalone arms, before dispatch."""

    def __init__(self, caps, journal):
        require(type(caps) is dict and set(caps) == set(AXES) and
                all(type(v) is int and v >= 0 for v in caps.values()), "exact nonnegative caps required")
        require(callable(journal), "durable journal callback required")
        self.caps, self.journal = deepcopy(caps), journal
        self.attempted = {a: {k: 0 for k in AXES} for a in ARMS}
        self.failed = False

    def event(self, event):
        require(not self.failed, "gate poisoned")
        try:
            charges = self.charges(event)
            next_counts = deepcopy(self.attempted)
            for arm, amounts in charges.items():
                for axis, amount in amounts.items():
                    next_counts[arm][axis] += amount
                    require(next_counts[arm][axis] <= self.caps[axis], "common cap exhausted: " + axis)
            # Both the reservation and original event must be durable before the
            # caller is allowed to enter model/AD/SVD work.
            if charges:
                self.journal({"kind": "common_cap_reservation", "charges": deepcopy(charges),
                              "attempted_after": deepcopy(next_counts)})
                self.attempted = next_counts
            self.journal(deepcopy(event))
        except BaseException:
            self.failed = True
            raise

    def charges(self, event):
        kind = event.get("kind")
        arm, amount = None, {}
        if kind == "entry_attempt":
            entry = event["entry"]
            scope = entry["scope"]
            require(type(scope) is dict, "entry scope missing")
            arm = scope["arm"]
            require(entry["kind"] in ("predict", "action_encoder") and entry["samples"] == 1,
                    "unexpected model entry")
            amount = {entry["kind"]: 1}
        elif event.get("completed") is False and kind in (
                "jacobian", "suffix_jacobian", "bounded_ridge_svd", "mode_svd", "suffix_solve"):
            arm = SHARED_ARM if kind in ("jacobian", "bounded_ridge_svd", "mode_svd") else ARMS[1]
            if kind in ("jacobian", "suffix_jacobian"):
                require(event["shape"] == [192, 20 if kind == "jacobian" else 10] and
                        event["vectorize"] is True, "unexpected Jacobian")
                amount = {"jacobian": 1, "vjp_directions": 192}
            elif kind == "suffix_solve":
                count = event["svd_calls_requested"]
                require(type(count) is int and count in (0, 1), "suffix solve count")
                amount = {"svd": count, "solve": count}
            else:
                amount = {"svd": 1}
                if kind == "bounded_ridge_svd":
                    amount["solve"] = 1
        if not amount:
            return {}
        require(arm in (SHARED_ARM, *ARMS), "unknown arm")
        owners = ARMS if arm == SHARED_ARM else (arm,)
        return {owner: amount for owner in owners}

    def snapshot(self):
        return {"caps": deepcopy(self.caps), "attempted": deepcopy(self.attempted),
                "failed": self.failed, "coverage_complete": False,
                "equal_total_compute": False, "P2_admission": False}


@contextmanager
def guarded_entries(model, replay, gate, faults=None):
    """Native entry hooks; pass *the same* gate.event to compiler and run_arm."""
    require(type(gate) is SharedCapGate and not gate.failed, "fresh shared cap gate required")
    with frontier.guarded_entries(model, replay, gate.event, faults) as entries:
        yield entries


def run_native(*, model, device, ready, compiled, provenance_by_arm, output, caps):
    """Execute existing native frontier under common caps; caller supplies sealed inputs.

    Requires coordinated host CUDA. No model loading, environment or physics.
    This function does not authenticate caller-provided sources or preregistration.
    """
    require(getattr(device, "type", None) == "cuda", "host CUDA only")
    require(type(provenance_by_arm) is dict and set(provenance_by_arm) == set(ARMS),
            "both arm provenance required")
    out = Path(output)
    require(out.is_absolute() and out.parent.is_dir() and out.parent.resolve() == out.parent
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

        gate = SharedCapGate(caps, sink)
        old = frontier.legacy()
        replay = old.full20.RootAnchoredCudaReplay(model, device)
        binding = {"root_sha256": ready["root_sha256"], "experiment": "exploratory_common_cap"}

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
            with guarded_entries(model, replay, gate, faults) as entries:
                base = frontier.compile_cuda(model, device, ready, compiled, replay,
                                             gate.event, save_shared, faults, stages)
                bank = frontier.candidate_bank(base, ready["root_sha256"])
                frontier.write(out / "bank.json", {"base": base, "expanded": bank})
                for arm in ARMS:
                    result = frontier.run_arm(
                        arm=arm, bank=bank, root=ready["root"], provenance=provenance_by_arm[arm],
                        action_domain=ready["action_domain"], target_metric=old.full20.metric_for(compiled),
                        replay=replay, suffix=suffix, journal=gate.event, faults=faults)
                    results.append(result)
                    require(result["complete"] and not gate.failed, "incomplete frontier arm")
            frontier.validate_actual(entries)
            frontier.validate_operations(operations, results)
            require(faults.primary is None, "frontier faults")
            report = {"status": "exploratory_complete", "results": results,
                      "budget": gate.snapshot(), "arrays": arrays,
                      "fixture": False, "P1_admission": False, "P2_admission": False,
                      "physical_authorized": False, "equal_total_compute": False}
            frontier.write(out / "report.json", report)
            return report
        except BaseException as exc:
            # Preserve pending reservations and partial files. Failure report is
            # best effort and never transforms an exception into success.
            try:
                frontier.write(out / "failure.json", {
                    "status": "partial_or_failed", "error_type": type(exc).__name__,
                    "error": str(exc), "budget": gate.snapshot(), "P2_admission": False})
            except BaseException:
                pass
            raise
