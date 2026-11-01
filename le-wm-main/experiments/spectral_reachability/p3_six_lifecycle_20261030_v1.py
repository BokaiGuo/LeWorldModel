"""Six-slot injected lifecycle; not a HOST launcher or execution authorization."""
from copy import deepcopy

from experiments.spectral_reachability import p3_feedback_reverse_control_20261030_v1 as control
from experiments.spectral_reachability import p3_six_numeric_bridge_20261030_v1 as bridge
from experiments.spectral_reachability import p3_matched_order_feedback_lifecycle_codex_20261029 as old

need = control.need


def execute(roots, store, create, backend_factory, *, permit_ram=False):
    """Each slot attempted once; create must clean its own partial setup on raise."""
    need(type(roots) is list and len(roots) == 2, "two roots")
    rows = [dict(slot=s, root=r, policy=p, status="not_started", result_sha256=None,
                 cleanup="not_required", error=None) for s, r, p in control.ROSTER]
    store.event("physical", "record", "six_denominator", store.index,
                dict(rows=deepcopy(rows), fixed_denominator=6))
    for row in rows:
        slot, root, policy = row["slot"], row["root"], row["policy"]
        store.event("physical", "record", "six_slot_attempt", store.index,
                    dict(slot=slot, root=root, policy=policy))
        sink = old.prior.ArmSink(bridge.Bridge(store, slot), slot)
        close = None
        try:
            session, raw, close = create(slot, root, policy)
            need(callable(close), "explicit cleanup")
            row["cleanup"] = "pending"
            result = control.execute_slot(roots[root], policy, raw, session, backend_factory, sink,
                                          permit_ram=permit_ram)
            need(result["policy"] == policy, "policy identity")
            row["status"] = result["status"]
            row["result_sha256"] = control.core.digest(result)
        except old.native.NativeInitialStop as exc:
            row.update(status="native_initial_stop", error=str(exc))
        except Exception as exc:
            row.update(status="engineering_failed", error=type(exc).__name__ + ": " + str(exc))
        finally:
            if close is not None:
                try:
                    close()
                    row["cleanup"] = "returned"
                except Exception as exc:
                    row["cleanup"] = "failed"
                    row["error"] = (row["error"] + "; " if row["error"] else "") + \
                                   "cleanup: " + type(exc).__name__ + ": " + str(exc)
                    row["status"] = "engineering_failed"
        store.event("physical", "record", "six_slot_return", store.index, deepcopy(row))
    return dict(rows=rows, fixed_denominator=6, physical_certified=False, scientific_pass=False)
