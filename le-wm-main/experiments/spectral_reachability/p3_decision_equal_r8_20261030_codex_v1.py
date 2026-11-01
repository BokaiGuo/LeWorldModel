"""Matched H1/H3 real spectral8 R-group decision seam; no host launcher.

Both panels are paid for every arm before horizon routing. These are new
matched controls, not the unmodified original search tree or a P1 admission.
"""
from copy import deepcopy
import hashlib

import numpy as np

from experiments.spectral_reachability import agent_p3_r_native_orchestrator_20261029_v1 as r
from experiments.spectral_reachability import p3_decision_equal_work_control_20261030_codex_v1 as prior

NAME = "p3_decision_equal_r8_20261030_codex_v1"
core = r.core
need = core.require
POLICIES = ("fixed_h1", "signal", "reverse_signal")
ROSTER = [(3 * root + i, root, policy) for root in range(2) for i, policy in enumerate(POLICIES)]


def add(*vectors):
    return {key: sum(v[key] for v in vectors) for key in core.AXES}


def verification_cost():
    # One whole and one segmented replay of each H1 and H3 selected R edge.
    v = r.live.predict_axes(2 * (1 + 3))
    v["replay_calls"] = 4
    return v


def decision_cost():
    signal = r.live.predict_axes(8)
    signal.update(jacobian_API=2, VJP_directions=384, grad_API=384,
                  raw_backward_callbacks=384, signal_compile_calls=2,
                  array_SVD_calls=2, schedule_decisions=1)
    return add(signal, r.group_cost(1), r.group_cost(3), verification_cost())


def contract():
    return dict(schema=NAME + ".contract", policies=list(POLICIES), horizons=[1, 3],
                decision_gate="original continue/reconfiguration/reconfiguration/8/horizon 1 or 3",
                panels="both spectral8 R groups with separately paid zero and actual postcontext suffix",
                selection="all eligible; min immediate model cost then candidate index within each horizon",
                verification="whole and segmented replay of both winners before routing",
                routing={"fixed_h1": "H1", "signal": "original horizon", "reverse_signal": "4-original horizon"},
                failure="no witness in either panel: stop without packet; no padding or substitution",
                per_decision_axes=decision_cost(), per_slot_decisions=4, fixed_denominator=6,
                same_flops=False, scientific_pass=False, P1_admission=False,
                original_R_equivalence=False, physical_certified=False)


def validate_decision(decision):
    need(type(decision) is dict and decision.get("continue_expanding") is True and
         decision.get("program_type") == "reconfiguration" and
         decision.get("extension_type") == "reconfiguration" and
         type(decision.get("candidates")) is int and decision["candidates"] == 8 and
         type(decision.get("horizon")) is int and decision["horizon"] in (1, 3),
         "unsupported original R decision")


def choose(policy, decision):
    need(policy in POLICIES, "policy")
    validate_decision(decision)
    return 1 if policy == "fixed_h1" else decision["horizon"] if policy == "signal" else 4 - decision["horizon"]


def host_backend_factory(adapter):
    """Bind R Session's call sink to the existing native CUDA tree, not a replay stub.

    The caller still owns a suitable durable sink/capsule, source pins and
    per-slot native feedback. Existing V1 decision terminal is incompatible.
    """
    from experiments.spectral_reachability import p3_decision_native_cuda_adapter_20261030_codex_v1 as a
    need(type(adapter) is a.Adapter and not adapter.poisoned and not adapter.closed, "live native adapter")

    def factory(task, sink):
        need(not adapter.poisoned and adapter.active_slot is not None and callable(getattr(sink, "event", None)),
             "active slot and R session sink")
        root = prior.ROSTER[adapter.active_slot][1]
        need(task == prior.design.parent.parent_task(adapter.prepared["tasks"][root], task["root_context"]),
             "R task bound to active root and observed context")
        try:
            adapter.guard("R_backend_construct")
            tree = a.identity.host.prior.NativeTree(adapter.model, adapter.device, task, sink)
            return a.TrackedBackend(adapter, tree)
        except BaseException:
            adapter.poisoned = True
            adapter.factory.poisoned = True
            raise
    return factory


def execute_decision(root, context, stage, policy, factory, sink, capsule, *,
                     kind="HOST_CUDA", permit_ram=False, guard=None):
    """Paid decision; explicit native backend and durable capsule provider supplied by caller.

    A returned selected raw edge is NOT a physical packet. Caller owns slot,
    bounded durable sink, failure handling, source closure and host supervision.
    """
    need(policy in POLICIES and type(stage) is int and stage in range(4), "decision identity")
    need(kind == "HOST_CUDA" or (kind == "RAM_FIXTURE" and permit_ram is True),
         "explicit HOST CUDA or RAM fixture")
    if guard is None and kind == "RAM_FIXTURE" and permit_ram is True:
        guard = lambda: None
    need(callable(capsule) and callable(guard), "capsule and explicit HOST guard")
    root = core.validate_task(root)
    task = r.parent.parent_task(root, context)
    node = r.parent.root_node(task)
    ledger = core.Ledger(decision_cost())
    session = r.Session(task, factory, sink, ledger, kind, guard, capsule)
    ledger.sink = session.event
    mapped_sink = type("MappedSink", (), {"event": lambda self, e: session.event(e)})()
    adapter = r.parent.ParentNativeAdapter(
        task, lambda local, _: r.SignalBackend(session, local), mapped_sink, permit_ram=permit_ram)
    result = dict(schema=NAME + ".decision", contract=contract(), stage=stage, policy=policy,
                  root_sha256=core.digest(root), context_sha256=core.digest(context),
                  status="attempted", decision=None, diagnostic=None, signal_receipt=None,
                  panels=[], error=None, selected_horizon=None, selected_actions=None,
                  physical_certified=False, scientific_pass=False)
    try:
        native = []
        for h in (1, 3):
            ledger.scope = dict(stage=stage, phase="signal", horizon=h)
            request = core.request(task, node, operation="compile_finite_horizon_J", horizon=h)
            ledger.call("signal_compile", lambda request=request: adapter.signal(request, ledger))
            native.append(deepcopy(adapter.maps[-1]["native_response"]))
        ledger.scope = dict(stage=stage, phase="diagnostic")
        diagnostic = prior.design.signal.spectra(task, native,
            svd=lambda a: ledger.call("array_svd", lambda: np.linalg.svd(a, full_matrices=False)))
        signal_node, signals, signal_receipt = prior.design.signal.runtime_signal(task, native, diagnostic)
        ledger.scope = dict(stage=stage, phase="schedule")
        decision = ledger.call("schedule", lambda: core.choose(task, signal_node, signals, "adaptive"))
        result.update(decision=deepcopy(decision), diagnostic=deepcopy(diagnostic),
                      signal_receipt=deepcopy(signal_receipt))
        session.event(dict(kind="original_decision", stage=stage, decision=deepcopy(decision),
                           diagnostic=deepcopy(diagnostic), signal_receipt=deepcopy(signal_receipt)))
        validate_decision(decision)
        for h in (1, 3):
            ledger.scope = dict(stage=stage, phase="R_group", horizon=h)
            mapping = next(x for x in adapter.maps if x["native_response"]["horizon"] == h)
            reserve = add(verification_cost(), r.group_cost(3)) if h == 1 else verification_cost()
            panel_request = dict(extension_type="reconfiguration", candidates=8, horizon=h)
            group = r.evaluate_R(task, node, panel_request, mapping, session, reserve, session.event,
                                 on_start=lambda g: result["panels"].append(g))
            need(result["panels"][-1] is group, "captured R group identity")
        need(all(g["status"] == "witness_found" and g["eligible_indices"] for g in result["panels"]),
             "R group has no witness; both panels required")
        winners = {}
        for h, group in zip((1, 3), result["panels"], strict=True):
            candidates = [row for row in group["rows"] if row["eligible"] is True]
            need([row["index"] for row in candidates] == group["eligible_indices"], "eligible census")
            winner = min(candidates, key=lambda row: (row["immediate"], row["index"]))
            ledger.scope = dict(stage=stage, phase="verification", horizon=h)
            paths = [ledger.call("replay", lambda segmented=segmented:
                                 session.replay(task, winner["actions"], segmented))
                     for segmented in (False, True)]
            need(paths[0] == paths[1] == winner["path"], "R whole/segmented replay mismatch")
            winners[h] = winner
        need(ledger.attempted == ledger.completed == decision_cost(), "complete matched decision budget")
        selected = choose(policy, decision)
        result.update(status="R_panels_complete_not_certified", selected_horizon=selected,
                      selected_actions=deepcopy(winners[selected]["actions"][:5]),
                      winner_indices={str(h): winners[h]["index"] for h in (1, 3)})
    except Exception as exc:
        result.update(status="failed", error=type(exc).__name__ + ": " + str(exc))
    result["resource_receipt"] = ledger.receipt()
    core.audit_ledger(result["resource_receipt"])
    session.event(dict(kind="R_decision_final", result=deepcopy(result)))
    return result


def execute_slot(root, policy, prior_raw, feedback, factory, sink, capsule, *,
                 kind="HOST_CUDA", permit_ram=False, guard=None):
    """Four decision/real-feedback callbacks; no environment construction here."""
    root = core.validate_task(root)
    need(policy in POLICIES and type(feedback.episode_id) is str and feedback.episode_id,
         "slot feedback identity")
    context = deepcopy(root["root_context"])
    previous = prior.old.f32(prior_raw, (5, 2), True).copy()
    mean, std = np.asarray(root["action_mean"], np.float32), np.asarray(root["action_std"], np.float32)
    result = dict(schema=NAME + ".slot", contract=contract(), policy=policy,
                  root_sha256=core.digest(root), status="attempted", decisions=[], packets=[],
                  error=None, callback_attempted=dict(packet=0, encode=0),
                  callback_completed=dict(packet=0, encode=0),
                  physical_certified=False, scientific_pass=False)
    try:
        for stage in range(4):
            decision = execute_decision(root, context, stage, policy, factory, sink,
                                        lambda cid, call, stage=stage: capsule(stage, cid, call),
                                        kind=kind, permit_ram=permit_ram, guard=guard)
            result["decisions"].append(decision)
            need(decision["status"] == "R_panels_complete_not_certified", "R decision failed: " +
                 str(decision["error"]))
            sent = prior.old.f32(np.asarray(decision["selected_actions"], np.float32), (5, 2), True).copy()
            sink.event(dict(kind="packet_attempt", stage=stage, start=5 * stage, raw=sent.tolist()))
            result["callback_attempted"]["packet"] += 1
            ret = feedback.packet(5 * stage, sent.copy())
            result["callback_completed"]["packet"] += 1
            need(type(ret) is dict and set(ret) == {"raw", "rgb_bytes", "end", "stopped", "episode_id"},
                 "packet ABI")
            raw = np.asarray(ret["raw"])
            need(raw.ndim == 2 and raw.shape[1:] == (2,) and 1 <= len(raw) <= 5, "packet length")
            got = prior.old.f32(raw, (len(raw), 2), True)
            need(ret["episode_id"] == feedback.episode_id and ret["end"] == 5 * stage + len(got) and
                 type(ret["stopped"]) is bool and got.tobytes() == sent[:len(got)].tobytes() and
                 (len(got) == 5 or ret["stopped"]), "packet ACK")
            result["packets"].append(dict(start=5 * stage, end=ret["end"], raw=got.tolist(),
                                          stopped=ret["stopped"]))
            sink.event(dict(kind="packet_return", stage=stage, end=ret["end"], stopped=ret["stopped"]))
            if ret["stopped"]:
                result["status"] = "native_stopped"
                break
            rgb = ret["rgb_bytes"]
            need(type(rgb) is bytes and rgb, "actual RGB")
            rgb_sha = hashlib.sha256(rgb).hexdigest()
            sink.event(dict(kind="encode_attempt", stage=stage, end=ret["end"], rgb_sha256=rgb_sha))
            result["callback_attempted"]["encode"] += 1
            encoded = feedback.encode(ret["end"], rgb)
            result["callback_completed"]["encode"] += 1
            need(type(encoded) is dict and set(encoded) == {"latent", "input_sha256", "end", "episode_id"} and
                 encoded["episode_id"] == feedback.episode_id and encoded["end"] == ret["end"] and
                 encoded["input_sha256"] == rgb_sha, "actual encoder binding")
            context = prior.old.join(context, previous, sent, got, encoded["latent"], mean, std)
            previous = got.copy()
            sink.event(dict(kind="encode_return", stage=stage, end=ret["end"], rgb_sha256=rgb_sha,
                            context_sha256=core.digest(context)))
        else:
            result["status"] = "callback_complete_not_certified"
    except Exception as exc:
        result.update(status="failed", error=type(exc).__name__ + ": " + str(exc))
    sink.event(dict(kind="R_slot_final", result=deepcopy(result)))
    return result


def execute_roster(roots, session_factory, backend_factory, sink, capsule, *,
                   kind="HOST_CUDA", permit_ram=False, guard=None):
    """Six fixed attempts; returned rows are not a native certificate."""
    need(type(roots) is list and len(roots) == 2 and callable(capsule), "two fixed roots and capsule")
    rows = [dict(slot=s, root=root, policy=policy, status="not_started", result=None,
                 error=None, cleanup="not_required") for s, root, policy in ROSTER]
    sink.event(dict(kind="R_fixed_denominator", rows=deepcopy(rows)))
    for row in rows:
        slot = row["slot"]
        sink.event(dict(kind="R_slot_attempt", slot=slot, root=row["root"], policy=row["policy"]))
        close = None
        try:
            feedback, raw, close = session_factory(slot, row["root"], row["policy"])
            need(callable(close), "cleanup callback")
            row["cleanup"] = "pending"
            result = execute_slot(roots[row["root"]], row["policy"], raw, feedback, backend_factory, sink,
                                  lambda stage, cid, call: capsule(slot, stage, cid, call),
                                  kind=kind, permit_ram=permit_ram, guard=guard)
            row["result"], row["status"] = result, result["status"]
        except Exception as exc:
            row["status"], row["error"] = "setup_failed", type(exc).__name__ + ": " + str(exc)
        finally:
            if callable(close):
                try:
                    close()
                    row["cleanup"] = "returned"
                except Exception as exc:
                    row.update(cleanup="failed", status="cleanup_failed",
                               error=type(exc).__name__ + ": " + str(exc))
        sink.event(dict(kind="R_slot_return", row=deepcopy(row)))
    return dict(schema=NAME + ".roster", contract=contract(), fixed_denominator=6,
                rows=rows, source_authenticated=False, physical_certified=False, scientific_pass=False)
