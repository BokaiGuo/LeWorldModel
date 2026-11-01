"""Three-arm observed-feedback negative-control seam; no host or certification.

Injected callback runner only. No model, simulator, GPU or output filesystem is
constructed here. A backend label or test fixture never authenticates execution.
"""
from collections import Counter
from copy import deepcopy
import hashlib

import numpy as np

from experiments.spectral_reachability import p3_matched_order_feedback_seam_codex_20261029 as old

NAME = "p3_feedback_reverse_control_20261030_v1"
core, design = old.core, old.design
POLICIES = ("fixed_order", "signal_order", "reverse_signal_order")
ROSTER = [(3 * root + arm, root, policy)
          for root in range(2) for arm, policy in enumerate(POLICIES)]
need = old.need


def choose(policy, decision, remaining, stage):
    """Reverse only the original admissible first horizon, never the gate."""
    need(policy in POLICIES, "policy")
    if policy == "reverse_signal_order":
        selected = design.choose("signal_order", decision, remaining, stage)
        return (4 - selected) if stage == 0 else selected
    return design.choose(policy, decision, remaining, stage)


def compile_block(root, context, policy, remaining, stage, factory, store, ledger, permit_ram=False):
    """Reuse original real signal, endpoint bank and complete replay primitives."""
    task = design.parent.parent_task(root, context)
    backend = factory(task, store)
    need(getattr(backend, "kind", None) == "HOST_CUDA" or
         (permit_ram is True and getattr(backend, "kind", None) == "RAM_FIXTURE"), "backend kind")
    responses = []
    for h in (1, 3):
        ledger.scope = dict(stage=stage, phase="signal", horizon=h)
        req = core.request(task, dict(context=context), operation="compile_finite_horizon_J", horizon=h)
        response = ledger.call("signal_compile", lambda req=req: backend.signal(req, ledger))
        need(response["binding_sha256"] == req["binding_sha256"] and
             response["parent_context_sha256"] == req["parent_context_sha256"] and
             response["horizon"] == h, "signal binding")
        responses.append(response)
    ledger.scope = dict(stage=stage, phase="diagnostic")
    diagnostic = design.signal.spectra(task, responses,
        svd=lambda a: ledger.call("array_svd", lambda: np.linalg.svd(a, full_matrices=False)))
    node, signals, _ = design.signal.runtime_signal(task, responses, diagnostic)
    ledger.scope = dict(stage=stage, phase="schedule")
    decision = ledger.call("schedule", lambda: core.choose(task, node, signals, "adaptive"))
    horizon = choose(policy, decision, remaining, stage)
    ledger.scope = dict(stage=stage, phase="program", horizon=horizon)
    bank = ledger.call("program_compile", lambda: design.frozen.generate(
        task, responses[0 if horizon == 1 else 1], "endpoints", ledger))
    candidates = []
    for index, actions in enumerate(bank["actions"]):
        ledger.scope = dict(stage=stage, phase="candidate", horizon=horizon, index=index)
        whole = ledger.call("replay", lambda: backend.replay(context, actions, False, ledger))
        segmented = ledger.call("replay", lambda: backend.replay(context, actions, True, ledger))
        need(whole == segmented, "whole/segmented mismatch")
        terminal = core.checked_path(task, context, actions, whole)
        cost = ledger.call("metric", lambda: core.model_cost(task, terminal))
        candidates.append(dict(index=index, actions=actions, whole=whole, segmented=segmented, cost=cost))
    need(len(candidates) == 4, "four candidates")
    winner = min(candidates, key=lambda c: (c["cost"], c["index"]))
    ledger.scope = dict(stage=stage, phase="winner", horizon=horizon, index=winner["index"])
    path = ledger.call("replay", lambda: backend.replay(context, winner["actions"], False, ledger))
    need(path == winner["whole"], "winner replay mismatch")
    core.checked_path(task, context, winner["actions"], path)
    return dict(stage=stage, context_sha256=core.digest(context), responses=responses,
                diagnostic=diagnostic, decision=decision, horizon=horizon, bank=bank,
                candidates=candidates, winner_index=winner["index"], actions=winner["actions"])


def execute_slot(root, policy, prior_raw, session, factory, store, *, permit_ram=False):
    """Actual callback feedback at every 5 actions; errors preserve attempted work."""
    root = core.validate_task(root)
    need(policy in POLICIES, "policy")
    context = deepcopy(root["root_context"])
    previous = old.f32(prior_raw, (5, 2), True).copy()
    mean, std = np.asarray(root["action_mean"], np.float32), np.asarray(root["action_std"], np.float32)
    episode = session.episode_id
    need(type(episode) is str and bool(episode), "episode identity")
    ledger = core.Ledger(design.expected(), sink=store.event)
    remaining = [1, 3]
    attempted, completed = Counter(), Counter()
    result = dict(schema=NAME + ".slot", policy=policy, root_sha256=core.digest(root),
                  status="attempted", blocks=[], packets=[], errors=[],
                  callback_receipts=[], source_authenticated=False,
                  native_execution_certified=False, scientific_pass=False)

    def invoke(name, fn):
        attempted[name] += 1
        store.event(dict(kind="callback_attempt", callback=name, ordinal=attempted[name]))
        try:
            output = fn()
        except Exception as exc:
            store.event(dict(kind="callback_failure", callback=name, ordinal=attempted[name],
                             error=type(exc).__name__))
            raise
        completed[name] += 1
        store.event(dict(kind="callback_return", callback=name, ordinal=attempted[name]))
        return output

    try:
        for stage in range(2):
            block = compile_block(root, context, policy, remaining, stage, factory, store, ledger, permit_ram)
            remaining.remove(block["horizon"])
            result["blocks"].append(block)
            store.event(dict(kind="block_compiled", stage=stage, context_sha256=block["context_sha256"],
                             horizon=block["horizon"], actions=deepcopy(block["actions"])))
            commands = old.f32(np.asarray(block["actions"], dtype=np.float32), (5 * block["horizon"], 2), True)
            for offset in range(0, len(commands), 5):
                start = len(result["packets"]) * 5
                sent = commands[offset:offset + 5].copy()
                store.event(dict(kind="packet_intent", stage=stage, start=start, raw=sent.tolist()))
                ret = invoke("packet", lambda: session.packet(start, sent.copy()))
                need(type(ret) is dict and set(ret) == {"raw", "rgb_bytes", "end", "stopped", "episode_id"}, "packet ABI")
                got = np.asarray(ret["raw"])
                need(got.ndim == 2 and got.shape[1:] == (2,) and 1 <= len(got) <= 5, "returned packet length")
                got = old.f32(got, (len(got), 2), True)
                need(ret["episode_id"] == episode and ret["end"] == start + len(got) and
                     type(ret["stopped"]) is bool and got.tobytes() == sent[:len(got)].tobytes() and
                     (len(got) == 5 or ret["stopped"]), "packet exact same episode ACK")
                packet = dict(stage=stage, start=start, end=ret["end"], raw=got.tolist(), stopped=ret["stopped"])
                result["packets"].append(packet)
                store.event(dict(kind="packet_return", **packet))
                if ret["stopped"]:
                    result["status"] = "native_stopped"
                    return result
                rgb = ret["rgb_bytes"]
                need(type(rgb) is bytes and len(rgb) > 0, "actual RGB bytes")
                rgb_sha = hashlib.sha256(rgb).hexdigest()
                enc = invoke("encode", lambda: session.encode(ret["end"], rgb))
                need(type(enc) is dict and set(enc) == {"latent", "input_sha256", "end", "episode_id"} and
                     enc["episode_id"] == episode and enc["end"] == ret["end"] and
                     enc["input_sha256"] == rgb_sha, "same actual RGB encoder binding")
                context = old.join(context, previous, sent, got, enc["latent"], mean, std)
                previous = got.copy()
                store.event(dict(kind="actual_context", stage=stage, end=ret["end"], rgb_sha256=rgb_sha,
                                 context_sha256=core.digest(context)))
        need(not remaining and ledger.attempted == ledger.completed == design.expected(), "complete measured axes")
        result["status"] = "callback_feedback_complete_not_certified"
    except Exception as exc:
        result["status"] = "failed"
        result["errors"].append(type(exc).__name__ + ": " + str(exc))
    finally:
        result["resource_receipt"] = ledger.receipt()
        result["callback_receipts"] = dict(attempted=dict(attempted), completed=dict(completed),
                                           physical_steps=None, encoder_kernels=None, FLOPs=None)
        store.event(dict(kind="slot_final", result=deepcopy(result)))
    return result


def execute_roster(roots, session_factory, backend_factory, store, *, permit_ram=False):
    """Fixed six-slot callback lifecycle with cleanup and no retries."""
    need(type(roots) is list and len(roots) == 2, "two fixed roots")
    rows = [dict(slot=s, root=r, policy=p, status="not_started", result=None, error=None, cleanup="not_required")
            for s, r, p in ROSTER]
    store.event(dict(kind="fixed_denominator", rows=deepcopy(rows)))
    for row in rows:
        store.event(dict(kind="slot_attempt", slot=row["slot"], root=row["root"], policy=row["policy"]))
        close = None
        try:
            session, raw, close = session_factory(row["slot"], row["root"], row["policy"])
            need(callable(close), "cleanup callback")
            row["cleanup"] = "pending"
            result = execute_slot(roots[row["root"]], row["policy"], raw, session,
                                  backend_factory, store, permit_ram=permit_ram)
            row["result"], row["status"] = result, result["status"]
        except Exception as exc:
            row["status"], row["error"] = "setup_failed", type(exc).__name__ + ": " + str(exc)
        finally:
            if close is not None:
                try:
                    close()
                    row["cleanup"] = "returned"
                except Exception as exc:
                    row["cleanup"] = "failed"
                    row["status"] = "cleanup_failed"
                    row["error"] = type(exc).__name__ + ": " + str(exc)
        store.event(dict(kind="roster_slot_return", row=deepcopy(row)))
    return dict(schema=NAME + ".roster", fixed_denominator=6, rows=rows,
                native_execution_certified=False, scientific_pass=False)


def compare(a, b):
    """Structural accounting only; neither sources nor callback reality certified."""
    need(a["root_sha256"] == b["root_sha256"] and a["policy"] in POLICIES and b["policy"] in POLICIES
         and a["policy"] != b["policy"], "paired identity")
    if any(x["status"] != "callback_feedback_complete_not_certified" for x in (a, b)):
        return dict(matched_measured_axes=None, matched_hook_shape_multiset=None,
                    callback_counts_match=None, scientific_pass=False)
    for x in (a, b):
        core.audit_ledger(x["resource_receipt"])
        need(x["resource_receipt"]["caps"] == design.expected() and
             x["resource_receipt"]["attempted"] == x["resource_receipt"]["completed"] == design.expected(),
             "complete common budget")
        need(len(x["blocks"]) == 2 and sorted(y["horizon"] for y in x["blocks"]) == [1, 3], "quota")
        remaining = [1, 3]
        for i, block in enumerate(x["blocks"]):
            need(block["stage"] == i and block["horizon"] == choose(x["policy"], block["decision"], remaining, i),
                 "original decision and intervention")
            remaining.remove(block["horizon"])
        need(x["callback_receipts"]["attempted"] == x["callback_receipts"]["completed"] ==
             {"packet": 4, "encode": 4}, "complete callback counts")
    def shape(x):
        return Counter(core.canonical([e["resource"], e["shape"], e["directions"], e["status"]])
                       for e in x["resource_receipt"]["events"])
    return dict(matched_measured_axes=True, matched_hook_shape_multiset=shape(a) == shape(b),
                callback_counts_match=True, physical_meter_match=None, FLOPs=None,
                native_execution_certified=False, scientific_pass=False)
