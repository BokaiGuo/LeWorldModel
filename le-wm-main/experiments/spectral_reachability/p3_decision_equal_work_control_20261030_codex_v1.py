"""Four feedback decisions with paid H1/H3 panels before signal/reverse routing.

Callback seam only: no native host, production launcher, or source certification.
"""
from collections import Counter
import argparse
from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import p3_matched_order_feedback_seam_codex_20261029 as old

NAME = "p3_decision_equal_work_control_20261030_codex_v1"
core, design = old.core, old.design
POLICIES = ("fixed_h1", "signal", "reverse_signal")
ROSTER = [(3 * root + i, root, policy) for root in range(2) for i, policy in enumerate(POLICIES)]


def expected():
    # One H1+H3 signal (8 predict), both four-candidate banks plus winner
    # (9*1 + 9*3 predict). These axes overlap and must never be summed.
    caps = design.live.predict_axes(44)
    caps.update(jacobian_API=2, VJP_directions=384, grad_API=384, raw_backward_callbacks=384,
                array_SVD_calls=4, solve_calls=2, signal_compile_calls=2, program_compile_calls=2,
                replay_calls=18, metric_calls=8, schedule_decisions=1)
    return caps


def contract():
    return dict(schema=NAME + ".contract", policies=list(POLICIES), roster=[list(x) for x in ROSTER],
                decisions=[0, 5, 10, 15], packet_steps=5, horizons=[1, 3],
                per_decision_caps=expected(), common_gate="continue/progress/progress with horizon in {1,3}",
                execution="only first 5 actions of selected full-horizon winner; replan after actual RGB",
                primary="per-root native7 distance signal minus reverse at step 20",
                failure="preserve attempted work and all slots; no retry, padding, refund or fallback",
                known_roots=True, scientific_pass=False)


def choose(policy, decision):
    old.need(policy in POLICIES, "policy")
    old.need(type(decision) is dict and decision.get("continue_expanding") is True and
             decision.get("program_type") == "progress" and decision.get("extension_type") == "progress"
             and type(decision.get("horizon")) is int and decision["horizon"] in (1, 3), "unsupported decision")
    return 1 if policy == "fixed_h1" else decision["horizon"] if policy == "signal" else 4 - decision["horizon"]


def decision_work(root, context, stage, factory, store, ledger, permit_ram=False):
    task = design.parent.parent_task(root, context)
    backend = factory(task, store)
    old.need(getattr(backend, "kind", None) == "HOST_CUDA" or
             (permit_ram is True and getattr(backend, "kind", None) == "RAM_FIXTURE"), "backend kind")
    responses = []
    for h in (1, 3):
        ledger.scope = dict(stage=stage, phase="signal", horizon=h)
        req = core.request(task, dict(context=context), operation="compile_finite_horizon_J", horizon=h)
        r = ledger.call("signal_compile", lambda req=req: backend.signal(req, ledger))
        old.need(r["binding_sha256"] == req["binding_sha256"] and
                 r["parent_context_sha256"] == req["parent_context_sha256"] and r["horizon"] == h,
                 "signal binding")
        responses.append(r)
    ledger.scope = dict(stage=stage, phase="diagnostic")
    diagnostic = design.signal.spectra(task, responses,
        svd=lambda a: ledger.call("array_svd", lambda: np.linalg.svd(a, full_matrices=False)))
    node, signals, _ = design.signal.runtime_signal(task, responses, diagnostic)
    ledger.scope = dict(stage=stage, phase="schedule")
    decision = ledger.call("schedule", lambda: core.choose(task, node, signals, "adaptive"))
    # All policies share this fail-stop gate, before any panel is dispatched.
    choose("signal", decision)
    panels = []
    for h, response in zip((1, 3), responses, strict=True):
        ledger.scope = dict(stage=stage, phase="program", horizon=h)
        bank = ledger.call("program_compile", lambda: design.frozen.generate(task, response, "endpoints", ledger))
        candidates = []
        for index, actions in enumerate(bank["actions"]):
            ledger.scope = dict(stage=stage, phase="candidate", horizon=h, index=index)
            whole = ledger.call("replay", lambda: backend.replay(context, actions, False, ledger))
            segmented = ledger.call("replay", lambda: backend.replay(context, actions, True, ledger))
            old.need(whole == segmented, "whole/segmented mismatch")
            terminal = core.checked_path(task, context, actions, whole)
            cost = ledger.call("metric", lambda: core.model_cost(task, terminal))
            candidates.append(dict(index=index, actions=actions, whole=whole, segmented=segmented, cost=cost))
        old.need(len(candidates) == 4, "four candidates")
        winner = min(candidates, key=lambda c: (c["cost"], c["index"]))
        ledger.scope = dict(stage=stage, phase="winner", horizon=h, index=winner["index"])
        path = ledger.call("replay", lambda: backend.replay(context, winner["actions"], False, ledger))
        old.need(path == winner["whole"], "winner replay mismatch")
        core.checked_path(task, context, winner["actions"], path)
        panels.append(dict(horizon=h, bank=bank, candidates=candidates, winner_index=winner["index"],
                           winner_actions=winner["actions"]))
    old.need(ledger.attempted == ledger.completed == expected(), "decision measured work")
    return dict(stage=stage, context_sha256=core.digest(context), responses=responses,
                diagnostic=diagnostic, decision=decision, panels=panels)


def execute_slot(root, policy, prior_raw, session, factory, store, *, permit_ram=False):
    root = core.validate_task(root)
    old.need(policy in POLICIES and type(session.episode_id) is str and bool(session.episode_id), "slot identity")
    context = deepcopy(root["root_context"])
    previous = old.f32(prior_raw, (5, 2), True).copy()
    mean, std = np.asarray(root["action_mean"], np.float32), np.asarray(root["action_std"], np.float32)
    result = dict(schema=NAME + ".slot", contract=contract(), root_sha256=core.digest(root), policy=policy,
                  status="attempted", decisions=[], packets=[], error=None,
                  callback_attempted=dict(packet=0, encode=0), callback_completed=dict(packet=0, encode=0),
                  physical_certified=False, scientific_pass=False)
    ledger = None
    try:
        for stage in range(4):
            ledger = core.Ledger(expected(), sink=store.event)
            block = decision_work(root, context, stage, factory, store, ledger, permit_ram)
            horizon = choose(policy, block["decision"])
            block["selected_horizon"] = horizon
            block["resource_receipt"] = ledger.receipt()
            core.audit_ledger(block["resource_receipt"])
            result["decisions"].append(block)
            store.event(dict(kind="decision_paid", stage=stage, context_sha256=block["context_sha256"],
                             selected_horizon=horizon, resource_receipt=block["resource_receipt"]))
            sent = old.f32(np.asarray(block["panels"][0 if horizon == 1 else 1]["winner_actions"],
                                      dtype=np.float32), (5 * horizon, 2), True)[:5].copy()
            store.event(dict(kind="packet_attempt", stage=stage, start=stage * 5, raw=sent.tolist()))
            result["callback_attempted"]["packet"] += 1
            ret = session.packet(stage * 5, sent.copy())
            result["callback_completed"]["packet"] += 1
            old.need(type(ret) is dict and set(ret) == {"raw", "rgb_bytes", "end", "stopped", "episode_id"},
                     "packet ABI")
            raw = np.asarray(ret["raw"])
            old.need(raw.ndim == 2 and raw.shape[1:] == (2,) and 1 <= len(raw) <= 5, "returned length")
            got = old.f32(raw, (len(raw), 2), True)
            old.need(ret["episode_id"] == session.episode_id and ret["end"] == stage * 5 + len(got)
                     and type(ret["stopped"]) is bool and got.tobytes() == sent[:len(got)].tobytes()
                     and (len(got) == 5 or ret["stopped"]), "actual packet ACK")
            result["packets"].append(dict(start=stage * 5, end=ret["end"], raw=got.tolist(),
                                          stopped=ret["stopped"]))
            store.event(dict(kind="packet_return", stage=stage, end=ret["end"], stopped=ret["stopped"]))
            if ret["stopped"]:
                result["status"] = "native_stopped"
                break
            rgb = ret["rgb_bytes"]
            old.need(type(rgb) is bytes and len(rgb) > 0, "actual RGB")
            rgb_sha = hashlib.sha256(rgb).hexdigest()
            store.event(dict(kind="encode_attempt", stage=stage, end=ret["end"], rgb_sha256=rgb_sha))
            result["callback_attempted"]["encode"] += 1
            enc = session.encode(ret["end"], rgb)
            result["callback_completed"]["encode"] += 1
            old.need(type(enc) is dict and set(enc) == {"latent", "input_sha256", "end", "episode_id"}
                     and enc["episode_id"] == session.episode_id and enc["end"] == ret["end"]
                     and enc["input_sha256"] == rgb_sha, "actual encoder binding")
            context = old.join(context, previous, sent, got, enc["latent"], mean, std)
            previous = got.copy()
            store.event(dict(kind="encode_return", stage=stage, end=ret["end"], rgb_sha256=rgb_sha,
                             context_sha256=core.digest(context)))
        else:
            result["status"] = "callback_complete_unverified"
    except Exception as exc:
        result["status"], result["error"] = "failed", type(exc).__name__ + ": " + str(exc)
        if ledger is not None and len(result["decisions"]) <= stage:
            result["failed_decision_receipt"] = ledger.receipt()
    finally:
        store.event(dict(kind="slot_final", result=deepcopy(result)))
    return result


def execute_roster(roots, session_factory, backend_factory, store, *, permit_ram=False):
    old.need(type(roots) is list and len(roots) == 2, "two fixed roots")
    rows = [dict(slot=s, root=r, policy=p, status="not_started", result=None,
                 error=None, cleanup="not_required") for s, r, p in ROSTER]
    store.event(dict(kind="fixed_denominator", rows=deepcopy(rows)))
    for row in rows:
        store.event(dict(kind="slot_attempt", slot=row["slot"], root=row["root"], policy=row["policy"]))
        close = None
        try:
            session, raw, close = session_factory(row["slot"], row["root"], row["policy"])
            old.need(callable(close), "cleanup callback")
            row["cleanup"] = "pending"
            result = execute_slot(roots[row["root"]], row["policy"], raw, session,
                                  backend_factory, store, permit_ram=permit_ram)
            row["result"], row["status"] = result, result["status"]
        except Exception as exc:
            row["status"], row["error"] = "setup_failed", type(exc).__name__ + ": " + str(exc)
        finally:
            if callable(close):
                try:
                    close()
                    row["cleanup"] = "returned"
                except Exception as exc:
                    row["cleanup"], row["status"] = "failed", "cleanup_failed"
                    row["error"] = type(exc).__name__ + ": " + str(exc)
        store.event(dict(kind="slot_return", row=deepcopy(row)))
    return dict(schema=NAME + ".roster", contract=contract(), fixed_denominator=6, rows=rows,
                source_authenticated=False, physical_certified=False, scientific_pass=False)


def compare(a, b):
    old.need(a["contract"] == b["contract"] == contract() and a["root_sha256"] == b["root_sha256"]
             and (a["policy"], b["policy"]) == ("signal", "reverse_signal"), "pair identity")
    if any(x["status"] != "callback_complete_unverified" for x in (a, b)):
        return dict(per_decision=None, reason="partial or failed", scientific_pass=False)
    for slot in (a, b):
        old.need(len(slot["decisions"]) == len(slot["packets"]) == 4 and
                 slot["callback_attempted"] == slot["callback_completed"] == {"packet": 4, "encode": 4},
                 "fixed denominator and callbacks")
        for stage, block in enumerate(slot["decisions"]):
            receipt = block["resource_receipt"]
            core.audit_ledger(receipt)
            old.need(block["stage"] == stage and receipt["caps"] == receipt["attempted"] ==
                     receipt["completed"] == expected() and
                     all(e["scope"].get("stage") == stage for e in receipt["events"]) and
                     block["selected_horizon"] == choose(slot["policy"], block["decision"]) and
                     [p["horizon"] for p in block["panels"]] == [1, 3] and
                     slot["packets"][stage]["start"] == 5 * stage and
                     slot["packets"][stage]["end"] == 5 * (stage + 1) and
                     slot["packets"][stage]["stopped"] is False and
                     slot["packets"][stage]["raw"] ==
                     block["panels"][0 if block["selected_horizon"] == 1 else 1]["winner_actions"][:5],
                     "decision accounting")
    def shapes(block):
        return Counter(core.canonical([e["resource"], e["shape"], e["directions"], e["status"]])
                       for e in block["resource_receipt"]["events"])
    return dict(per_decision=[dict(step=5 * i, measured_vector_match=True,
                                    hook_shape_multiset_match=shapes(a["decisions"][i]) == shapes(b["decisions"][i]))
                              for i in range(4)], physical_meter_match=None, equal_FLOPs=False,
                source_authenticated=False, physical_certified=False, scientific_pass=False)


def build_prepare():
    """Pin known-root source closure and this implementation; never load a model."""
    from experiments.spectral_reachability import p3_feedback_online_binding_20261030_v1 as binding
    binding.verify_inputs()
    old_prepare = binding.host.verify(binding.PREPARED, binding.budget.PREPARE_SHA)
    return dict(schema=NAME + ".prepare", contract=contract(), prior_prepare=old_prepare,
                prior_prepare_sha256=binding.budget.PREPARE_SHA, source_sha256=binding.budget.sha(__file__),
                binding_source_sha256=binding.budget.sha(binding.__file__),
                host_runner_ready=False, full_saved_strict=False, scientific_pass=False,
                external_prerun_timestamp_verified=False)


def prepare(output):
    from experiments.spectral_reachability import p3_matched_order_feedback_host_v2_codex_20261029 as host
    path = Path(output)
    old.need(path.is_absolute() and path.parent == host.ROOT / "results" and path.resolve() == path
             and path.name.startswith(NAME + "_prepare_") and path.suffix == ".json"
             and not path.exists() and not path.is_symlink(), "exclusive prepare output")
    return host.write_new(path, build_prepare())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-output", required=True, help="new absolute results path; CPU source checks only")
    print(prepare(parser.parse_args().prepare_output))
