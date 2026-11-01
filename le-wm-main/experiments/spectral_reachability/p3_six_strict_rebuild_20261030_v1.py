"""Source-bound six-slot saved arithmetic audit, not a full HOST certificate.

This module never starts a model or environment. It requires future explicit
source-binding journal records that existing six-slot runs do not provide.
"""
from copy import deepcopy
import io
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import p3_feedback_reverse_control_20261030_v1 as control
from experiments.spectral_reachability import p3_six_terminal_20261030_v1 as terminal
from experiments.spectral_reachability import p3_six_native_saved_20261030_v1 as native_saved
from experiments.spectral_reachability import p3_six_durable_transport_20261030_v1 as transport
from experiments.spectral_reachability import p3_six_numeric_bridge_20261030_v1 as bridge
from experiments.spectral_reachability import p3_six_saved_consumer_20261030_v1 as callback_saved
from experiments.spectral_reachability import p3_matched_order_feedback_host_saved_v2_codex_20261029 as old_saved

need, core, design = control.need, control.core, control.design
NAME = "p3_six_strict_rebuild_20261030_v1"


def source_material():
    """Reconstruct pinned old sources and extend source closure to this verifier."""
    prior = terminal.source_contract()
    source = old_saved.host.source
    prepared = source.host.verify_prepare(source.PREP, source.PREP_SHA)
    data = source.physical.source_material()
    need(len(prepared["tasks"]) == len(data["roots"]) == 2, "source known roots")
    roots = []
    for task, physical in zip(prepared["tasks"], data["roots"], strict=True):
        task = core.validate_task(task)
        need(task["binding"]["root_sha256"] == physical["root_sha256"], "source root/task identity")
        roots.append(dict(task_sha256=core.digest(task), root_sha256=physical["root_sha256"]))
    path = Path(__file__).resolve()
    need(path.is_file() and not path.is_symlink(), "strict verifier source")
    contract = dict(schema=NAME + ".source", previous=prior, verifier_sha256=terminal.file_sha(path),
                    roots=roots, expected_model_hash=prepared["expected_model_hash"],
                    precision=old_saved.host.prior.precision_contract(), scientific_pass=False)
    return contract, prepared, data


def check_binding(events, contract, expected):
    """Require exact source binding bracketing the six-slot lifecycle."""
    need(type(expected) is str and core.digest(contract) == expected, "pinned source contract")
    starts = [e for e in events if e["origin"] == "physical" and e["resource"] == "six_strict_source_begin"]
    ends = [e for e in events if e["origin"] == "physical" and e["resource"] == "six_strict_source_end"]
    den = [e for e in events if e["origin"] == "physical" and e["resource"] == "six_denominator"]
    returned = [e for e in events if e["origin"] == "physical" and e["resource"] == "six_slot_return"]
    need(len(starts) == len(ends) == len(den) == 1 and len(returned) == 6 and
         starts[0]["kind"] == ends[0]["kind"] == "record" and
         starts[0]["index"] == 0 and den[0]["index"] == 1 and
         den[0]["index"] < returned[-1]["index"] < ends[0]["index"] == len(events) - 1,
         "source binding boundaries")
    detail = dict(source_contract_sha256=expected, task_roots=contract["roots"],
                  expected_model_hash=contract["expected_model_hash"], precision=contract["precision"])
    need(starts[0]["detail"] == ends[0]["detail"] == detail, "source binding content")


def rebuild(root, warm_raw10, result, wrapped, observed):
    """Rebuild all contexts, decisions, candidate bank, Q and paid call bindings."""
    root = core.validate_task(root)
    need(result["root_sha256"] == core.digest(root), "root task binding")
    need(type(wrapped) is list and all(type(x) is dict and set(x) == {"kind", "slot", "call", "payload"} and
         x["kind"] == "arm_event" for x in wrapped), "numeric wrapper")
    values = [x["payload"] for x in wrapped]
    callback_saved.audit_slot(result, values)
    need(result["status"] == "callback_feedback_complete_not_certified" and
         len(result["blocks"]) == 2 and len(result["packets"]) == 4, "complete callback required")
    need(type(observed) is list and len(observed) == 4, "four observed latents")
    context = deepcopy(root["root_context"])
    latent = np.asarray(context["latent_history"], dtype=np.float32)
    action = np.asarray(context["action_history"], dtype=np.float32)
    warm = np.asarray(warm_raw10, dtype=np.float32)
    mean, std = np.asarray(root["action_mean"], np.float32), np.asarray(root["action_std"], np.float32)
    need(latent.shape == (3, 192) and action.shape == (2, 10) and warm.shape == (10, 2) and
         np.isfinite(warm).all() and (np.abs(warm) <= 1).all() and (std > 0).all() and
         ((warm - mean) / std).reshape(2, 10).astype(np.float32).tobytes() == action.tobytes(),
         "full warm10 history")
    compiled = [x for x in values if x.get("kind") == "block_compiled"]
    intents = [x for x in values if x.get("kind") == "packet_intent"]
    returns = [x for x in values if x.get("kind") == "packet_return"]
    actual = [x for x in values if x.get("kind") == "actual_context"]
    need(len(compiled) == 2 and len(intents) == len(returns) == len(actual) == 4, "complete callback events")
    pos = {id(x): i for i, x in enumerate(values)}
    contexts = [deepcopy(context)]
    for i, (packet, enc, intent, ret, event) in enumerate(
            zip(result["packets"], observed, intents, returns, actual, strict=True)):
        need(set(enc) == {"boundary", "latent", "rgb_sha256"} and enc["boundary"] == 5 * (i + 1) ==
             packet["end"] == event["end"] and enc["rgb_sha256"] == event["rgb_sha256"] and
             intent == dict(kind="packet_intent", stage=packet["stage"], start=packet["start"], raw=packet["raw"]) and
             ret == dict(kind="packet_return", **packet) and
             pos[id(intent)] < pos[id(ret)] < pos[id(event)] and
             (i == 3 or pos[id(event)] < pos[id(intents[i + 1])]), "packet context event binding")
        z = np.asarray(enc["latent"])
        raw = np.asarray(packet["raw"], dtype=np.float32)
        need(z.dtype == np.float32 and z.shape == (192,) and np.isfinite(z).all() and
             raw.shape == (5, 2) and np.isfinite(raw).all() and (np.abs(raw) <= 1).all(), "latent/raw ABI")
        latent = np.concatenate((latent[1:], z[None]), axis=0)
        action = np.stack((action[-1], ((raw - mean) / std).reshape(10).astype(np.float32)))
        context = dict(context, latent_history=latent.tolist(), action_history=action.tolist())
        need(core.digest(core.context(context)) == event["context_sha256"], "observed context digest")
        contexts.append(deepcopy(context))
    calls = old_saved._native_calls(wrapped, result)
    cursor, packet_cursor, remaining = 0, 0, [1, 3]
    for stage, block in enumerate(result["blocks"]):
        context = contexts[packet_cursor]
        task = design.parent.parent_task(root, context)
        need(block["stage"] == stage and core.digest(context) == block["context_sha256"] and
             compiled[stage] == dict(kind="block_compiled", stage=stage, context_sha256=block["context_sha256"],
                                     horizon=block["horizon"], actions=block["actions"]) and
             (stage == 0 or pos[id(actual[packet_cursor - 1])] < pos[id(compiled[stage])]) and
             pos[id(compiled[stage])] < pos[id(intents[packet_cursor])], "stage context and order")
        for h, response in zip((1, 3), block["responses"], strict=True):
            call = calls[cursor]; cursor += 1
            need(call[0]["method"] == "signal" and call[0]["task"] == task and
                 call[0]["request"] == core.request(task, dict(context=context),
                     operation="compile_finite_horizon_J", horizon=h) and
                 call[-1]["result"] == response, "paid signal")
        diagnostic = design.signal.spectra(task, block["responses"])
        need(diagnostic == block["diagnostic"], "signal spectra")
        node, signals, _ = design.signal.runtime_signal(task, block["responses"], diagnostic)
        decision = core.choose(task, node, signals, "adaptive")
        need(decision == block["decision"], "original decision")
        h = control.choose(result["policy"], decision, remaining, stage)
        need(h == block["horizon"], "policy intervention")
        remaining.remove(h)
        ledger = core.Ledger(design.expected()); ledger.scope = dict(phase="saved_recompute")
        bank = design.frozen.generate(task, block["responses"][0 if h == 1 else 1], "endpoints", ledger)
        need(bank == block["bank"] and len(block["candidates"]) == 4, "endpoint bank")
        for index, candidate in enumerate(block["candidates"]):
            need(candidate["index"] == index and candidate["actions"] == bank["actions"][index] and
                 candidate["whole"] == candidate["segmented"], "candidate identity")
            for segmented, path in ((False, candidate["whole"]), (True, candidate["segmented"])):
                call = calls[cursor]; cursor += 1
                need(call[0]["method"] == "replay" and call[0]["task"] == task and
                     call[0]["request"] == dict(parent_context=context, actions=candidate["actions"],
                                                segmented=segmented) and call[-1]["result"] == path, "paid replay")
            end = core.checked_path(task, context, candidate["actions"], candidate["whole"])
            need(candidate["cost"] == core.model_cost(task, end), "model Q")
        winner = min(block["candidates"], key=lambda c: (c["cost"], c["index"]))
        need(block["winner_index"] == winner["index"] and block["actions"] == winner["actions"], "winner")
        call = calls[cursor]; cursor += 1
        need(call[0]["method"] == "replay" and call[0]["task"] == task and
             call[0]["request"] == dict(parent_context=context, actions=winner["actions"], segmented=False) and
             call[-1]["result"] == winner["whole"], "independent winner replay")
        selected = result["packets"][packet_cursor:packet_cursor + h]
        need(len(selected) == h and all(p["stage"] == stage for p in selected) and
             [x for p in selected for x in p["raw"]] == block["actions"], "selected packet actions")
        packet_cursor += h
    need(cursor == len(calls) and packet_cursor == 4 and not remaining, "all calls and quotas")
    return dict(contexts=4, paid_native_calls=len(calls), policy=result["policy"],
                mathematical_rebuild=True, physical_certified=False, scientific_pass=False)


def consume(directory, report_sha256, completion_sha256, source_contract_sha256):
    """Require complete saved records and source-bound arithmetic; no production."""
    directory = Path(directory).absolute()
    prior = native_saved.consume(directory, report_sha256, completion_sha256)
    need(prior["journal_transport_complete"] and prior["trailing_bytes"] == 0 and
         not prior["artifacts"]["orphans"] and prior["callback_audit"] is not None and
         all(row["status"] == "callback_feedback_complete_not_certified" and
             row["cleanup"] == "returned" and row["error"] is None for row in prior["rows"]),
         "six complete records required")
    report = native_saved._pinned(directory / terminal.REPORT, report_sha256)
    need(report["store_poisoned"] is False, "unpoisoned storage")
    j = transport.inspect(directory / "events.jsonl", report["journal_sha256"], report["cap"])
    need(not j["pending"] and j["trailing_bytes"] == 0, "complete journal")
    events = j["events"]
    contract, prepared, data = source_material()
    check_binding(events, contract, source_contract_sha256)
    rows = prior["callback_audit"]["rows"]
    audits = []
    for row in rows:
        slot = row["slot"]
        root, source_root = prepared["tasks"][row["root"]], data["roots"][row["root"]]
        start = next(e["index"] for e in events if e["resource"] == "six_slot_attempt" and
                     e["origin"] == "physical" and e["detail"]["slot"] == slot)
        stop = next(e["index"] for e in events if e["resource"] == "six_slot_return" and
                    e["origin"] == "physical" and e["detail"]["slot"] == slot)
        section = [e for e in events if start < e["index"] < stop]
        warm = [e for e in section if e["origin"] == "physical" and e["resource"] == "native_fresh_observation" and
                e["detail"]["receipt"]["phase"] == "warmup"]
        need(len(warm) == 11, "warm observations")
        traces = {(x["receipt"]["step"], x["receipt"]["overlay_index"]): x["receipt"]
                  for x in source_root["warm_trace"]}
        for step, event in enumerate(warm):
            obs, reference = event["detail"]["receipt"], traces[(step, 0)]
            need(obs["step"] == step and all(obs[k] == reference[k]
                 for k in ("pixel_sha256", "snapshot", "state")), "warm source identity")
        guard = [e for e in section if e["origin"] == "physical" and
                 e["resource"] == "native_matched_root_accepted"]
        need(len(guard) == 1 and guard[0]["detail"]["root_sha256"] ==
             root["binding"]["root_sha256"], "root guard")
        selected = [e for e in section if e["origin"] == "numerical" and e["resource"] == bridge.RESOURCE and
                    e["detail"].get("slot") == slot]
        wrapped = bridge.reconstruct(section, slot)
        need(all(w.get("slot") == slot for w in wrapped), "numeric slot identity")
        positions, cursor = [], 0
        for value in wrapped:
            count = selected[cursor]["detail"]["total"]
            positions.append((selected[cursor]["index"], selected[cursor + count - 1]["index"]))
            cursor += count
        need(cursor == len(selected), "all numeric chunks accounted")
        observed = []
        encoded = [e for e in section if e["origin"] == "encoding" and e["resource"] == "encoding_complete"]
        need(len(encoded) == 4, "four encoder outputs")
        for e in encoded:
            detail = e["detail"]; boundary = detail["boundary"]
            desc = detail["arrays"]["latent"]
            raw = transport.base.read_bytes(directory / desc["file"], 4096 + 192 * 4)
            need(transport.base.fsha(raw) == desc["sha256"], "saved latent hash")
            z = np.load(io.BytesIO(raw), allow_pickle=False)
            observed.append(dict(boundary=boundary, latent=z, rgb_sha256=detail["frame"]["sha256"]))
            context_pos = [p for w, p in zip(wrapped, positions, strict=True)
                           if w["payload"].get("kind") == "actual_context" and
                           w["payload"].get("end") == boundary]
            return_pos = [p for w, p in zip(wrapped, positions, strict=True)
                          if w["payload"].get("kind") == "packet_return" and
                          w["payload"].get("end") == boundary]
            need(len(context_pos) == len(return_pos) == 1 and
                 return_pos[0][1] < e["index"] < context_pos[0][0], "returned frame context timeline")
        audits.append(rebuild(root, source_root["packets"]["off"]["raw_action_history"],
                              row["result"], wrapped, observed))
    final_contract, _, _ = source_material()
    need(final_contract == contract, "source closure stable")
    return dict(schema=NAME + ".consume", status="source_bound_saved_math_only",
                slots=audits, source_contract_sha256=source_contract_sha256,
                source_binding_is_attestation=False, full_saved_strict=False,
                physical_certified=False, scientific_pass=False,
                limitation="no device attestation or full host/resource certification")
