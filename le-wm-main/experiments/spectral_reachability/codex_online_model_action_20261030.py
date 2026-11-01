"""Online callback interface with source checks and journaled budget reservations.

No planner, model, simulator, GPU, or production driver is imported here.
Callback claims and counters are not hardware or execution attestation.
"""
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

HERE = Path(__file__).resolve()
JOURNAL = HERE.with_name("codex_paired_durable_journal_20261030.py")
JOURNAL_SHA = "79546cca63a80b6c63ec913cb79d64ea29f77af3fd2c9947938a87aacdfa86db"


def need(value, reason):
    if not value:
        raise ValueError(reason)


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load_journal():
    need(sha(JOURNAL.read_bytes()) == JOURNAL_SHA, "journal source drift")
    spec = importlib.util.spec_from_file_location("online_journal_pinned", JOURNAL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_sources(sources, identity):
    base = load_journal().load_base()
    need(set(sources) == set(base.IDENTITY), "incomplete source closure")
    for key in base.IDENTITY:
        record = sources[key]
        need(set(record) == {"path", "sha256"} and record["sha256"] == identity[key], "source identity mismatch")
        need(sha(Path(record["path"]).read_bytes()) == record["sha256"], "source drift: " + key)


def validate_context(context):
    need(isinstance(context, dict) and set(context) == {"layout", "context_length", "latent_dim", "action_dim",
                                                       "latent_history", "action_history"}
         and context["layout"] == "states_with_between_state_actions_v1"
         and context["context_length"] == 3 and context["latent_dim"] == 192 and context["action_dim"] == 10,
         "model context ABI")
    for key, n, width in (("latent_history", 3, 192), ("action_history", 2, 10)):
        rows = context[key]
        need(isinstance(rows, list) and len(rows) == n and all(
            isinstance(row, list) and len(row) == width and all(
                type(x) in (int, float) and math.isfinite(x) for x in row) for row in rows),
            "nonfinite or wrong context shape")


class Meter:
    """Calls are reserved before entry; callback failures leave pending attempts."""
    def __init__(self, journal, arm, decision_id):
        self.journal, self.arm, self.decision_id, self.n = journal, arm, decision_id, 0
        self.events = []

    def run(self, label, amounts, callback, *args):
        need(isinstance(label, str) and label and callable(callback), "invalid metered call")
        base = load_journal().load_base()
        need(isinstance(amounts, dict) and amounts and set(amounts) <= set(base.AXES) - {"physical_steps"}
             and all(type(v) is int and v > 0 for v in amounts.values()), "invalid budget charge")
        state = self.journal.protocol.state(self.arm)
        need(all(state["attempted"][k] + v <= self.journal.config["caps"][self.arm][k]
                 for k, v in amounts.items()), "budget exhausted before callback")
        operation_id = f"{self.decision_id}:{self.n}:{label}"
        self.n += 1
        self.journal.call("record_resource_attempt", self.arm, operation_id, amounts)
        event = {"operation_id": operation_id, "label": label, "amounts": copy.deepcopy(amounts), "returned": False}
        self.events.append(event)
        value = callback(*args)
        self.journal.call("record_resource_return", self.arm, operation_id)
        event["returned"] = True
        return value


def validate_plan(plan, request_sha, ctx_sha, packet_size):
    need(isinstance(plan, dict) and set(plan) == {"mode", "request_sha256", "context_sha256", "candidates", "selected_id"}
         and plan["mode"] == "online_callback" and plan["request_sha256"] == request_sha
         and plan["context_sha256"] == ctx_sha, "plan provenance or mode")
    candidates = plan["candidates"]
    need(isinstance(candidates, list) and candidates and all(
        isinstance(c, dict) and set(c) == {"id", "cost", "commands"} and isinstance(c["id"], str) and c["id"]
        and type(c["cost"]) in (int, float) and math.isfinite(c["cost"]) for c in candidates)
        and len({c["id"] for c in candidates}) == len(candidates), "invalid candidate census")
    base = load_journal().load_base()
    for candidate in candidates:
        base.packet(candidate["commands"], packet_size)
    best = min(candidates, key=lambda c: c["cost"])
    need(plan["selected_id"] == best["id"], "winner is not stable minimum")
    return copy.deepcopy(best["commands"])


def online_select(journal, arm, frames, sources, method_source, encoder, planner, receipt_path):
    """One live callback decision; fail closed and retain pending work on errors.

    Encoder(frame_bytes, raw_history) produces native context.
    Planner(request, context, meter) must call meter.run for each instrumented
    model primitive and return a complete online_callback candidate record.
    Neither callback may receive evaluator physical state/reward here.
    """
    need(not journal.poisoned and not journal.closed, "journal unavailable")
    need(journal.config["packet_steps"] == 5, "native action packet ABI")
    s = journal.protocol.state(arm)
    need(s["phase"] == "observe" and s["step"] < journal.config["horizon"], "not an observation boundary")
    check_sources(sources, journal.protocol.identity)
    need(set(method_source) == {"path", "sha256"} and sha(Path(method_source["path"]).read_bytes()) == method_source["sha256"],
         "method source drift")
    need(isinstance(frames, list) and len(frames) == 3 and all(isinstance(f, bytes) for f in frames)
         and [sha(f) for f in frames] == s["frames"], "own observed RGB bytes mismatch")
    request = journal.call("observe", arm, s["step"], copy.deepcopy(s["frames"]), copy.deepcopy(s["actions"]))
    request_sha = sha(encode(request))
    decision_id = sha(encode({"request": request, "method_sha256": method_source["sha256"]}))
    meter = Meter(journal, arm, decision_id)
    try:
        ctx = meter.run("visual_encoder", {"visual_encoder": 1}, encoder, frames, copy.deepcopy(request["actions"]))
        validate_context(ctx)
        ctx_sha = sha(encode(ctx))
        plan = planner(copy.deepcopy(request), copy.deepcopy(ctx), meter)
        need(len(meter.events) > 1 and all(e["returned"] for e in meter.events), "planning requires metered returned work")
        commands = validate_plan(plan, request_sha, ctx_sha, journal.config["packet_steps"])
        check_sources(sources, journal.protocol.identity)
        need(sha(Path(method_source["path"]).read_bytes()) == method_source["sha256"], "method source drift")
        receipt = {"schema": "codex_online_model_action_20261030_v1", "request": request,
                   "request_sha256": request_sha, "context": ctx, "context_sha256": ctx_sha,
                   "plan": plan, "metered_events": meter.events, "sources": copy.deepcopy(sources),
                   "method_source": copy.deepcopy(method_source), "decision_id": decision_id,
                   "source_sha256": sha(HERE.read_bytes()), "journal_source_sha256": JOURNAL_SHA,
                   "claims": {"actual_execution_verified": False, "production_authorized": False,
                              "equal_total_compute": False, "flops": None}}
        raw = json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
        receipt_sha = sha(raw)
        with Path(receipt_path).open("xb") as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        load_journal().sync_dir(Path(receipt_path).parent)
        token = journal.call("select", arm, request, commands, receipt_sha)
        return {"selection_sha256": token, "receipt_sha256": receipt_sha}
    except BaseException:
        if not journal.poisoned and not journal.closed and journal.protocol.state(arm)["phase"] not in ("stopped", "complete"):
            journal.call("stop", arm, "online_selection_failed")
        raise


def consume_online(journal_path, journal_sha256, receipt_paths_and_hashes):
    """Saved-only linkage; never reruns callbacks or certifies real hardware work."""
    j = load_journal()
    result = j.consume(journal_path, journal_sha256)
    events = [json.loads(line, object_pairs_hook=j.no_duplicates) for line in Path(journal_path).read_bytes().splitlines()]
    begins = [e["payload"] for e in events if e["kind"] == "begin"]
    selections = [(i, e) for i, e in enumerate(begins) if e["operation"] == "select"]
    need(len(selections) == len(receipt_paths_and_hashes), "selection receipt census")
    used = set()
    for (selection_index, selection), (path, expected) in zip(selections, receipt_paths_and_hashes):
        raw = Path(path).read_bytes()
        need(sha(raw) == expected and expected not in used, "receipt hash or duplicate")
        used.add(expected)
        r = json.loads(raw, object_pairs_hook=j.no_duplicates)
        need(set(r) == {"schema", "request", "request_sha256", "context", "context_sha256", "plan",
                        "metered_events", "sources", "method_source", "decision_id", "source_sha256",
                        "journal_source_sha256", "claims"}
             and r["schema"] == "codex_online_model_action_20261030_v1"
             and r["source_sha256"] == sha(HERE.read_bytes()) and r["journal_source_sha256"] == JOURNAL_SHA,
             "online receipt source")
        need(r["claims"] == {"actual_execution_verified": False, "production_authorized": False,
                              "equal_total_compute": False, "flops": None}, "invalid claims")
        request = r["request"]
        need(sha(encode(request)) == r["request_sha256"] and sha(encode(r["context"])) == r["context_sha256"],
             "context/request digest")
        validate_context(r["context"])
        check_sources(r["sources"], request["identity"])
        need(sha(Path(r["method_source"]["path"]).read_bytes()) == r["method_source"]["sha256"], "method source drift")
        need(r["decision_id"] == sha(encode({"request": request, "method_sha256": r["method_source"]["sha256"]})),
             "decision identity")
        commands = validate_plan(r["plan"], r["request_sha256"], r["context_sha256"], 5)
        need(selection["args"] == [request["arm"], request, commands, expected] and not selection["kwargs"], "journal selection binding")
        metered = r["metered_events"]
        need(len(metered) > 1 and metered[0]["label"] == "visual_encoder" and
             metered[0]["amounts"] == {"visual_encoder": 1} and all(e["returned"] is True for e in metered),
             "incomplete metering")
        observations = [i for i, x in enumerate(begins) if x["operation"] == "observe"
                        and x["args"] == [request["arm"], request["step"], request["frames"], request["actions"]]
                        and not x["kwargs"]]
        need(len(observations) == 1 and observations[0] < selection_index, "journal observation binding")
        previous = observations[0]
        for index, e in enumerate(metered):
            need(set(e) == {"operation_id", "label", "amounts", "returned"}
                 and isinstance(e["label"], str) and e["label"]
                 and e["operation_id"] == f'{r["decision_id"]}:{index}:{e["label"]}',
                 "meter identity")
            attempts = [i for i, x in enumerate(begins) if x["operation"] == "record_resource_attempt"
                        and x["args"] == [request["arm"], e["operation_id"], e["amounts"]] and not x["kwargs"]]
            returns = [i for i, x in enumerate(begins) if x["operation"] == "record_resource_return"
                       and x["args"] == [request["arm"], e["operation_id"]] and not x["kwargs"]]
            need(len(attempts) == len(returns) == 1 and previous < attempts[0] < returns[0] < selection_index,
                 "journal budget binding or order")
            previous = returns[0]
    return {"schema": "codex_online_model_action_consume_20261030_v1", "journal": result,
            "receipt_count": len(selections), "callbacks_reexecuted": False,
            "actual_model_or_physical_verified": False, "equal_total_compute": False, "flops": None}
