"""Independent saved-journal AP audit; never loads a model or runs physics."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import zlib

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "results/pc_ap_durable_host_20261102_codexexclusive_RUN_01"
REFERENCE = ROOT / "results/pc_ap_official_kernel_physical_dev_20261102_v1/report.json"
REF_SHA = "f2c95637b9dda338bf7ed4165934ac384c4c78308cc0aa932ade776fd04acd91"
ARMS = ("ap_observed_target", "final_goal")
SEEDS = (20263160, 20263161)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_chain(path):
    raw = path.read_bytes()
    require(bool(raw) and raw.endswith(b"\n"), "empty or truncated journal")
    previous = None
    events = []
    for index, line in enumerate(raw.splitlines()):
        obj = json.loads(line)
        require(set(obj) == {"index", "previous_sha256", "payload", "event_sha256"}, "event schema")
        require(obj["index"] == index and obj["previous_sha256"] == previous, "chain sequence")
        expected = sha(canonical({k: obj[k] for k in ("index", "previous_sha256", "payload")}))
        require(obj["event_sha256"] == expected, "event hash")
        previous = expected
        events.append(obj["payload"])
    return events, {"file_sha256": sha(raw), "events": len(events), "head": previous}


def image(obj):
    shape = obj["shape"]
    require(len(shape) == 3 and all(type(n) is int and 0 < n <= 4096 for n in shape)
            and shape[2] == 3 and math.prod(shape) <= 32_000_000, "RGB shape")
    packed = base64.b64decode(obj["uint8_zlib_base64"], validate=True)
    decoder = zlib.decompressobj()
    raw = decoder.decompress(packed, math.prod(shape) + 1)
    require(decoder.eof and not decoder.unused_data and len(raw) == math.prod(shape), "RGB encoding")
    require(sha(raw) == obj["sha256"], "RGB digest")
    return raw


def check_context(c):
    require(len(c["frames"]) == 3 and len(c["history_action_float32_hex"]) == 3, "context history size")
    for x in c["frames"] + [c["root"], c["goal"]]:
        image(x)
    require(image(c["execution_snapshot"]["rendered_goal_rgb"]) == image(c["goal"]), "snapshot goal mismatch")
    for action in c["history_action_float32_hex"]:
        raw = bytes.fromhex(action)
        require(len(raw) == 40 and np.isfinite(np.frombuffer(raw, dtype="<f4")).all(), "history action bytes")
    return sha(canonical(c))


class Cursor:
    def __init__(self, events):
        self.events, self.pos = events, 0

    def take(self, kind, decision=None, sub=None):
        require(self.pos < len(self.events), f"missing {kind}")
        event = self.events[self.pos]
        self.pos += 1
        require(event.get("kind") == kind, f"expected {kind} at {self.pos-1}, got {event.get('kind')}")
        if decision is not None:
            require(event.get("decision") == decision, "decision sequence")
        if sub is not None:
            require(event["event"].get("kind") == sub, f"expected bridge {sub}")
            return event["event"]
        return event


def cost_summary(events):
    """Retain observed costs and failure markers even when the strict audit fails."""
    counts = {"predictor_requested_blocks": 0, "predictor_call_intents": 0,
              "predictor_returned_calls": 0, "predictor_failed_calls": 0,
              "step_intents": 0, "step_returned": 0, "evaluator_distances": 0}
    failures = []
    for i, e in enumerate(events):
        kind = e.get("kind")
        if kind == "predictor_call_intent":
            counts["predictor_call_intents"] += 1
            counts["predictor_requested_blocks"] += e.get("requested_blocks", 0)
        if kind in ("predictor_call_returned", "predictor_call_failed"):
            counts["predictor_returned_calls" if kind.endswith("returned") else "predictor_failed_calls"] += 1
        if kind == "evaluator_distance":
            counts["evaluator_distances"] += 1
        if kind == "bridge_event":
            sub = e.get("event", {})
            if sub.get("kind") in ("step_intent", "step_returned"):
                counts["step_intents" if sub["kind"] == "step_intent" else "step_returned"] += 1
            if sub.get("kind") in ("step_failed", "render_failed", "block_stopped"):
                failures.append({"index": i, "event": sub})
        if kind in ("host_failed", "host_arm_failed", "arm_failed", "arm_stopped", "block_incomplete"):
            failures.append({"index": i, "event": e})
    return {"observed": counts, "failures_and_stops": failures}


def audit_arm(events, seed, arm, expected_root, normalization=None):
    c = Cursor(events)
    warm = c.take("warmup_intent")
    require((warm["seed"], warm["arm"], warm["shape"]) == (seed, arm, [3, 5, 2]), "warmup identity")
    warm_actions = bytes.fromhex(warm["actions_float32_hex"])
    require(len(warm_actions) == 120 and np.isfinite(np.frombuffer(warm_actions, dtype="<f4")).all(), "warmup actions")
    opened = c.take("warmup_returned")
    require((opened["root_rgb_sha256"], opened["goal_rgb_sha256"]) == expected_root, "reference root/goal")
    start = c.take("arm_started")
    require((start["seed"], start["arm"], start["decisions"], start["action_block"],
             start["population"], start["iterations"], start["elites"]) == (seed, arm, 2, 5, 16, 2, 4), "arm contract")
    prior = None
    first_context = None
    timings, distances = [], []
    for decision in range(2):
        plan = c.take("planning_intent", decision)
        require(plan["seed"] == seed + 8100 + decision, "search seed")
        blocks = batches = 0
        for requested in (16, 16, 2):
            intent = c.take("predictor_call_intent")
            require((intent["blocks_before"], intent["batches_before"], intent["requested_blocks"])
                    == (blocks, batches, requested), "predictor intent counts")
            blocks, batches = blocks + requested, batches + 1
            returned = c.take("predictor_call_returned")
            require((returned["blocks"], returned["batches"]) == (blocks, batches), "predictor actual counts")
        audit = c.take("planning_returned", decision)["audit"]
        require((audit["prediction_samples"], audit["prediction_batches"], audit["search_seed"])
                == (34, 3, seed + 8100 + decision), "plan counts")
        elapsed = audit["planning_seconds_including_bridge"]
        require(isinstance(elapsed, (int, float)) and math.isfinite(elapsed) and elapsed >= 0, "invalid planning time")
        timings.append(elapsed)
        action = np.asarray(audit["raw_actions"], dtype="<f4")
        require(action.shape == (5, 2) and np.isfinite(action).all(), "planned action shape")
        normalized = np.asarray(audit["normalized_actions"], dtype="<f4")
        require(normalized.shape == (5, 2) and np.isfinite(normalized).all(), "normalized action shape")
        if normalization is not None:
            mean, std = normalization
            require(np.allclose(action.reshape(-1), normalized.reshape(-1) * std + mean, rtol=0, atol=2e-6),
                    "action normalization binding")
        action_bytes = action.tobytes()
        bound = c.take("context_bound", decision)
        context = bound["context"]
        require(check_context(context) == bound["context_sha256"], "bound context digest")
        if prior is None:
            first_context = context
            require(b"".join(bytes.fromhex(x) for x in context["history_action_float32_hex"]) == warm_actions,
                    "warmup/history mismatch")
            require((context["root"]["sha256"], context["goal"]["sha256"]) == expected_root, "context root/goal")
        else:
            require(context == prior, "previous committed context mismatch")
        require(plan["root_rgb_sha256"] == context["root"]["sha256"], "planner root mismatch")
        opened_block = c.take("bridge_event", decision, "block_open")
        require(opened_block["planned_action_sha256"] == sha(action_bytes), "planned action digest")
        require((opened_block["root_rgb_sha256"], opened_block["goal_rgb_sha256"])
                == (context["root"]["sha256"], context["goal"]["sha256"]), "block root/goal")
        records = []
        for step in range(1, 6):
            intent = c.take("bridge_event", decision, "step_intent")
            expected_action = action_bytes[(step-1)*8:step*8].hex()
            require((intent["step"], intent["action_float32_hex"]) == (step, expected_action), "step intent action")
            distance = c.take("evaluator_distance", decision)
            require(distance["step"] == step and math.isfinite(distance["qpos_distance"]), "evaluator distance")
            distances.append(distance["qpos_distance"])
            returned = c.take("bridge_event", decision, "step_returned")
            require((returned["step"], returned["action_float32_hex"]) == (step, expected_action), "step returned action")
            require(type(returned["terminated"]) is bool and type(returned["truncated"]) is bool
                    and not returned["terminated"] and not returned["truncated"]
                    and math.isfinite(returned["reward"]), "stopped or invalid step")
            records.append({k: returned[k] for k in ("step", "action_float32_hex", "reward", "terminated", "truncated")})
        require(c.take("bridge_event", decision, "render_intent")["after_returned_steps"] == 5, "render order")
        feedback = c.take("bridge_event", decision, "feedback_ready")
        committed = c.take("context_committed", decision)
        next_context, receipt = committed["context"], committed["receipt"]
        require(committed["previous_context_sha256"] == bound["context_sha256"]
                and check_context(next_context) == committed["context_sha256"], "committed context digest")
        require(receipt["status"] == "complete" and receipt["failure"] is None
                and receipt["attempted"] == receipt["returned"] == 5 and receipt["steps"] == records, "receipt steps")
        require(receipt["planned_action_sha256"] == feedback["executed_action_sha256"] == sha(action_bytes)
                and receipt["root_rgb_sha256"] == feedback["root_rgb_sha256"] == context["root"]["sha256"]
                and receipt["goal_rgb_sha256"] == context["goal"]["sha256"]
                and receipt["successor_rgb_sha256"] == feedback["successor_rgb_sha256"]
                == next_context["root"]["sha256"], "receipt digest binding")
        require(next_context["frames"] == context["frames"][1:] + [context["root"]]
                and next_context["goal"] == context["goal"]
                and next_context["history_action_float32_hex"]
                == context["history_action_float32_hex"][1:] + [action_bytes.hex()], "history handoff")
        prior = next_context
    require(c.take("arm_completed")["decisions"] == 2 and c.pos == len(events), "arm completion/trailing events")
    return {"status": "complete", "root": expected_root, "initial_context_sha256": sha(canonical(first_context)),
            "warmup_actions_sha256": sha(warm_actions), "planning_seconds": timings,
            "qpos_distances_after_steps": distances}


def audit_host(events, reference):
    c = Cursor(events)
    start = c.take("host_started")
    require(start["reference_sha256"] == REF_SHA, "host reference")
    for episode in ("0", "2", "3"):
        require(start["memory"][episode]["rgb_sha256"] == reference["memory_rgb_sha256"][episode], "memory binding")
    norm = c.take("normalization_validated")
    mean, std = np.asarray(norm["mean"]), np.asarray(norm["std"])
    require(mean.shape == std.shape == (10,) and np.isfinite(mean).all()
            and np.isfinite(std).all() and (std > 0).all(), "normalization")
    c.take("model_load_intent")
    loaded = c.take("model_loaded")
    require(loaded["parameter_sha256"] == reference["parameter_hash_before"], "model hash")
    for episode in ("0", "2", "3"):
        for kind in ("memory_encode_intent", "memory_encode_returned"):
            event = c.take(kind)
            require((event["episode"], event["frames"]) == (episode, 31), "memory encode count")
    for seed in SEEDS:
        for arm in ARMS:
            event = c.take("arm_returned")
            require((event["seed"], event["arm"], event["result"]["status"],
                     event["result"]["decisions"]) == (seed, arm, "complete", 2), "host arm completion")
    done = c.take("host_completed")
    require(done["parameter_sha256"] == loaded["parameter_sha256"] and c.pos == len(events), "host completion")
    return {"status": "complete", "memory_frames_returned": 93, "parameter_sha256": loaded["parameter_sha256"],
            "mean": mean.tolist(), "std": std.tolist()}


def audit(directory, reference_path):
    result = {"scope": "saved journal consistency only; known development roots; not independent confirmation",
              "journals": {}, "errors": []}
    try:
        reference_bytes = reference_path.read_bytes()
        require(sha(reference_bytes) == REF_SHA, "reference SHA mismatch")
        reference = json.loads(reference_bytes)
    except Exception as exc:
        result["errors"].append(f"reference: {type(exc).__name__}: {exc}")
        result["status"] = "incomplete_or_invalid"
        return result
    for name in ["host"] + [f"{seed}_{arm}" for seed in SEEDS for arm in ARMS]:
        entry = result["journals"][name] = {}
        try:
            events, chain = read_chain(directory / f"{name}.jsonl")
            entry.update(chain)
            entry.update(cost_summary(events))
            if name == "host":
                entry["verified"] = audit_host(events, reference)
            else:
                seed_text, arm = name.split("_", 1)
                seed = int(seed_text)
                rows = [r for r in reference["rows"] if r["seed"] == seed and r["status"] == "complete"]
                identities = {(r["root_rgb_sha256"], r["goal_rgb_sha256"]) for r in rows}
                require(len(rows) == 2 and len(identities) == 1, "reference pair")
                host_verified = result["journals"]["host"].get("verified")
                require(host_verified is not None, "host normalization not verified")
                norm = (np.asarray(host_verified["mean"]), np.asarray(host_verified["std"]))
                entry["verified"] = audit_arm(events, seed, arm, next(iter(identities)), norm)
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
            result["errors"].append(f"{name}: {entry['error']}")
            if "observed" not in entry:
                # Preserve parseable failure/cost evidence, explicitly untrusted.
                try:
                    raw = (directory / f"{name}.jsonl").read_bytes()
                    prefix = []
                    for line in raw.splitlines():
                        try:
                            prefix.append(json.loads(line)["payload"])
                        except Exception:
                            break
                    entry["unverified_parseable_prefix"] = cost_summary(prefix)
                    entry["unverified_file_sha256"] = sha(raw)
                except Exception as salvage_exc:
                    entry["salvage_error"] = str(salvage_exc)
    for seed in SEEDS:
        a, b = (result["journals"][f"{seed}_{arm}"].get("verified") for arm in ARMS)
        if a and b:
            same = a["initial_context_sha256"] == b["initial_context_sha256"]
            result.setdefault("paired_initial_context_equal", {})[str(seed)] = same
            if not same:
                result["errors"].append(f"{seed}: paired initial context differs")
    result["status"] = "verified_saved_consistency" if not result["errors"] else "incomplete_or_invalid"
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=INPUT)
    p.add_argument("--reference", type=Path, default=REFERENCE)
    p.add_argument("--output", type=Path, required=True, help="new exclusive audit JSON")
    args = p.parse_args()
    result = audit(args.input, args.reference)
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2, sort_keys=True, allow_nan=False)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())
    print(json.dumps({"status": result.get("status", "incomplete_or_invalid"), "errors": result["errors"]}))
    if result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
