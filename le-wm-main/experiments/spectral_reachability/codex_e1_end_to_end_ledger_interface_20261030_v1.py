"""E1 resource-ledger structural preflight; never certifies cost parity."""
import hashlib
import json
import re
import sys

SCHEMA = "codex.e1.end_to_end_ledger.v1"
PHASES = ("proposal", "inherited", "controller", "replay", "encoder", "native")
AXES = ("predict_api_calls", "predictor_samples", "action_encoder_calls", "pred_proj_calls",
        "image_encoder_calls", "high_cost_calls", "high_predictor_calls", "high_input_rows",
        "low_predictor_calls", "cpu_proposal_draws", "ad_calls", "environment_construct",
        "reset", "public_step", "physics_step", "render", "storage_bytes", "wall_ns", "flops")
LIMIT = 1024 * 1024


def need(condition, text):
    if not condition:
        raise ValueError(text)


def fields(obj, names):
    need(type(obj) is dict and set(obj) == set(names), "missing or unexpected fields")


def digest(s):
    need(type(s) is str and re.fullmatch("[0-9a-f]{64}", s) is not None, "invalid sha256")


def inspect(ledger):
    """Check declared accounting shape. Hash references and completeness are unverified."""
    fields(ledger, ("schema", "scope_sha256", "arm", "task_sha256", "target_sha256", "phases"))
    need(ledger["schema"] == SCHEMA, "schema")
    for name in ("scope_sha256", "task_sha256", "target_sha256"):
        digest(ledger[name])
    need(ledger["arm"] in ("arbitrary_latent_waypoint", "existing_learned_prior",
                           "dynamics_compiled_witness", "bank_winner", "zero_extra", "witness_extra"),
         "unknown arm")
    phases = ledger["phases"]
    fields(phases, PHASES)
    blockers = []
    seen = set()
    for phase in PHASES:
        entry = phases[phase]
        fields(entry, ("scope", "receipt_sha256", "measurements"))
        need(entry["scope"] in ("per_arm", "shared_unallocated", "unknown"), "scope")
        if entry["receipt_sha256"] is not None:
            digest(entry["receipt_sha256"])
        else:
            blockers.append(phase + ":receipt_missing")
        if entry["scope"] != "per_arm":
            blockers.append(phase + ":not_attributed_per_arm")
        fields(entry["measurements"], AXES)
        for axis in AXES:
            v = entry["measurements"][axis]
            if v is None:
                blockers.append(phase + ":" + axis + ":unknown")
                continue
            fields(v, ("attempted", "completed", "event_set_sha256"))
            a, c = v["attempted"], v["completed"]
            need(type(a) is int and type(c) is int and 0 <= c <= a, "invalid attempted/completed")
            digest(v["event_set_sha256"])
            # A repeated claim of exactly the same event-set on the same axis
            # across phases would double charge (or obscure an overlap).
            key = (axis, v["event_set_sha256"])
            need(key not in seen, "duplicate event-set on same axis")
            seen.add(key)
            if entry["receipt_sha256"] is None:
                blockers.append(phase + ":" + axis + ":unbound_measurement")
    return {
        "schema": SCHEMA + ".report",
        "structural_status": "incomplete" if blockers else "complete_declarations_only",
        "blockers": blockers,
        "independent_evidence_verified": False,
        "coverage_verified": False,
        "equal_budget_certified": False,
        "scientific_admission": False,
        "ranking_authorized": False,
        "warning": "Counts and zeroes are unverified declarations; no FLOPs or nested axes are summed.",
    }


def parse(raw):
    need(len(raw) <= LIMIT, "input too large")
    def pairs(items):
        out = {}
        for k, v in items:
            need(k not in out, "duplicate key")
            out[k] = v
        return out
    def reject(_):
        raise ValueError("nonfinite JSON")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=reject)


def main():
    try:
        need(len(sys.argv) == 2, "expected input SHA256 argument")
        digest(sys.argv[1])
        raw = sys.stdin.buffer.read(LIMIT + 1)
        need(len(raw) <= LIMIT and hashlib.sha256(raw).hexdigest() == sys.argv[1], "input hash or size")
        report = inspect(parse(raw))
    except (ValueError, KeyError, TypeError, UnicodeError, RecursionError) as exc:
        print(json.dumps({"status": "blocked_invalid", "reason": str(exc),
                          "equal_budget_certified": False, "ranking_authorized": False}),
              file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True))
    return 0 if report["structural_status"] == "complete_declarations_only" else 3


if __name__ == "__main__":
    raise SystemExit(main())
