"""Mandatory checkpoint guard for this new SRA entrypoint, not a global patch.

No model loading, CUDA allocation or physical dispatch. Guard checks reserve
setup and declared wall budget in the durable journal before entering.
"""
import json
import time
from pathlib import Path

from experiments.spectral_reachability import codex_online_model_action_20261030 as online
from experiments.spectral_reachability import codex_sra_checkpoint_identity_20261030 as identity_guard
from experiments.spectral_reachability import codex_sra_online_run_bank_20261030 as sra
from experiments.spectral_reachability import codex_native_runtime_meter_20261030 as native

HERE = Path(__file__).resolve()
ARM = "sra_no_composition"
PHASES = ("before_encoder", "after_encoder", "before_planner", "after_planner")


def need(value, reason):
    if not value:
        raise ValueError(reason)


def _reserve(journal, arm, operation, amounts):
    state = journal.protocol.state(arm)
    need(all(state["attempted"][key] + value <= journal.config["caps"][arm][key]
             for key, value in amounts.items()), "guard budget exhausted")
    journal.call("record_resource_attempt", arm, operation, amounts)


def _identity_check(model, replay, identity, proof_path, proof_sha, manifest_path, manifest_sha, fixture):
    manifest = sra.checked_manifest(manifest_path, manifest_sha)
    proof = identity_guard.verify_proof(proof_path, proof_sha, model, replay, identity, allow_fixture=fixture)
    need(proof["fixture"] is fixture, "fixture proof mode mismatch")
    cp = proof["checkpoint"]
    required = {
        "checkpoint_identity_proof": (proof_path, proof_sha),
        "checkpoint_loader": (identity_guard.LOADER, identity_guard.LOADER_SHA),
        "checkpoint_weights": (cp["weights_path"], cp["weights_sha256"]),
        "checkpoint_config": (cp["config_path"], cp["config_sha256"]),
        "checkpoint_guard": (identity_guard.HERE, sra.file_sha(identity_guard.HERE)),
        "guarded_sra_entry": (HERE, sra.file_sha(HERE)),
        "pusht_transform_source": (HERE.parents[1] / "run_pusht_counterfactual.py",
                                   sra.file_sha(HERE.parents[1] / "run_pusht_counterfactual.py")),
    }
    for name, (path, digest) in required.items():
        ref = manifest["files"].get(name)
        need(ref is not None and Path(ref["path"]).resolve() == Path(path).resolve()
             and ref["sha256"] == digest, "missing guarded source: " + name)


def _select(*, journal, frames, sources, method_source, model, device, replay,
            frame_shape, proof_path, proof_sha256, target_path, audit_path, receipt_path,
            guard_receipt_path, check_wall_ms, fixture_encoder=None, fixture_compile=None,
            fixture_replay=False):
    fixture = fixture_encoder is not None
    need((fixture and fixture_compile is not None and fixture_replay) or
         (not fixture and fixture_compile is None and not fixture_replay), "incomplete fixture mode")
    need(type(check_wall_ms) is int and check_wall_ms > 0, "positive per-check wall reservation required")
    need(not journal.poisoned and not journal.closed, "journal unavailable")
    state = journal.protocol.state(ARM)
    request = {"arm": ARM, "step": state["step"], "identity": journal.protocol.identity,
               "frames": state["frames"], "actions": state["actions"]}
    decision = online.sha(online.encode({"request": request, "method_sha256": method_source["sha256"]}))
    phases = []

    def check(label):
        need(label == PHASES[len(phases)], "guard order")
        operation = decision + ":checkpoint_guard:" + label
        amounts = {"setup_steps": 1, "wall_ms": check_wall_ms}
        _reserve(journal, ARM, operation, amounts)
        started = time.monotonic_ns()
        _identity_check(model, replay, journal.protocol.identity, proof_path, proof_sha256,
                        method_source["path"], method_source["sha256"], fixture)
        elapsed_ns = time.monotonic_ns() - started
        need(elapsed_ns <= check_wall_ms * 1_000_000, "guard wall reservation exceeded")
        journal.call("record_resource_return", ARM, operation)
        phases.append({"phase": label, "operation_id": operation, "reserved_wall_ms": check_wall_ms,
                       "observed_monotonic_ns": elapsed_ns})

    def encode(frames_in, history):
        check("before_encoder")
        if fixture:
            encoder = fixture_encoder
        else:
            from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
            encoder = native.native_rgb_encoder(model, device, p0, p0.image_transform(), frame_shape)
        result = encoder(frames_in, history)
        check("after_encoder")
        return result

    def plan(req, context, meter):
        need(req == request and meter.decision_id == decision, "guard request mismatch")
        check("before_planner")
        choose = sra.planner(model=model, device=device, replay=replay,
                             manifest_path=method_source["path"], manifest_sha256=method_source["sha256"],
                             target_path=target_path, audit_path=audit_path,
                             fixture_compile=fixture_compile, fixture_replay=fixture_replay)
        result = choose(req, context, meter)
        check("after_planner")
        guard_receipt = {"schema": "codex_sra_guarded_budget_v1", "decision_id": decision,
                         "request_sha256": online.sha(online.encode(req)),
                         "manifest_sha256": method_source["sha256"], "proof_sha256": proof_sha256,
                         "fixture": fixture, "phases": phases, "io_bytes": None, "flops": None,
                         "wall_reservation_is_realtime_preemption": False, "equal_total_compute": False}
        guard_sha = sra._save_exclusive(guard_receipt_path, guard_receipt)
        marker = decision + ":checkpoint_guard_receipt:" + guard_sha
        _reserve(journal, ARM, marker, {"setup_steps": 1})
        journal.call("record_resource_return", ARM, marker)
        return result

    return online.online_select(journal, ARM, frames, sources, method_source, encode, plan, receipt_path)


def select(*, journal, frames, sources, method_source, model, device, replay,
           frame_shape, proof_path, proof_sha256, target_path, audit_path, receipt_path,
           guard_receipt_path, check_wall_ms):
    """Native-only guarded SRA entry; callbacks and saved winners are not inputs."""
    return _select(journal=journal, frames=frames, sources=sources, method_source=method_source,
                   model=model, device=device, replay=replay,
                   frame_shape=frame_shape, proof_path=proof_path, proof_sha256=proof_sha256,
                   target_path=target_path, audit_path=audit_path, receipt_path=receipt_path,
                   guard_receipt_path=guard_receipt_path, check_wall_ms=check_wall_ms)


def consume(journal_path, journal_sha256, receipts, audit_path, audit_sha256,
            guard_receipt_path, guard_receipt_sha256):
    """Saved-only audit; reject selections without all four ordered guard checks."""
    result = sra.consume_sra(journal_path, journal_sha256, receipts, audit_path, audit_sha256)
    need(sra.file_sha(guard_receipt_path) == guard_receipt_sha256, "guard receipt hash")
    guard = sra.read_json(guard_receipt_path)
    need(guard["schema"] == "codex_sra_guarded_budget_v1" and guard["fixture"] == result["fixture"]
         and guard["io_bytes"] is None and guard["flops"] is None
         and guard["wall_reservation_is_realtime_preemption"] is False
         and guard["equal_total_compute"] is False, "guard receipt schema")
    matches = [sra.read_json(path) for path, _ in receipts
               if sra.read_json(path)["decision_id"] == guard["decision_id"]]
    need(len(matches) == 1, "guard decision identity")
    receipt = matches[0]
    need(receipt["request_sha256"] == guard["request_sha256"] and
         receipt["method_source"]["sha256"] == guard["manifest_sha256"], "guard binding")
    manifest = sra.checked_manifest(receipt["method_source"]["path"], guard["manifest_sha256"])
    source = manifest["files"]["guarded_sra_entry"]
    need(manifest["files"]["checkpoint_identity_proof"]["sha256"] == guard["proof_sha256"] and
         Path(source["path"]).resolve() == HERE and source["sha256"] == sra.file_sha(HERE),
         "guard source binding")
    phases = guard["phases"]
    need(len(phases) == 4 and [p["phase"] for p in phases] == list(PHASES), "guard phase census")
    begins = [e["payload"] for e in (json.loads(line) for line in Path(journal_path).read_bytes().splitlines())
              if e["kind"] == "begin"]
    positions = []
    for phase in phases:
        ms = phase["reserved_wall_ms"]
        elapsed = phase["observed_monotonic_ns"]
        op = guard["decision_id"] + ":checkpoint_guard:" + phase["phase"]
        need(type(ms) is int and ms > 0 and type(elapsed) is int and 0 <= elapsed <= ms * 1_000_000
             and phase["operation_id"] == op, "guard elapsed or identity")
        attempt = [i for i, e in enumerate(begins) if e["operation"] == "record_resource_attempt"
                   and e["args"] == [ARM, op, {"setup_steps": 1, "wall_ms": ms}]]
        ret = [i for i, e in enumerate(begins) if e["operation"] == "record_resource_return"
               and e["args"] == [ARM, op]]
        need(len(attempt) == len(ret) == 1 and attempt[0] < ret[0], "guard budget binding")
        positions.append((attempt[0], ret[0]))
    marker = guard["decision_id"] + ":checkpoint_guard_receipt:" + guard_receipt_sha256
    marker_a = [i for i, e in enumerate(begins) if e["operation"] == "record_resource_attempt"
                and e["args"] == [ARM, marker, {"setup_steps": 1}]]
    marker_r = [i for i, e in enumerate(begins) if e["operation"] == "record_resource_return"
                and e["args"] == [ARM, marker]]
    select = [i for i, e in enumerate(begins) if e["operation"] == "select" and e["args"][1] == receipt["request"]]
    events = receipt["metered_events"]
    def position(op, op_id):
        found = [i for i, e in enumerate(begins) if e["operation"] == op and e["args"][:2] == [ARM, op_id]]
        need(len(found) == 1, "meter position")
        return found[0]
    visual_a = position("record_resource_attempt", events[0]["operation_id"])
    visual_r = position("record_resource_return", events[0]["operation_id"])
    first_model = position("record_resource_attempt", events[1]["operation_id"])
    last_model = position("record_resource_return", events[-1]["operation_id"])
    need(len(marker_a) == len(marker_r) == len(select) == 1 and
         visual_a < positions[0][0] < positions[0][1] < positions[1][0] < positions[1][1] < visual_r
         < positions[2][0] < positions[2][1] < first_model <= last_model
         < positions[3][0] < positions[3][1] < marker_a[0] < marker_r[0] < select[0],
         "guard transaction order")
    return {"sra": result, "guard_receipt_sha256": guard_receipt_sha256,
            "guard_checks": 4, "setup_steps_reserved": 5,
            "guard_wall_ms_reserved": sum(p["reserved_wall_ms"] for p in phases),
            "guard_wall_ns_observed": sum(p["observed_monotonic_ns"] for p in phases),
            "io_bytes": None, "flops": None, "equal_total_compute": False}
