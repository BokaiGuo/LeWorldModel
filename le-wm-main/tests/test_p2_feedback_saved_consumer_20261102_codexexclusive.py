import hashlib
import json

import numpy as np
import pytest

from experiments.spectral_reachability import p2_prefix_feedback_common_budget_20261102_codexexclusive as k
from experiments.spectral_reachability import p2_cp5_launcher_preflight_20261102_codexexclusive as ledger_module
from experiments.spectral_reachability import p2_feedback_saved_consumer_20261102_codexexclusive as consumer


def raw_json(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(x):
    return hashlib.sha256(x).hexdigest()


def build_archive(path):
    path.mkdir()
    def put(name, data):
        (path / name).write_bytes(data)
        return {"path": name, "sha256": sha(data)}
    root = "a" * 64
    history = np.zeros((2, 10), dtype="<f4")
    base = np.zeros((224, 224, 3), np.uint8)
    bank = [np.full((10, 2), i / 100, np.float32).tobytes() for i in range(16)]
    warm = np.zeros((10, 2), np.float32)
    shared = np.zeros((15, 2), np.float32)
    artifacts = {}
    def start(arm, action, gradient, meter):
        meter.call("start_transition", lambda: None)
        meter.call("start_transition", lambda: None)
        if gradient:
            meter.call("start_gradient", lambda: None)
        return float(np.square(action - .2).sum()), np.full((10, 2), -.1, np.float32) if gradient else None
    def capture(arm, index, prefix):
        rgb = np.stack((base, base, np.full_like(base, index + (0 if arm == k.ARMS[0] else 32))))
        actions = np.stack((history[1], k.normalization.normalized_history(prefix)[0])).astype("<f4")
        receipt = {"arm": arm, "candidate_index": index, "root_sha256": root, "task_id": "task",
                   "prefix_hex": prefix.tobytes().hex(), "rgb_sha256": [sha(f.tobytes()) for f in rgb],
                   "action_history_hex": actions.tobytes().hex(), "guard": "fake", "snapshot": {"fake": True},
                   "terminated": False, "truncated": False}
        key = f"{index:02d}_{arm}"
        native_events = []
        for phase, seq in (("warmup", warm), ("plan", np.concatenate((shared, prefix)))):
            if phase == "plan":
                native_events.append({"kind": "root_guard_accepted", "root_sha256": root, "task_id": "task"})
            for step, command in enumerate(seq, 1):
                if phase == "plan" and step == 16:
                    native_events.append({"kind": "shared_prefix_guard_accepted", "root_sha256": root,
                                          "task_id": "task", "prefix_float32_hex": shared.tobytes().hex()})
                native_events.extend(({"kind": "segment_step_intent", "phase": phase, "step": step,
                                       "raw_float32_hex": command.tobytes().hex()},
                                      {"kind": "segment_step_returned", "phase": phase, "step": step,
                                       "terminated": False, "truncated": False}))
        native_events.append({"kind": "cp5_capture", "step": 20, "snapshot": {"fake": True},
                              "rgb_sha256": sha(rgb[2].tobytes())})
        artifacts[key] = {"receipt": put(key + "_receipt.json", raw_json(receipt)),
                          "rgb": put(key + "_rgb.bin", rgb.tobytes()),
                          "native_journal": put(key + "_native.jsonl",
                              b"".join(raw_json(e) + b"\n" for e in native_events))}
        return {"arm": arm, "candidate_index": index, "root_sha256": root, "task_id": "task",
                "prefix_hex": prefix.tobytes().hex(), "rgb_history": rgb, "action_history": actions,
                "capture_receipt_sha256": sha(raw_json(receipt)), "terminated": False, "truncated": False}
    def encode(rgb, actions):
        context = {"layout": "states_with_between_state_actions_v1",
                   "latent_history": np.full((3, 192), float(rgb[2, 0, 0, 0]), np.float32).tolist(),
                   "action_history": actions.tolist()}
        index = int(rgb[2, 0, 0, 0])
        key = f"{index % 32:02d}_{k.ARMS[0] if index < 32 else k.ARMS[1]}"
        artifacts[key]["context"] = put(key + "_context.json", raw_json(context))
        return context
    def suffix(arm, context, action, gradient, meter):
        meter.call("suffix_transition", lambda: None)
        if gradient:
            meter.call("suffix_gradient", lambda: None)
        return float(np.square(action[5:] - .1).sum()), np.full((10, 2), -.1, np.float32) if gradient else None
    events = []
    result = k.run_case(bank, root, "task", history, [sha(base.tobytes())] * 2,
                        evaluate_start=start, capture=capture, encode=encode,
                        evaluate_suffix=suffix, emit=events.append)
    l = ledger_module.Ledger(path / "budget.jsonl", {"public_step": 960})
    for i in range(16):
        for arm in k.ARMS:
            for _ in range(30):
                l.call("public_step", lambda: None, session=f"capture_{i:02d}_{arm}")
    tail = l.previous
    l.close()
    sources = {role: put("source_" + role, role.encode()) for role in consumer.SOURCE_ROLES}
    manifest = {"schema": consumer.SCHEMA, "root_sha256": root, "task_id": "task",
                "base_history": history.tolist(), "base_rgb_sha256": [sha(base.tobytes())] * 2,
                "warmup_hex": warm.tobytes().hex(), "shared_prefix_hex": shared.tobytes().hex(),
                "sources": sources, "bank": put("bank.bin", b"".join(bank)),
                "result": put("result.json", raw_json(result)),
                "events": put("events.json", raw_json(events)), "captures": artifacts,
                "budget": {"file": {"path": "budget.jsonl", "sha256": sha((path / "budget.jsonl").read_bytes())},
                           "tail_sha256": tail, "expected_counts": {"public_step": 960}}}
    ref = put("manifest.json", raw_json(manifest))
    return ref["sha256"], manifest


def reseal(path, manifest):
    data = raw_json(manifest)
    (path / "manifest.json").write_bytes(data)
    return sha(data)


def test_complete_fixture_audits_without_physical_authority(tmp_path):
    archive = tmp_path / "archive"
    anchor, _ = build_archive(archive)
    receipt = consumer.consume(archive, anchor)
    assert len(receipt["winners"]) == 2
    assert receipt["physical_authorization"] is False


@pytest.mark.parametrize("kind", ["bank", "context", "budget", "missing_source", "winner",
                                      "score_event", "native_stop", "meter_scope", "winner_action"])
def test_tampered_or_incomplete_archive_rejected(tmp_path, kind):
    archive = tmp_path / "archive"
    anchor, manifest = build_archive(archive)
    if kind == "bank":
        data = bytes(1280)
        (archive / "bank.bin").write_bytes(data)
        manifest["bank"]["sha256"] = sha(data)
        anchor = reseal(archive, manifest)
    elif kind == "context":
        ref = manifest["captures"]["00_composition"]["context"]
        context = json.loads((archive / ref["path"]).read_bytes())
        context["latent_history"][0][0] += 1
        data = raw_json(context)
        (archive / ref["path"]).write_bytes(data)
        ref["sha256"] = sha(data)
        anchor = reseal(archive, manifest)
    elif kind == "budget":
        manifest["budget"]["tail_sha256"] = "f" * 64
        anchor = reseal(archive, manifest)
    elif kind == "missing_source":
        del manifest["sources"]["model"]
        anchor = reseal(archive, manifest)
    elif kind in ("winner", "winner_action"):
        result = json.loads((archive / "result.json").read_bytes())
        if kind == "winner":
            result["winners"][k.ARMS[0]]["candidate_index"] = 15
        else:
            result["winners"][k.ARMS[0]]["raw_float32_hex"] = bytes(80).hex()
        data = raw_json(result)
        (archive / "result.json").write_bytes(data)
        manifest["result"]["sha256"] = sha(data)
        anchor = reseal(archive, manifest)
    elif kind == "native_stop":
        ref = manifest["captures"]["00_composition"]["native_journal"]
        data = (archive / ref["path"]).read_bytes() + raw_json({"kind": "segment_native_stop"}) + b"\n"
        (archive / ref["path"]).write_bytes(data)
        ref["sha256"] = sha(data)
        anchor = reseal(archive, manifest)
    else:
        events = json.loads((archive / "events.json").read_bytes())
        if kind == "score_event":
            result = json.loads((archive / "result.json").read_bytes())
            win = result["winners"][k.ARMS[0]]
            e = next(x for x in events if x.get("stage") == "suffix" and x["kind"] == "returned"
                     and x["candidate"] == win["candidate_index"] and x["arm"] == k.ARMS[0]
                     and x["iteration"] == win["iteration"])
            e["score"] += 10
        else:
            e = next(x for x in events if x["kind"] == "resource_finish" and "candidate" in x)
            e["candidate"] = 99
        data = raw_json(events)
        (archive / "events.json").write_bytes(data)
        manifest["events"]["sha256"] = sha(data)
        anchor = reseal(archive, manifest)
    with pytest.raises((ValueError, KeyError)):
        consumer.consume(archive, anchor)
