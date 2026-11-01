import copy
import importlib.util
import json
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "experiments/spectral_reachability/codex_online_model_action_20261030.py"
spec = importlib.util.spec_from_file_location("online_action", PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
jmod = m.load_journal()
base = jmod.load_base()
A, B = base.ARMS
FRAMES = [b"fixture-one", b"fixture-two", b"fixture-three"]
PACKET = [[0.1, -0.2] for _ in range(5)]


def setup(tmp_path, cap=100):
    sources = {}
    for name in base.IDENTITY:
        path = tmp_path / name
        path.write_bytes(name.encode())
        sources[name] = {"path": str(path), "sha256": m.sha(path.read_bytes())}
    method = tmp_path / "method"
    method.write_bytes(b"fixture method, not a real planner")
    method = {"path": str(method), "sha256": m.sha(method.read_bytes())}
    identity = {k: v["sha256"] for k, v in sources.items()}
    config = {"identities": {a: copy.deepcopy(identity) for a in base.ARMS},
              "initial_frames": {a: [m.sha(f) for f in FRAMES] for a in base.ARMS},
              "initial_actions": {a: [copy.deepcopy(PACKET), copy.deepcopy(PACKET)] for a in base.ARMS},
              "horizon": 5, "packet_steps": 5,
              "caps": {a: {k: cap for k in base.AXES} for a in base.ARMS}}
    return jmod.Journal(tmp_path / "journal", config), sources, method


def encoder(frames, history):
    assert frames == FRAMES and history == [PACKET, PACKET]
    return {"layout": "states_with_between_state_actions_v1", "context_length": 3,
            "latent_dim": 192, "action_dim": 10, "latent_history": [[0.] * 192 for _ in range(3)],
            "action_history": [[0.] * 10 for _ in range(2)]}


def planner(request, ctx, meter):
    meter.run("predict", {"predict_calls": 1, "predict_samples": 2}, lambda: None)
    return {"mode": "online_callback", "request_sha256": m.sha(m.encode(request)),
            "context_sha256": m.sha(m.encode(ctx)),
            "candidates": [{"id": "a", "cost": 1., "commands": PACKET},
                           {"id": "b", "cost": 2., "commands": PACKET}],
            "selected_id": "a"}


def select(j, sources, method, tmp_path, arm=A, enc=encoder, plan=planner, frames=FRAMES):
    return m.online_select(j, arm, frames, sources, method, enc, plan, tmp_path / (arm + ".json"))


def test_two_arms_online_receipts_and_saved_consumer(tmp_path):
    j, sources, method = setup(tmp_path)
    seen = []
    def checked_encoder(frames, history):
        records = [json.loads(line) for line in j.path.read_bytes().splitlines()]
        assert any(x["kind"] == "begin" and x["payload"]["operation"] == "record_resource_attempt"
                   and x["payload"]["args"][2] == {"visual_encoder": 1} for x in records)
        seen.append("encoder")
        return encoder(frames, history)
    outputs = [select(j, sources, method, tmp_path, arm, checked_encoder) for arm in (A, B)]
    assert seen == ["encoder", "encoder"]
    for arm in (A, B):
        j.call("stop", arm, "fixture_no_physical")
    digest = j.finish()
    receipt_list = [(tmp_path / (arm + ".json"), out["receipt_sha256"]) for arm, out in zip((A, B), outputs)]
    result = m.consume_online(j.path, digest, receipt_list)
    assert result["receipt_count"] == 2 and not result["actual_model_or_physical_verified"]
    assert not result["equal_total_compute"] and result["flops"] is None
    assert result["journal"]["snapshot"]["fixed_arm_denominator"] == 2
    with pytest.raises(ValueError, match="census"):
        m.consume_online(j.path, digest, receipt_list[:1])
    (tmp_path / (A + ".json")).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="receipt hash"):
        m.consume_online(j.path, digest, receipt_list)


@pytest.mark.parametrize("variant", ["saved", "request", "context", "winner", "nan", "action", "unmetered"])
def test_invalid_plans_fail_closed(tmp_path, variant):
    j, sources, method = setup(tmp_path)
    def bad(request, ctx, meter):
        if variant == "unmetered":
            return {}
        result = planner(request, ctx, meter)
        if variant == "saved":
            result["mode"] = "saved_winner"
        elif variant == "request":
            result["request_sha256"] = "0" * 64
        elif variant == "context":
            result["context_sha256"] = "0" * 64
        elif variant == "winner":
            result["selected_id"] = "b"
        elif variant == "nan":
            result["candidates"][0]["cost"] = float("nan")
        else:
            result["candidates"][0]["commands"] = [[2, 0]] * 5
        return result
    with pytest.raises(ValueError):
        select(j, sources, method, tmp_path, plan=bad)
    assert j.protocol.state(A)["phase"] == "stopped"
    assert not (tmp_path / (A + ".json")).exists()
    j.call("stop", B, "fixture")
    j.finish()


def test_cap_prevents_callback_and_failure_preserves_pending(tmp_path):
    j, sources, method = setup(tmp_path, cap=0)
    called = []
    with pytest.raises(ValueError, match="budget exhausted"):
        select(j, sources, method, tmp_path, enc=lambda *args: called.append(True))
    assert not called and j.protocol.state(A)["phase"] == "stopped"
    j.call("stop", B, "fixture")
    j.finish()


def test_callback_failure_leaves_attempt_pending(tmp_path):
    j, sources, method = setup(tmp_path)
    def failure(request, ctx, meter):
        def fail():
            raise RuntimeError("fixture failure")
        meter.run("predict", {"predict_calls": 1}, fail)
    with pytest.raises(RuntimeError, match="fixture failure"):
        select(j, sources, method, tmp_path, plan=failure)
    assert len(j.protocol.state(A)["pending_resources"]) == 1
    j.call("stop", B, "fixture")
    out = jmod.consume(j.path, j.finish())
    assert len(out["snapshot"]["rows"][0]["pending_resources"]) == 1


def test_saved_consumer_refuses_source_drift(tmp_path):
    j, sources, method = setup(tmp_path)
    out = select(j, sources, method, tmp_path)
    j.call("stop", A, "fixture")
    j.call("stop", B, "fixture")
    digest = j.finish()
    receipts = [(tmp_path / (A + ".json"), out["receipt_sha256"])]
    assert m.consume_online(j.path, digest, receipts)["receipt_count"] == 1
    Path(sources["model_sha256"]["path"]).write_bytes(b"replaced")
    with pytest.raises(ValueError, match="source drift"):
        m.consume_online(j.path, digest, receipts)


@pytest.mark.parametrize("variant", ["source", "frames", "method", "during_callback", "existing_receipt"])
def test_provenance_and_exclusive_receipt(tmp_path, variant):
    j, sources, method = setup(tmp_path)
    frames = FRAMES
    enc = encoder
    if variant == "source":
        Path(sources["root_sha256"]["path"]).write_bytes(b"changed")
    elif variant == "frames":
        frames = [b"wrong", *FRAMES[1:]]
    elif variant == "method":
        Path(method["path"]).write_bytes(b"changed")
    elif variant == "during_callback":
        def enc(frames, history):
            Path(method["path"]).write_bytes(b"changed")
            return encoder(frames, history)
    else:
        (tmp_path / (A + ".json")).write_bytes(b"keep")
    with pytest.raises((ValueError, FileExistsError)):
        select(j, sources, method, tmp_path, enc=enc, frames=frames)
    if variant == "existing_receipt":
        assert (tmp_path / (A + ".json")).read_bytes() == b"keep"
    assert j.protocol.state(A)["phase"] in ("observe", "stopped")
    if j.protocol.state(A)["phase"] == "observe":
        j.call("stop", A, "fixture")
    j.call("stop", B, "fixture")
    j.finish()
