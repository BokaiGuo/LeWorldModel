import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.spectral_reachability import codex_sra_guarded_budget_20261030 as g
from experiments.spectral_reachability import codex_sra_checkpoint_identity_20261030 as identity
from experiments.spectral_reachability import codex_sra_online_run_bank_20261030 as sra
from experiments.spectral_reachability import codex_online_model_action_20261030 as online

spec = importlib.util.spec_from_file_location("sra_fixture", Path(__file__).parent / "test_codex_sra_online_run_bank_20261030.py")
f = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f)
t = f.t


class ActionEncoder(torch.nn.Module):
    def forward(self, value):
        return value


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.action_encoder = ActionEncoder()
        self.register_buffer("transient", torch.tensor([2.]), persistent=False)
        self.interpolate_pos_encoding = True
        self.eval().requires_grad_(False)

    def predict(self, state, actions):
        return state


class Replay:
    def __init__(self, model):
        self.model = model

    def __call__(self, context, actions):
        self.model.predict(np.zeros((1, 3, 192)), self.model.action_encoder(np.zeros((1, 3, 10))))
        return context


def setup(tmp_path, *, setup_cap=10, wall_cap=50000):
    model = Model()
    policy = tmp_path / "policy"
    policy.mkdir()
    torch.save(model.state_dict(), policy / "weights.pt")
    (policy / "config.json").write_text("{}")
    weights = sra.file_sha(policy / "weights.pt")
    config_sha = sra.file_sha(policy / "config.json")
    proof = identity.checkpoint_proof(policy, weights, config_sha, fixture_loader=lambda path, device: copy.deepcopy(model))
    proof_path = tmp_path / "proof.json"
    proof_sha = identity.save_proof(proof_path, proof)
    sources = {}
    target = tmp_path / "target.json"
    target.write_text(json.dumps({"goal_rgb_sha256": "c" * 64, "goal_vectors": [[0.] * 192],
                                  "scoring": {"metric": {"projection": np.eye(192).tolist()}}}))
    for name in t.base.IDENTITY:
        path = policy / "weights.pt" if name == "model_sha256" else target if name == "goal_sha256" else tmp_path / name
        if path not in (policy / "weights.pt", target):
            path.write_bytes(name.encode())
        sources[name] = {"path": str(path), "sha256": sra.file_sha(path)}
    files = {name: {"path": str(sra.DIR / name), "sha256": sra.file_sha(sra.DIR / name)}
             for name in sra.REQUIRED}
    for name, path in {"checkpoint_identity_proof": proof_path, "checkpoint_loader": identity.LOADER,
                       "checkpoint_weights": policy / "weights.pt", "checkpoint_config": policy / "config.json",
                       "checkpoint_guard": identity.HERE, "guarded_sra_entry": g.HERE,
                       "pusht_transform_source": g.HERE.parents[1] / "run_pusht_counterfactual.py"}.items():
        files[name] = {"path": str(path), "sha256": sra.file_sha(path)}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema": "codex_sra_online_sources_v1", "files": files, "action_units": "fixture"}))
    identity_values = {k: v["sha256"] for k, v in sources.items()}
    caps = {arm: {key: 5000 for key in t.base.AXES} for arm in t.base.ARMS}
    caps[t.B]["setup_steps"] = setup_cap
    caps[t.B]["wall_ms"] = wall_cap
    config = {"identities": {a: copy.deepcopy(identity_values) for a in t.base.ARMS},
              "initial_frames": {a: [online.sha(x) for x in t.FRAMES] for a in t.base.ARMS},
              "initial_actions": {a: [copy.deepcopy(t.PACKET), copy.deepcopy(t.PACKET)] for a in t.base.ARMS},
              "horizon": 5, "packet_steps": 5, "caps": caps}
    j = t.jmod.Journal(tmp_path / "journal", config)
    args = dict(journal=j, frames=t.FRAMES, sources=sources,
                method_source={"path": str(manifest), "sha256": sra.file_sha(manifest)},
                model=model, device=None, replay=Replay(model),
                frame_shape=None, proof_path=proof_path, proof_sha256=proof_sha, target_path=target,
                audit_path=tmp_path / "audit.json", receipt_path=tmp_path / "receipt.json",
                guard_receipt_path=tmp_path / "guard.json", check_wall_ms=10000,
                fixture_encoder=t.encoder, fixture_compile=f.compile_fixture, fixture_replay=True)
    return args


def finish(args):
    j = args["journal"]
    if j.protocol.state(t.A)["phase"] != "stopped":
        j.call("stop", t.A, "fixture")
    if j.protocol.state(t.B)["phase"] != "stopped":
        j.call("stop", t.B, "fixture")
    return j.finish()


def test_guarded_run_bank_journal_and_consumer(tmp_path):
    args = setup(tmp_path)
    out = g._select(**args)
    state = args["journal"].protocol.state(t.B)
    assert state["attempted"]["setup_steps"] == 5
    assert state["attempted"]["wall_ms"] == 40000
    assert state["attempted"]["predict_calls"] == 88
    journal_sha = finish(args)
    receipt = json.loads(args["receipt_path"].read_text())
    audit_sha = receipt["plan"]["selected_id"].split("@")[1]
    guard_sha = sra.file_sha(args["guard_receipt_path"])
    saved = g.consume(args["journal"].path, journal_sha, [(args["receipt_path"], out["receipt_sha256"])],
                      args["audit_path"], audit_sha, args["guard_receipt_path"], guard_sha)
    assert saved["guard_checks"] == 4 and saved["setup_steps_reserved"] == 5
    assert saved["guard_wall_ms_reserved"] == 40000 and saved["guard_wall_ns_observed"] > 0
    assert saved["sra"]["fixture"] and saved["io_bytes"] is None and saved["flops"] is None
    assert saved["equal_total_compute"] is False


@pytest.mark.parametrize("axis", ["setup", "wall"])
def test_guard_budget_checked_before_checkpoint_read(tmp_path, monkeypatch, axis):
    args = setup(tmp_path, setup_cap=0 if axis == "setup" else 10, wall_cap=0 if axis == "wall" else 50000)
    called = []
    monkeypatch.setattr(g, "_identity_check", lambda *a: called.append(True))
    with pytest.raises(ValueError, match="guard budget exhausted"):
        g._select(**args)
    assert not called and args["journal"].protocol.state(t.B)["phase"] == "stopped"
    assert not args["receipt_path"].exists()
    digest = finish(args)
    assert t.jmod.consume(args["journal"].path, digest)["snapshot"]["rows"][1]["pending_resources"]


def test_guard_overrun_pending_without_selection(tmp_path, monkeypatch):
    args = setup(tmp_path)
    args["check_wall_ms"] = 1
    moments = iter([0, 2_000_001])
    monkeypatch.setattr(g.time, "monotonic_ns", lambda: next(moments))
    with pytest.raises(ValueError, match="wall reservation exceeded"):
        g._select(**args)
    assert args["journal"].protocol.state(t.B)["attempted"]["wall_ms"] == 1
    assert args["journal"].protocol.state(t.B)["pending_resources"]
    assert not args["receipt_path"].exists()
    finish(args)


def test_model_drift_after_run_bank_refuses_selection(tmp_path):
    args = setup(tmp_path)
    def change(model, device, ready, compiled, replay, emit):
        value = f.compile_fixture(model, device, ready, compiled, replay, emit)
        model.transient[0] += 1
        return value
    args["fixture_compile"] = change
    with pytest.raises(ValueError, match="buffer drift"):
        g._select(**args)
    assert args["audit_path"].exists()  # orphan audit is not selection
    assert not args["receipt_path"].exists() and not args["guard_receipt_path"].exists()
    assert args["journal"].protocol.state(t.B)["pending_resources"]
    finish(args)


def test_bad_checkpoint_stops_before_encoder(tmp_path):
    args = setup(tmp_path)
    with torch.no_grad():
        args["model"].linear.weight.add_(1)
    called = []
    args["fixture_encoder"] = lambda *x: called.append(True)
    with pytest.raises(ValueError, match="does not match checkpoint"):
        g._select(**args)
    assert not called and args["journal"].protocol.state(t.B)["pending_resources"]
    assert not args["receipt_path"].exists()
    finish(args)


def test_receipt_marker_budget_blocks_selection(tmp_path):
    args = setup(tmp_path, setup_cap=4)
    with pytest.raises(ValueError, match="guard budget exhausted"):
        g._select(**args)
    assert args["audit_path"].exists() and args["guard_receipt_path"].exists()
    assert not args["receipt_path"].exists()
    assert args["journal"].protocol.state(t.B)["attempted"]["setup_steps"] == 4
    finish(args)


def test_saved_guard_hash_and_fixture_hooks(tmp_path):
    args = setup(tmp_path)
    with pytest.raises(ValueError, match="incomplete fixture mode"):
        g._select(**{**args, "fixture_compile": None})
    out = g._select(**args)
    digest = finish(args)
    receipt = json.loads(args["receipt_path"].read_text())
    audit_sha = receipt["plan"]["selected_id"].split("@")[1]
    args["guard_receipt_path"].write_bytes(b"tampered")
    with pytest.raises(ValueError, match="guard receipt hash"):
        g.consume(args["journal"].path, digest, [(args["receipt_path"], out["receipt_sha256"])],
                  args["audit_path"], audit_sha, args["guard_receipt_path"], "0" * 64)
