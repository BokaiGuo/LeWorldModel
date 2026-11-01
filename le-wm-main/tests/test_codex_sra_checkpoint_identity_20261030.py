import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.spectral_reachability import codex_sra_checkpoint_identity_20261030 as m
from experiments.spectral_reachability import codex_sra_online_run_bank_20261030 as s
from experiments.spectral_reachability import codex_online_model_action_20261030 as online

spec = importlib.util.spec_from_file_location("online_helpers", Path(__file__).parent / "test_codex_online_model_action_20261030.py")
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(2, 2)
        self.register_buffer("persistent", torch.tensor([3.]))
        self.register_buffer("transient", torch.tensor([4.]), persistent=False)
        self.interpolate_pos_encoding = True
        self.eval().requires_grad_(False)


@pytest.fixture
def prepared(tmp_path):
    torch.manual_seed(41)
    model = Tiny()
    policy = tmp_path / "policy"
    policy.mkdir()
    torch.save(model.state_dict(), policy / "weights.pt")
    (policy / "config.json").write_text('{"fixture":true}')
    weights, config = s.file_sha(policy / "weights.pt"), s.file_sha(policy / "config.json")
    proof = m.checkpoint_proof(policy, weights, config, fixture_loader=lambda path, device: copy.deepcopy(model))
    proof_path = tmp_path / "identity.json"
    proof_sha = m.save_proof(proof_path, proof)
    refs = {name: {"path": str(s.DIR / name), "sha256": s.file_sha(s.DIR / name)} for name in s.REQUIRED}
    for name, path in {"checkpoint_identity_proof": proof_path, "checkpoint_loader": m.LOADER,
                       "checkpoint_weights": policy / "weights.pt", "checkpoint_config": policy / "config.json",
                       "checkpoint_guard": m.HERE}.items():
        refs[name] = {"path": str(path), "sha256": s.file_sha(path)}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema": "codex_sra_online_sources_v1",
                                    "files": refs, "action_units": "fixture"}))
    return SimpleNamespace(model=model, replay=SimpleNamespace(model=model), identity={"model_sha256": weights},
                           policy=policy, proof=proof, proof_path=proof_path, proof_sha=proof_sha,
                           manifest=manifest, manifest_sha=s.file_sha(manifest))


def verify(p, *, allow=True):
    return m.verify_proof(p.proof_path, p.proof_sha, p.model, p.replay, p.identity, allow_fixture=allow)


def guards(p, encoder=lambda *a: "encoded", planner=lambda *a: "planned"):
    return m.guarded_callbacks(model=p.model, replay=p.replay, identity=p.identity,
                               proof_path=p.proof_path, proof_sha256=p.proof_sha,
                               manifest_path=p.manifest, manifest_sha256=p.manifest_sha,
                               encoder=encoder, planner=planner, allow_fixture=True)


def test_fixture_checkpoint_reference_and_exclusive_proof(prepared):
    p = prepared
    assert verify(p)["fixture"] is True
    assert p.proof["checkpoint_tensor_count"] == 3
    assert "transient" in p.proof["reference"]["buffers"]
    assert "transient" not in p.proof["reference"]["persistent_state"]
    with pytest.raises(ValueError, match="fixture"):
        verify(p, allow=False)
    with pytest.raises(FileExistsError):
        m.save_proof(p.proof_path, p.proof)


@pytest.mark.parametrize("defect", ["parameter", "persistent", "transient", "mode", "requires_grad", "replay", "identity"])
def test_runtime_identity_rejects_drift(prepared, defect):
    p = prepared
    if defect == "parameter":
        with torch.no_grad():
            p.model.layer.weight[0, 0] += 1
    elif defect == "persistent":
        p.model.persistent[0] += 1
    elif defect == "transient":
        p.model.transient[0] += 1
    elif defect == "mode":
        p.model.train()
    elif defect == "requires_grad":
        p.model.layer.weight.requires_grad_(True)
    elif defect == "replay":
        p.replay.model = copy.deepcopy(p.model)
    else:
        p.identity["model_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        verify(p)


@pytest.mark.parametrize("asset", ["weights.pt", "config.json"])
def test_checkpoint_drift_rejected(prepared, asset):
    path = prepared.policy / asset
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="drift"):
        verify(prepared)


def test_checkpoint_and_reference_mismatch_refused(tmp_path):
    policy = tmp_path / "policy"
    policy.mkdir()
    a, b = Tiny(), Tiny()
    with torch.no_grad():
        b.layer.weight.add_(1)
    torch.save(a.state_dict(), policy / "weights.pt")
    (policy / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="reference state differ"):
        m.checkpoint_proof(policy, s.file_sha(policy / "weights.pt"), s.file_sha(policy / "config.json"),
                           fixture_loader=lambda path, device: b)


def test_guard_detects_mutation_inside_encoder(prepared):
    p = prepared
    def mutate(*args):
        p.model.transient[0] += 1
        return "encoded"
    enc, _ = guards(p, encoder=mutate)
    with pytest.raises(ValueError, match="buffer drift"):
        enc([], [])


def test_guard_requires_manifest_binding(prepared):
    p = prepared
    data = json.loads(p.manifest.read_text())
    del data["files"]["checkpoint_guard"]
    p.manifest.write_text(json.dumps(data))
    p.manifest_sha = s.file_sha(p.manifest)
    with pytest.raises(ValueError, match="missing checkpoint binding"):
        guards(p)


def test_guarded_request_requires_manifest_and_identity(prepared):
    p = prepared
    enc, plan = guards(p)
    assert enc([], []) == "encoded"
    req = {"identity": p.identity}
    decision = m.online.sha(m.online.encode({"request": req, "method_sha256": p.manifest_sha}))
    assert plan(req, {}, SimpleNamespace(decision_id=decision)) == "planned"
    with pytest.raises(ValueError, match="request binding"):
        plan(req, {}, SimpleNamespace(decision_id="0" * 64))


def test_forged_proof_cannot_hide_wrong_live_checkpoint(prepared):
    p = prepared
    with torch.no_grad():
        p.model.layer.weight.add_(1)
    proof = copy.deepcopy(p.proof)
    proof["reference"] = m.model_snapshot(p.model)
    forged = p.proof_path.parent / "forged.json"
    forged_sha = m.save_proof(forged, proof)
    with pytest.raises(ValueError, match="does not match checkpoint"):
        m.verify_proof(forged, forged_sha, p.model, p.replay, p.identity, allow_fixture=True)


@pytest.mark.parametrize("mutate", [False, True])
def test_guarded_online_select_and_fail_closed(prepared, tmp_path, mutate):
    p = prepared
    sources = {}
    for name in t.base.IDENTITY:
        path = p.policy / "weights.pt" if name == "model_sha256" else tmp_path / name
        if name != "model_sha256":
            path.write_bytes(name.encode())
        sources[name] = {"path": str(path), "sha256": s.file_sha(path)}
    identity = {k: v["sha256"] for k, v in sources.items()}
    config = {"identities": {a: copy.deepcopy(identity) for a in t.base.ARMS},
              "initial_frames": {a: [online.sha(f) for f in t.FRAMES] for a in t.base.ARMS},
              "initial_actions": {a: [copy.deepcopy(t.PACKET), copy.deepcopy(t.PACKET)] for a in t.base.ARMS},
              "horizon": 5, "packet_steps": 5,
              "caps": {a: {k: 50 for k in t.base.AXES} for a in t.base.ARMS}}
    j = t.jmod.Journal(tmp_path / "journal", config)
    def enc(frames, history):
        if mutate:
            p.model.transient[0] += 1
        return t.encoder(frames, history)
    args = dict(journal=j, arm=t.B, frames=t.FRAMES, sources=sources,
                method_source={"path": str(p.manifest), "sha256": p.manifest_sha},
                model=p.model, replay=p.replay, proof_path=p.proof_path, proof_sha256=p.proof_sha,
                encoder=enc, planner=t.planner, receipt_path=tmp_path / "selection.json", allow_fixture=True)
    if mutate:
        with pytest.raises(ValueError, match="buffer drift"):
            m.guarded_online_select(**args)
        assert j.protocol.state(t.B)["phase"] == "stopped"
        assert not (tmp_path / "selection.json").exists()
    else:
        result = m.guarded_online_select(**args)
        j.call("stop", t.B, "fixture")
        assert result["receipt_sha256"] == s.file_sha(tmp_path / "selection.json")
    j.call("stop", t.A, "fixture")
    t.jmod.consume(j.path, j.finish())
