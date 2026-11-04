import json
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.spectral_reachability import reacher_pc_p1_fd_cp5_host_20261102_codexexclusive as host
from experiments.spectral_reachability import reacher_pc_p1_fd_cp5_saved_20261102_codexexclusive as saved


def image(n):
    return np.full((3, 4, 3), n, dtype=np.uint8)


def test_gpu_uuid_accepts_bare_and_prefixed_but_not_other_device():
    assert host.canonical_gpu_uuid(host.UUID) == host.canonical_gpu_uuid(host.UUID.removeprefix("GPU-"))
    assert host.canonical_gpu_uuid(host.UUID) != host.canonical_gpu_uuid("00000000-0000-0000-0000-000000000000")
    with pytest.raises(ValueError):
        host.canonical_gpu_uuid("not-a-uuid")


def test_capture_timeline_and_actions():
    frames = [image(1), image(2), image(3)]
    actions = np.arange(30, dtype=np.float32).reshape(3, 10)
    packed = host.capture(frames, image(4), image(5), actions)
    assert [host.context_lib.unpack_rgb_observation(x)[0, 0, 0] for x in packed["frames"]] == [2, 3, 4]
    assert packed["history"] == actions.tolist()
    with pytest.raises(ValueError):
        host.capture(frames, image(4), image(5), actions[:2])


def test_context_binds_prefix_and_timeline():
    warm = np.arange(30, dtype=np.float32).reshape(3, 5, 2)
    prefix = np.full((5, 2), .5, dtype=np.float32)
    root = {"root_rgb_sha256": host.context_lib.rgb_digest(image(3)),
            "goal_rgb_sha256": host.context_lib.rgb_digest(image(5)), "warmup_packets": warm.tolist()}
    arm = {"raw_prefix": prefix.tolist()}
    history = np.concatenate((warm[1:].reshape(2, 10), prefix.reshape(1, 10)))
    value = host.capture([image(0), image(2), image(3)], image(4), image(5), history)
    saved.verify_context(value, root, arm, "successor",
                         {"observed_successor_context": {"root_rgb_sha256": host.context_lib.rgb_digest(image(4))}})
    value["history"][-1][0] = -1
    with pytest.raises(ValueError):
        saved.verify_context(value, root, arm, "successor")


def test_durable_journal_rejects_tampering_and_truncation(tmp_path):
    path = tmp_path / "events"
    journal = host.durable.Journal(path)
    journal.emit({"kind": "first"})
    journal.emit({"kind": "second"})
    journal.close()
    assert len(saved.verify_journal(path)) == 2
    data = path.read_bytes()
    path.write_bytes(data.replace(b"second", b"Second"))
    with pytest.raises(ValueError):
        saved.verify_journal(path)
    path.write_bytes(data[:-1])
    with pytest.raises(ValueError):
        saved.verify_journal(path)


def test_instrument_records_error_and_restores_prepare(tmp_path):
    journal = host.durable.Journal(tmp_path / "events")
    def prepare(*args):
        return None
    def two_stage(**kwargs):
        runner.p0._prepare(None, [image(1)] * 3, image(2), image(3), np.zeros((3, 10)), None)
        kwargs["env"].step(np.zeros(2))
        raise RuntimeError("physical failure")
    runner = SimpleNamespace(p0=SimpleNamespace(_prepare=prepare), _two_stage=two_stage)
    env = SimpleNamespace(step=lambda action: (None, 0, True, False, {}))
    original = host.instrument(runner, journal, 20263160)
    with pytest.raises(RuntimeError):
        runner._two_stage(name="selected", env=env)
    assert runner.p0._prepare is prepare
    runner._two_stage = original
    journal.close()
    events = saved.verify_journal(tmp_path / "events")
    assert [x["kind"] for x in events] == ["arm_started", "context", "step_intent", "step_returned", "arm_error"]
    assert events[-2]["terminated"] is True
    assert events[-1]["error"] == "physical failure"


def test_run_requires_explicit_authority_before_inputs(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with pytest.raises(ValueError, match="authority"):
        host.run(tmp_path / "none", "bad", tmp_path / "out", "no")


def test_real_fixed_preparation_binds_zero_and_rejects_source_drift(tmp_path):
    results = host.ROOT / "results"
    args = (results / "reacher_pc_fd_suffix_cuda_20261102_v1/report.json",
            results / "reacher_pc_fd_suffix_saved_audit_20261102_v1.json",
            results / "reacher_pc_fd_common_physical_20261102_PLAN_v1.json",
            results / "reacher_pc_p1_common_tail_preflight_20261102_codexexclusive_02.json")
    plan = host.plan(*args)
    for root in plan["bridge"]["roots"]:
        assert root["arms"][1]["proposal_index"] == 0
        assert np.array_equal(root["arms"][1]["params"], np.zeros((1, 10)))
    altered = tmp_path / "altered.json"
    altered.write_text("{}")
    with pytest.raises(ValueError, match="source SHA"):
        host.plan(altered, *args[1:])
