"""Synthetic contracts only: never load a checkpoint or the physical simulator."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.spectral_reachability import pusht_d2p_packet_contract as contract
from experiments.spectral_reachability import audit_pusht_d2p_packet_contract as audit


class RecordingEnv:
    """Deterministic non-neural simulator substitute with observable step count."""

    def __init__(self, *, terminal_at=None, bad_goal=False, impure_render=False):
        self.calls = []
        self.terminal_at = terminal_at
        self.bad_goal = bad_goal
        self.impure_render = impure_render
        self.closed = False

    @property
    def unwrapped(self):
        return self

    def reset(self, *, seed, options):
        self.state = np.asarray(options["state"], dtype=float).copy()
        self.goal_state = np.asarray(options["goal_state"], dtype=float).copy()
        self.calls = []
        goal = self._image(self.goal_state)
        return {"state": self.state.copy()}, {
            "goal_state": self.goal_state.copy() + int(self.bad_goal),
            "goal_proprio": self.goal_state[[0, 1, 5, 6]],
            "goal_pose": self.goal_state[2:5],
            "goal": goal,
        }

    def _get_obs(self):
        return self.state.copy()

    @staticmethod
    def _image(state):
        return np.full((224, 224, 3), int(state[0]) % 256, dtype=np.uint8)

    def render(self):
        if self.impure_render:
            self.state[0] += 1
        return self._image(self.state)

    def step(self, action):
        self.calls.append(np.asarray(action).copy())
        self.state[:2] += action
        return {"state": self.state.copy()}, 0.0, len(self.calls) == self.terminal_at, False, {}

    def close(self):
        self.closed = True


ROOT = np.array([100., 100., 100., 100., 0., 0., 0.])


def make_env(**kwargs):
    env = RecordingEnv(**kwargs)
    env.reset(seed=0, options={"state": ROOT, "goal_state": audit.DIAGNOSTIC_GOAL})
    return env


def test_distinct_time_major_packet_executes_all_five_commands():
    commands = np.arange(10).reshape(5, 2) / 10
    env = make_env()
    results = contract.execute_packet(env, commands.reshape(10))
    np.testing.assert_array_equal(env.calls, commands)
    assert len(results) == 5
    np.testing.assert_allclose(env.state[:2], ROOT[:2] + commands.sum(axis=0))


@pytest.mark.parametrize("packet", [np.zeros((4, 2)), np.zeros((6, 2)),
                                    np.zeros((2, 5)), np.full((5, 2), np.nan),
                                    np.full((5, 2), 1.01)])
def test_invalid_packets_fail_before_any_physical_step(packet):
    env = make_env()
    with pytest.raises(ValueError):
        contract.execute_packet(env, packet)
    assert env.calls == []


def test_terminal_packet_is_not_padded_or_reused():
    env = make_env(terminal_at=3)
    history = contract.BoundaryHistory(env, root_state=ROOT, goal=audit.DIAGNOSTIC_GOAL)
    with pytest.raises(contract.PacketTerminated):
        history.execute(np.zeros((5, 2)))
    assert len(env.calls) == 3
    assert history.step == 3
    assert history.boundary_steps == (0,)
    with pytest.raises(ValueError, match="partial"):
        history.execute(np.zeros((5, 2)))


def test_live_normalization_uses_explicit_stats_without_extra_div100_and_has_gradient():
    commands = torch.tensor(np.arange(10).reshape(5, 2) / 20, requires_grad=True)
    mean, std = [-.01, .02], [.2, .3]
    tokens = contract.normalize_live_packet(commands, mean=mean, std=std)
    expected = ((commands - torch.tensor(mean)) / torch.tensor(std)).reshape(10)
    torch.testing.assert_close(tokens, expected, rtol=1e-7, atol=1e-7)
    tokens.sum().backward()
    torch.testing.assert_close(commands.grad, torch.tensor([5., 1 / .3]).expand(5, 2).double())
    zero = contract.normalize_live_packet(torch.zeros(2, 5, 2), mean=mean, std=std)
    assert zero.shape == (2, 10)
    assert zero[0, 0].item() == pytest.approx(.05)


def test_packet_position_stats_and_validation():
    mean = np.arange(10) / 100
    got = contract.normalize_live_packet(torch.zeros(5, 2), mean=mean, std=np.ones(10))
    torch.testing.assert_close(got, -torch.tensor(mean).float())
    for kwargs in ({"mean": [0, 0], "std": [0, 1]},
                   {"mean": [0], "std": [1, 1]},
                   {"mean": [float("nan"), 0], "std": [1, 1]}):
        with pytest.raises(ValueError):
            contract.normalize_live_packet(torch.zeros(5, 2), **kwargs)
    with pytest.raises(ValueError):
        contract.normalize_live_packet(torch.zeros(3, 2), mean=[0, 0], std=[1, 1])


def warmup(history):
    for i in range(3):
        history.execute(contract.repeat_command([i / 10, -i / 10]))
    return history.warmup_context()


def test_contiguous_history_keeps_bootstrap_boundary_and_root_anchored_actions():
    history = contract.BoundaryHistory(make_env(), root_state=ROOT, goal=audit.DIAGNOSTIC_GOAL)
    original = warmup(history)
    assert original.frame_steps == (0, 5, 10)
    assert original.decision_step == 15
    prefix = contract.repeat_command([.3, -.3])
    history.execute(prefix)
    observed = contract.update_observed_context(original, history)
    assert observed.frame_steps == (10, 15, 20)  # not (5, 10, 20)
    assert observed.decision_step == 20
    np.testing.assert_array_equal(observed.action_packets, history.packets[1:4])
    assert observed.root_id == original.root_id
    assert history.boundary_steps == (0, 5, 10, 15, 20)
    assert len(history.step_states) == 20
    with pytest.raises(ValueError, match="read-only"):
        observed.frames[0][0, 0, 0] = 1


@pytest.mark.parametrize("fault", ["other_root", "other_history", "skipped", "reordered", "changed_frame"])
def test_observed_update_rejects_noncontiguous_or_unanchored_context(fault):
    from dataclasses import replace

    history = contract.BoundaryHistory(make_env(), root_state=ROOT, goal=audit.DIAGNOSTIC_GOAL)
    context = warmup(history)
    history.execute(np.zeros((5, 2)))
    if fault == "other_root":
        context = replace(context, root_id="another-root")
    elif fault == "other_history":
        context = replace(context, history_id="another-runtime")
    elif fault == "skipped":
        history.execute(np.zeros((5, 2)))
    elif fault == "reordered":
        context = replace(context, frame_steps=(5, 0, 10))
    else:
        context = replace(context, frames=(np.zeros((224, 224, 3), np.uint8),) + context.frames[1:])
    with pytest.raises(ValueError):
        contract.update_observed_context(context, history)


def test_history_detects_out_of_band_physical_advancement():
    env = make_env()
    history = contract.BoundaryHistory(env, root_state=ROOT, goal=audit.DIAGNOSTIC_GOAL)
    env.step(np.ones(2))
    with pytest.raises(ValueError, match="outside"):
        history.execute(np.zeros((5, 2)))


def test_shared_projection_identity_guard_is_required():
    a = contract.projection_identity(np.eye(3))
    b = contract.projection_identity(np.ones((3, 3)))
    assert contract.require_shared_suffix_projection(a, a) == a
    for bad in (b, "", None):
        with pytest.raises(ValueError):
            contract.require_shared_suffix_projection(a, bad)
    assert a != contract.projection_identity(np.eye(3).astype(np.float32))


@pytest.mark.parametrize("receipt", [{}, {"status": "passed"},
    {"schema": "pusht_d2p_p0_preflight.v1", "status": "passed"},
    {"schema": contract.VERSION, "contract": {}, "status": "passed"}])
def test_old_or_incomplete_receipts_are_rejected(receipt):
    with pytest.raises(ValueError):
        contract.validate_receipt(receipt)


def synthetic_hdf5(path, *, invalid=False):
    import h5py

    lengths = [25, 25]
    actions = np.tile([.25, -.125], (50, 1)).astype(np.float32)
    actions[5] = [-.5, .25]  # a distinct, time-major command inside the window
    states = np.empty((50, 7), np.float32)
    for offset in (0, 25):
        states[offset] = ROOT
        for i in range(1, 25):
            states[offset + i] = states[offset + i - 1]
            states[offset + i, :2] += actions[offset + i - 1]
    if invalid:
        actions[26, 0] = 2  # never clip or silently drop the declared episode
    with h5py.File(path, "w") as f:
        f["action"], f["state"] = actions, states
        f["ep_len"], f["ep_offset"] = lengths, [0, 25]
        f["episode_idx"] = np.repeat([0, 1], 25)
        f["step_idx"] = np.tile(np.arange(25), 2)
        f["pixels"] = np.stack([RecordingEnv._image(s) for s in states])


def test_selected_hdf5_rows_are_predeclared_bounded_and_hashed(tmp_path):
    path = tmp_path / "data.h5"
    synthetic_hdf5(path)
    rows, metadata = audit.select_rows(path, (0, 1))
    assert [r["episode"] for r in rows] == [0, 1]
    assert rows[0]["actions"].shape == (20, 2)
    assert metadata["transition_ids"][0] == {"episode": 0, "step": 0, "row": 0, "next_row": 1}
    assert metadata["transition_ids"][-1]["row"] == 44
    assert "full_file_sha256" not in metadata["source"]
    assert len(metadata["selected_hashes"][0]["action"]) == 64
    first_hash = metadata["selected_hashes"][0]["action"]
    import h5py
    with h5py.File(path, "r+") as f:
        f["action"][24] = [1, 1]  # outside selected rows: no full-data binding claim
    _, second = audit.select_rows(path, (0, 1))
    assert second["selected_hashes"][0]["action"] == first_hash


def test_invalid_declared_data_fails_before_any_env_is_created(tmp_path):
    path = tmp_path / "data.h5"
    synthetic_hdf5(path, invalid=True)
    def forbidden():
        pytest.fail("must validate every declared row before simulator creation")
    result = audit.run_audit(path, tmp_path / "audit", (0, 1), env_factory=forbidden)
    assert result["status"] == "failed"
    assert "bounds" in result["error"]


def test_synthetic_audit_is_immutable_and_protocol_precedes_simulator(tmp_path, monkeypatch):
    path, out = tmp_path / "data.h5", tmp_path / "audit"
    synthetic_hdf5(path)
    receipt = tmp_path / "source_receipt.json"
    receipt.write_text('{"external_provenance":"fixture-only"}')
    def factory():
        protocol = json.loads((out / "protocol.json").read_text())
        assert protocol["data"]["episode_indices"] == [0, 1]
        assert (out / "implementation.json").exists()
        return RecordingEnv()
    # Catch accidental full-file materialisation, including the official HDF5.
    monkeypatch.setattr(Path, "read_bytes", lambda _: pytest.fail("read_bytes forbidden"))
    result = audit.run_audit(path, out, (0, 1), env_factory=factory, source_receipt=receipt)
    assert result["status"] == "passed", result.get("error")
    contract.validate_receipt(result)
    assert all(result["gates"].values())
    assert result["scope"] == "physical_contract_only"
    assert result["checkpoint_loaded"] is False
    assert result["normalization"]["checkpoint_training_truth_verified"] is False
    assert result["episodes"][0]["history"]["observed_frame_steps"] == [10, 15, 20]
    assert result["episodes"][0]["dense"]["max_state_errors"][0] == 0
    assert result["episodes"][0]["extra_div100"]["max_state_errors"][0] > 0
    assert result["episodes"][0]["repeated_packet_equal"] is True
    saved = (out / "report.json").read_text()
    with pytest.raises(FileExistsError):
        audit.run_audit(path, out, (0, 1), env_factory=factory)
    assert (out / "report.json").read_text() == saved
    manifest = json.loads((out / "implementation.json").read_text())
    assert all(len(s["sha256"]) == 64 for s in manifest["sources"])
    assert (out / "protocol.json").stat().st_mode & 0o222 == 0


@pytest.mark.parametrize("fault", ["goal", "render"])
def test_goal_metadata_and_render_state_advancement_fail_closed(tmp_path, fault):
    path = tmp_path / "data.h5"
    synthetic_hdf5(path)
    result = audit.run_audit(path, tmp_path / "audit", (0,), env_factory=lambda: RecordingEnv(
        bad_goal=fault == "goal", impure_render=fault == "render"))
    assert result["status"] == "failed"
    with pytest.raises(ValueError):
        contract.validate_receipt(result)


def test_cli_default_selection_is_exact_first16_not_performance_filtered():
    args = audit.parser().parse_args(["--output", "/unused"])
    assert args.episode_indices == tuple(range(16))
    with pytest.raises(ValueError):
        audit.select_rows(Path("not-opened"), (1, 1))
