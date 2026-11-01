"""Check packet/time alignment independently of neural inference."""
import numpy as np
import torch

from experiments.spectral_reachability import codex_p1_cp5_vector_replay_20261030_main as fixed
from experiments.spectral_reachability import codex_p1_two_stage_feedback_20261030_v1 as feedback


def test_real_roster_feedback_state_and_action_alignment():
    bundle, entries = fixed.inputs()
    assert len(bundle["rows"]) == 24 and len(entries) == 48
    for entry in entries:
        row = bundle["rows"][entry["index"]]
        event, _ = fixed.gap.read(fixed.Path(entry["event_path"]), entry["event_sha256"])
        packet = np.asarray(event["result"]["actions"], dtype="<f4")[0, 0]
        z5 = torch.arange(192, dtype=torch.float32)
        z10 = z5 + 1000
        states, history, goal, mean, std = feedback.second_context(row, z5, z10, packet)
        root = torch.tensor(row["root_latent_history"], dtype=torch.float32)
        prefix = np.frombuffer(bytes.fromhex(row["prefix_raw_f32_hex"]), dtype="<f4").copy().reshape(5, 2)
        expected_prefix = ((torch.from_numpy(prefix) - mean) / std).reshape(10)
        expected_packet = ((torch.from_numpy(packet.copy()) - mean) / std).reshape(10)
        assert torch.equal(states[0, 0], root[2])
        assert torch.equal(states[0, 1], z5)
        assert torch.equal(states[0, 2], z10)
        assert torch.equal(history[0, 0], expected_prefix)
        assert torch.equal(history[0, 1], expected_packet)
        assert torch.equal(goal[0], torch.tensor(row["goal_latent"], dtype=torch.float32))


def test_context_rejects_out_of_range_packet():
    bundle, _ = fixed.inputs()
    bad = np.zeros((5, 2), dtype="<f4")
    bad[2, 1] = 1.01
    z = torch.zeros(192, dtype=torch.float32)
    try:
        feedback.second_context(bundle["rows"][0], z, z, bad)
    except ValueError as error:
        assert "raw packet ABI" in str(error)
    else:
        raise AssertionError("invalid executed packet was accepted")
