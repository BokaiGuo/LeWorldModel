import copy
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from experiments.spectral_reachability import codex_p1_two_stage_feedback_20261030_v1 as bridge


def fixture():
    pixels = np.zeros((224, 224, 3), dtype=np.uint8)
    pixels[0, 0] = (1, 2, 3)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    packet = np.linspace(-1, 1, 10, dtype="<f4").reshape(5, 2)
    sha = hashlib.sha256(pixels.tobytes()).hexdigest()
    receipts = [dict(row=2, step=i+1, action_f32_hex=packet[i].tobytes().hex(),
                     terminated=False, truncated=False, pixel_sha256=sha) for i in range(5)]
    return packet, receipts, buffer.getvalue()


def test_feedback_requires_five_actual_ordered_actions_and_last_pixels():
    packet, receipts, png = fixture()
    assert bridge.verify_feedback(2, packet, receipts, png).shape == (224, 224, 3)
    cases = []
    cases.append(receipts[:-1])
    cases.append(list(reversed(receipts)))
    for field, value in (("row", 3), ("step", 3), ("action_f32_hex", "00"*8),
                         ("terminated", True), ("truncated", True), ("pixel_sha256", "0"*64)):
        bad = copy.deepcopy(receipts)
        bad[-1][field] = value
        cases.append(bad)
    bad = copy.deepcopy(receipts)
    bad[0]["future_state"] = [1]
    cases.append(bad)
    for bad in cases:
        with pytest.raises(ValueError):
            bridge.verify_feedback(2, packet, bad, png)


def test_second_context_uses_real_latent_and_executed_packet(monkeypatch):
    packet, _, _ = fixture()
    states = torch.arange(576, dtype=torch.float32).reshape(1, 3, 192)
    actions = torch.arange(20, dtype=torch.float32).reshape(1, 2, 10)
    mean, std = torch.tensor([0.1, -0.2]), torch.tensor([2.0, 4.0])
    goal = torch.zeros((1, 192))
    monkeypatch.setattr(bridge.common, "context", lambda *args, **kw: (states, actions, goal, mean, std))
    actual = torch.full((192,), 7.0)
    new_states, new_actions, *_ = bridge.second_context({}, torch.zeros(192), actual, packet)
    assert torch.equal(new_states[:, :2], states[:, 1:])
    assert torch.equal(new_states[0, 2], actual)
    assert torch.equal(new_actions[:, 0], actions[:, 1])
    assert torch.equal(new_actions[0, 1], ((torch.from_numpy(packet)-mean)/std).reshape(10))
    with pytest.raises(ValueError):
        bridge.second_context({}, torch.zeros(192), torch.full((192,), float("nan")), packet)
    with pytest.raises(ValueError):
        bridge.second_context({}, torch.zeros(192), actual, np.zeros((5, 2), dtype=np.float64))


def test_cuda_only_even_for_valid_image():
    with pytest.raises(ValueError, match="host CUDA"):
        bridge.encode_fresh(np.zeros((224, 224, 3), np.uint8), None, None, None, "cpu")


def test_fixed_blind_roster_and_real_history():
    path = Path(__file__).resolve().parents[1] / "results/codex_p1_three_prefix_blind_inputs_20261030_v1.json"
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == "070cb2fccffc79a1cf4a9480396f6b2d5eec4ff3b72928c8704fda8a23095cc8"
    rows = json.loads(data)["rows"]
    assert len(rows) == 24
    packet, _, _ = fixture()
    for index, row in enumerate(rows):
        assert row["index"] == index
        cp5, cp10 = torch.full((192,), 5.0), torch.full((192,), 10.0)
        first = bridge.common.context(row, cp5, device="cpu")
        second = bridge.second_context(row, cp5, cp10, packet)
        assert torch.equal(second[0][:, :2], first[0][:, 1:])
        assert torch.equal(second[0][0, 2], cp10)
        assert torch.equal(second[1][:, 0], first[1][:, 1])
        assert torch.equal(second[1][0, 1],
                           ((torch.from_numpy(packet) - first[3]) / first[4]).reshape(10))
