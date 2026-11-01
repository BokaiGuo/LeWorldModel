"""Small algebra/context fixtures, not CPU efficacy experiments."""
import numpy as np
import pytest
import torch

from experiments.spectral_reachability import reacher_d2p_action_words as words


def test_word_support_sign_order_and_balanced_commands():
    response = torch.diag(torch.tensor([3., 2., 1.], dtype=torch.float64))
    bank, receipt = words.build_words(response)
    commands = bank.tanh()
    assert bank.shape == (17, 4, 3)
    assert not torch.count_nonzero(bank[0])
    assert commands.abs().max() <= .5
    torch.testing.assert_close(commands.sum(dim=1), torch.zeros_like(commands[:, 0]), rtol=0, atol=1e-15)
    for left in range(1, 17, 2):
        right = left + 1
        torch.testing.assert_close(commands[left][[1, 0, 3, 2]], commands[right])
        assert receipt["labels"][left]["signs"] == receipt["labels"][right]["signs"]
    directions = torch.tensor(receipt["canonical_unit_command_directions"])
    for direction in directions:
        assert direction[direction.abs().argmax()] > 0


def test_word_generation_deterministic():
    response = torch.tensor([[1., -2., .5], [.1, .2, .3]], dtype=torch.float64)
    first, receipt = words.build_words(response)
    second, other = words.build_words(response)
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    assert receipt == other


@pytest.mark.parametrize("response", [torch.zeros(2, 3), torch.ones(2, 3), torch.ones(1, 4), torch.full((2, 3), float("nan"))])
def test_invalid_singular_directions_rejected(response):
    with pytest.raises(ValueError):
        words.build_words(response)


def test_four_packets_produce_four_separate_causal_updates(monkeypatch):
    calls = []
    def advance(env, *, context, raw_action_block, action_block, verified_root_frame):
        calls.append((context["root_frame"].copy(), raw_action_block.copy(), verified_root_frame.copy()))
        frame = context["root_frame"] + 1
        return {"root_frame": frame, "execution_snapshot": {}, "root_rgb_sha256": str(len(calls)),
            "history_rgb_sha256": [str(len(calls))], "advanced_action_sha256": str(raw_action_block[0, 0])}
    monkeypatch.setattr(words.execution, "advance_executed_context", advance)
    frame = np.zeros((2, 2, 3), np.uint8)
    raw = np.arange(40, dtype=np.float32).reshape(20, 2)
    advanced, boundaries = words.advance_program(None, {"root_frame": frame}, raw, verified_root_frame=frame)
    assert len(calls) == 4
    assert [cell["raw_steps_executed"] for cell in boundaries] == [5, 10, 15, 20]
    for i, (before, packet, verified) in enumerate(calls):
        np.testing.assert_array_equal(packet, raw[i * 5:(i + 1) * 5])
        np.testing.assert_array_equal(before, verified)
        assert (before == i).all()
    assert (advanced["root_frame"] == 4).all()


@pytest.mark.parametrize("raw", [np.zeros((19, 2)), np.zeros((0, 2)), np.zeros(20), np.full((20, 2), float("nan"))])
def test_partial_or_invalid_program_rejected(raw):
    with pytest.raises(ValueError):
        words.advance_program(None, {}, raw, verified_root_frame=None)
