import torch
import pytest

from experiments.pc_shadow_multistart_dev_20261103_codex import decide, second_seed


def test_both_searches_scored_and_better_result_selected():
    noise = torch.randn((3, 16, 10), generator=torch.Generator().manual_seed(9))
    calls, events = [], []
    def evaluate(x):
        calls.append(len(x))
        return (x - .35).square().sum(dim=1)
    result, structure, counts = decide("baseline", evaluate, None, torch.zeros(2),
                                       torch.ones(2), noise, event=lambda k, **v: events.append((k, v)))
    assert calls == [16, 16, 16, 2] * 2
    assert result["cem_counts"] == {"samples": 100, "batches": 8}
    assert counts["svd_calls"] == 0 and structure is None
    detail = events[0][1]
    best = min(min(item["final_scores"]) for item in detail["starts"])
    assert min(result["final_scores"]).item() == pytest.approx(best)
    assert len(detail["starts"][0]["trace"]) == len(detail["starts"][1]["trace"]) == 3


def test_seed_is_repeatable_and_arm_validation_fails_closed():
    noise = torch.zeros((3, 16, 10))
    assert second_seed(noise) == second_seed(noise.clone())
    assert second_seed(noise) != second_seed(noise + .1)
    with pytest.raises(ValueError):
        decide("spectral", None, None, torch.zeros(2), torch.ones(2), noise)
