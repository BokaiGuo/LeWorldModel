import numpy as np
import pytest
import torch
import copy
import json

from experiments.spectral_reachability import reacher_cem_context_sensitivity_v2_20261102 as run
from experiments.spectral_reachability import reacher_cem_context_sensitivity_saved_20261102 as saved


def test_saved_frames_are_not_shifted_twice():
    frames = [np.full((2, 2, 3), value, np.uint8) for value in (11, 22, 33)]
    goal = np.zeros((2, 2, 3), np.uint8)
    history = np.zeros((3, 10), np.float32)

    class Processor:
        def _prepare_info(self, raw):
            return {key: torch.from_numpy(value) for key, value in raw.items()}

    pixels, _, _ = run.prepare_saved_frames(run.setup.p0._prepare, Processor(),
                                             frames, goal, history, torch.device("cpu"))
    assert pixels[0, :, 0, 0, 0].tolist() == [11, 22, 33]


def test_bad_frame_count_rejected():
    with pytest.raises(ValueError, match="frame count"):
        run.prepare_saved_frames(None, None, [1, 2], None, None, None)


@pytest.fixture(scope="module")
def recorded():
    root = run.ROOT / "results"
    plan = json.loads((root / "reacher_cem_context_sensitivity_PLAN_20261102_v2.json").read_text())
    events = run.saved.verify_journal(root / "reacher_cem_context_sensitivity_RUN_20261102_v2/events.jsonl")
    return plan, events, "4ec03555ad0506f8a45861e83bb6b9563a902485d307e62771ea9204dba8296a"


def test_saved_results(recorded):
    result = saved.validate(*recorded)
    assert result["original_predicted_reproduced"] == 4
    assert result["material_differences"] == {"16x2": 0, "32x3": 6}
    assert result["P1_admission"] is False


@pytest.mark.parametrize("mutation", ["order", "suffix", "cost", "history"])
def test_rejects_mutated_records(recorded, mutation):
    plan, events, digest = copy.deepcopy(recorded)
    if mutation == "order":
        events[1], events[2] = events[2], events[1]
    elif mutation == "suffix":
        events[1]["contexts"]["predicted"]["suffix_parameters"][0][0] += 0.1
    elif mutation == "cost":
        events[1]["contexts"]["predicted"]["receipt"]["best_cost"] += 0.1
    else:
        events[1]["history_max_abs_error"] = 0.1
    with pytest.raises(ValueError):
        saved.validate(plan, events, digest)
