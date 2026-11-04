from types import SimpleNamespace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.pc_pointmaze_cp12_three_frame_same_action_20261103_codex import packets, mse
from experiments.pointmaze_action_history_unroll_20261103 import unroll_with_action_history


def test_clock_and_packet_normalization():
    actions = (np.arange(34, dtype=np.float32).reshape(17, 2) - 17) / 20
    mean, std = torch.tensor([.2, -.1]), torch.tensor([.5, .25])
    actual = packets(actions, mean, std)
    expected = ((torch.from_numpy(actions[2:17]) - mean) / std).reshape(3, 1, 10)
    assert torch.equal(actual, expected)
    assert torch.equal(actual[2].reshape(5, 2) * std + mean,
                       expected[2].reshape(5, 2) * std + mean)


def test_fail_closed_inputs():
    mean, std = torch.zeros(2), torch.ones(2)
    with pytest.raises(ValueError):
        packets(np.zeros((16, 2)), mean, std)
    bad = np.zeros((17, 2))
    bad[3, 0] = np.nan
    with pytest.raises(ValueError):
        packets(bad, mean, std)
    with pytest.raises(ValueError):
        packets(np.zeros((17, 2)), mean, torch.zeros(2))
    with pytest.raises(ValueError):
        mse(torch.tensor([float("nan")]), torch.tensor([0.]))


def test_observed_action_binding_and_same_final_candidate():
    class Predictor:
        def __init__(self):
            self.calls = []

        def encode_act(self, x):
            return x

        def forward_pred(self, visual, action, proprio):
            self.calls.append((visual.clone(), action.clone(), proprio.clone()))
            return visual + 1, None, proprio + 1

    predictor = Predictor()
    model = SimpleNamespace(model=predictor, proprio_mode="predict_proprio")
    ctx = {"visual": torch.arange(3.).reshape(1, 3, 1, 1, 1, 1),
           "proprio": torch.arange(3.).reshape(1, 3, 1, 1)}
    acts = torch.arange(30.).reshape(3, 1, 10)
    unroll_with_action_history(model, ctx, acts[:2], acts[2:], 3)
    one_ctx = {k: v[:, -1:] for k, v in ctx.items()}
    unroll_with_action_history(model, one_ctx, acts[:0], acts[2:], 1)
    assert torch.equal(predictor.calls[0][1][0], acts[:, 0])
    assert torch.equal(predictor.calls[1][1][0, 0], acts[2, 0])
    assert predictor.calls[0][0].flatten().tolist() == [0., 1., 2.]
    assert predictor.calls[1][0].flatten().tolist() == [2.]


def test_saved_real_run_source_binding_and_mse():
    root = Path(__file__).resolve().parents[1]
    report_path = root / "results/pc_pointmaze_cp12_three_frame_same_action_20261103_CODEX.json"
    if not report_path.exists():
        pytest.skip("real CUDA receipt not present")
    report = json.loads(report_path.read_text())
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    assert digest(report_path) == "e27017d25a31b7d35dc50d0ea9062d10c3541dbdfe4a12f3fc073b0a2af0aad7"
    assert digest(report_path.with_suffix(".npz")) == report["arrays_sha256"]
    assert report["clock"]["observations"] == [2, 7, 12]
    assert report["clock"]["action_ranges"] == [[2, 7], [7, 12], [12, 17]]
    assert report["clock"]["prediction_target"] == 17
    assert report["counts"]["branches"] == 32 and report["counts"]["physical_steps"] == 0
    assert report["model_before"] == report["model_after"]
    assert report["one_saved_max_abs"] == 0.0
    source = root / "results/pc_causal_mixed_rejected_physical_20261103_DEV01.npz"
    assert digest(source) == report["files_sha256"][str(source)]
    mean = np.array(report["action_mean"], dtype=np.float32)
    std = np.array(report["action_std"], dtype=np.float32)
    with np.load(report_path.with_suffix(".npz"), allow_pickle=False) as data, np.load(source, allow_pickle=False) as original:
        for row in report["rows"]:
            name = f"r{row['root']}_a{('raw', 'spectral').index(row['arm'])}_p{row['candidate']}"
            actions = data[name + "_raw_actions"]
            np.testing.assert_array_equal(actions, original[name + "_actions"])
            expected = ((actions[2:17] - mean) / std).reshape(3, 1, 10)
            np.testing.assert_array_equal(data[name + "_normalized_packets"], expected)
            for kind in ("visual", "proprio"):
                target = data[f"{name}_target_{kind}"]
                for arm in ("one", "three"):
                    predicted = data[f"{name}_{arm}_{kind}"]
                    error = np.square(predicted - target).mean()
                    np.testing.assert_allclose(error, row[f"{arm}_{kind}_mse"], rtol=2e-6, atol=1e-10)
