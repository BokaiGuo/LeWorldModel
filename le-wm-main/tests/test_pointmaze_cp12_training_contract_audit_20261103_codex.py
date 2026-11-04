"""Read-only, source-bound CPU audit of the executed cp12 artifacts."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from tensordict import TensorDict

from app.plan_common.datasets import DATA_STATS
from app.plan_common.datasets.point_maze_dset import PointMazeDataset
from app.plan_common.datasets.traj_dset import TrajSlicerDataset
from app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds import EncPredWM


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATA = ROOT / "data/point_maze_tail1800_1899"
CHECKPOINT = ROOT / "data/jepa_wm_checkpoints/jepa_wm_pointmaze.pth.tar"
EXPERIMENTS = ROOT / "experiments"
CP12 = ("replan", "controls", "imagined")
CP12_HASHES = {
    "refresh": "a2a967567427e2cf0d94a222547697505d7e595adea7a932b0f7ce9ad5bea1a3",
    "replan": "882cf9d46da6901cf8ca1c7f70ed942ea22349e01a03d380beccf9f5b50e14b9",
    "controls": "f03d384a7e18a345219a8e151d12d31e62e1ac0f57461671887656d99e566491",
    "imagined": "2f66c47b7b9db591f4992c35d09036e8ed0132f30a37518f4a3dbfd790ef27af",
}
EXPECTED_SOURCE = {
    "pc_causal_mixed_rejected_physical_20261103_DEV01":
        "07ada31f9a58c3472fd84778f749c2df6cfb14db6442531954e5e7003ab587f8",
    "pc_causal_mixed_prefix_search_20261103_DEV01":
        "f9d8f3014b44a95dfd14b0afd3d49c0b2d4b776a8112d9fa20370cb7f52fe368",
}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@pytest.fixture(scope="module")
def source_documents():
    docs = {}
    for name, expected in EXPECTED_SOURCE.items():
        path = RESULTS / (name + ".json")
        assert sha(path) == expected
        doc = json.loads(path.read_text())
        assert sha(path.with_suffix(".npz")) == doc["arrays_sha256"]
        docs[name] = doc
    for mode in ("refresh", *CP12):
        name = f"pc_causal_mixed_cp12_{mode}_20261103_DEV01"
        path = RESULTS / (name + ".json")
        assert sha(path) == CP12_HASHES[mode]
        doc = json.loads(path.read_text())
        assert sha(path.with_suffix(".npz")) == doc["arrays_sha256"]
        assert sha(EXPERIMENTS / f"pc_causal_mixed_cp12_{mode}_20261103.py") == doc["script_sha256"]
        assert sha(CHECKPOINT) == doc["checkpoint_sha256"]
        assert doc["model_before"] == doc["model_after"]
        docs[name] = doc
    replan = docs["pc_causal_mixed_cp12_replan_20261103_DEV01"]
    assert sha(EXPERIMENTS / "spectral_reachability/pc_bounded_packet_cem_20261103.py") == replan["cem_kernel_sha256"]
    return docs


@pytest.fixture(scope="module")
def dataset():
    return PointMazeDataset(data_path=str(DATA), normalize_action=True)


def test_live_loader_stats_match_published_registry_and_saved_receipt(dataset, source_documents):
    receipt = RESULTS / "a0_pointmaze_archive_bank_alignment_20261102_DEV01.json"
    assert sha(receipt) == "93cdc7067ca623f1eb80439d1e41ff18b3be4468d8befdc3d9c6cecad26d3291"
    saved = json.loads(receipt.read_text())
    provenance = saved["provenance"]
    for filename, key in (("actions.pth", "actions_sha256"), ("states.pth", "states_sha256"),
                          ("seq_lengths.pth", "sequence_lengths_sha256")):
        assert sha(DATA / filename) == provenance[key]
    assert sha(CHECKPOINT) == provenance["checkpoint_sha256"]
    published = DATA_STATS["pointmaze"]
    for name in ("action_mean", "action_std", "proprio_mean", "proprio_std"):
        assert torch.equal(getattr(dataset, name), torch.tensor(published[name]))
    assert dataset.action_mean.tolist() == provenance["action_mean"]
    assert dataset.action_std.tolist() == provenance["action_std"]
    assert len(dataset) == 2000 and torch.isfinite(dataset.actions).all()


def test_inspected_training_config_and_local_code_bindings(source_documents):
    external = ROOT / "external/jepa-wms"
    expected = {
        "app/plan_common/datasets/point_maze_dset.py": "ea78143e69840d519a944e3d470301cc21aa5bc540b5397de9207ca8e1d716f7",
        "app/plan_common/datasets/traj_dset.py": "865f31178caf99c800b272148f876ab85d2f82ee97e2d7f18946459a80e8c893",
        "app/vjepa_wm/modelcustom/simu_env_planning/vit_enc_preds.py": "105a6768050e4b34c95de8d7d0cc0c7ba07b69d364acf0649ba2625bacce907b",
        "configs/vjepa_wm/mz_sweep/mz_4f_fsk5_ask1_r224_vjtranoaug_predAdaLN_ftprop_depth6_repro_2roll_save_2n.yaml": "c666b4251f72f56c08c69ab223cfe645625f39705f1c84285320edc98c205a66",
    }
    for path, expected_hash in expected.items():
        assert sha(external / path) == expected_hash
    config = yaml.safe_load((external / list(expected)[-1]).read_text())
    custom = config["data"]["custom"]
    assert (custom["frameskip"], custom["action_skip"], custom["num_hist"], custom["num_pred"]) == (5, 1, 3, 1)
    assert custom["normalize_action"] is True
    assert config["model"]["rollout_cfg"]["ctxt_window_train_rollout"] == 3


class IndexedTrajectory:
    action_dim = 2
    proprio_dim = state_dim = 1

    def __len__(self):
        return 1

    def get_seq_length(self, index):
        return 30

    def __getitem__(self, index):
        t = torch.arange(30, dtype=torch.float32)
        obs = {"visual": t[:, None], "proprio": t[:, None]}
        return obs, torch.stack((t, 100 + t), dim=1), t[:, None], None, {}


def test_actual_training_slicer_outgoing_time_and_coordinate_order():
    slicer = TrajSlicerDataset(IndexedTrajectory(), num_frames=4, frameskip=5,
                              action_skip=1, process_actions="concat",
                              generator=torch.Generator().manual_seed(3))
    i = next(i for i, triple in enumerate(slicer.slices) if triple == (0, 3, 23))
    obs, actions, state, _ = slicer[i]
    assert obs["visual"].flatten().tolist() == [3., 8., 13., 18.]
    assert state.flatten().tolist() == [3., 8., 13., 18.]
    for packet in range(4):
        expected = torch.tensor([[float(t), float(t + 100)] for t in range(3 + 5 * packet, 8 + 5 * packet)])
        assert torch.equal(actions[packet], expected.reshape(10))


class FakePredictor:
    def __init__(self):
        self.calls = []

    def encode_act(self, actions):
        return actions

    def forward_pred(self, video, action, proprio, debug=False):
        self.calls.append((video.clone(), action.clone(), proprio.clone()))
        return video + 1, None, proprio + 10


def test_actual_wrapper_one_frame_uses_current_action_and_predicted_feedback():
    predictor = FakePredictor()
    wrapper = SimpleNamespace(model=predictor, ctxt_window=1, proprio_mode="predict_proprio")
    context = TensorDict({"visual": torch.zeros(1, 1, 1, 1, 1, 1),
                          "proprio": torch.zeros(1, 1, 1, 1)}, batch_size=[1, 1])
    actions = torch.arange(20, dtype=torch.float32).reshape(2, 1, 10)
    out = EncPredWM.unroll(wrapper, context, actions)
    assert len(predictor.calls) == 2
    assert all(call[0].shape[1] == call[1].shape[1] == call[2].shape[1] == 1
               for call in predictor.calls)
    assert torch.equal(predictor.calls[0][1][0, 0], actions[0, 0])
    assert torch.equal(predictor.calls[1][1][0, 0], actions[1, 0])
    assert torch.equal(predictor.calls[1][2], predictor.calls[0][2] + 10)
    assert out["proprio"].flatten().tolist() == [0., 10., 20.]


@pytest.mark.parametrize("mode,expected_branches", [("replan", 32), ("controls", 8), ("imagined", 40)])
def test_executed_prefix_tail_packing_and_elite_mean(dataset, source_documents, mode, expected_branches):
    name = f"pc_causal_mixed_cp12_{mode}_20261103_DEV01"
    candidate_source = RESULTS / "pc_causal_mixed_rejected_physical_20261103_DEV01.npz"
    control_source = RESULTS / "pc_causal_mixed_prefix_search_20261103_DEV01.npz"
    with np.load(RESULTS / (name + ".npz"), allow_pickle=False) as actual, \
            np.load(candidate_source, allow_pickle=False) as candidates, \
            np.load(control_source, allow_pickle=False) as controls:
        names = [key[:-len("_normalized_tail")] for key in actual.files if key.endswith("_normalized_tail")]
        assert len(names) == expected_branches
        for key in names:
            old = candidates if "_a0_" in key or "_a1_" in key else controls
            assert np.array_equal(actual[key + "_actions"][:12], old[key + "_actions"][:12])
            assert np.array_equal(actual[key + "_history"][:13], old[key + "_history"][:13])
            final = actual[key + "_feedback3_candidates"]
            final_scores = actual[key + "_feedback3_scores"]
            assert final.shape == (1, 2, 1, 10) and final_scores.shape == (2,)
            chosen_index = int(final_scores[1] < final_scores[0])
            normalized = actual[key + "_normalized_tail"]
            assert np.array_equal(normalized, final[:, chosen_index])
            decoded = (torch.from_numpy(normalized.reshape(5, 2)) * dataset.action_std + dataset.action_mean)
            expected = np.clip(decoded.numpy(), -1, 1).astype(np.float32)
            np.testing.assert_array_equal(actual[key + "_actions"][12:], expected)
            # Test what can be observed: the next scored population's incumbent
            # equals the mean of the previous scored, projected elite vectors.
            for j in (0, 1, 2):
                proposals = actual[f"{key}_feedback{j}_candidates"][0, :, 0]
                scores = actual[f"{key}_feedback{j}_scores"]
                assert len(np.unique(scores)) == len(scores)
                elite = torch.topk(torch.from_numpy(scores), 4, largest=False).indices.numpy()
                next_mean = actual[f"{key}_feedback{j+1}_candidates"][0, int(j == 2), 0]
                np.testing.assert_array_equal(next_mean, proposals[elite].mean(axis=0))
