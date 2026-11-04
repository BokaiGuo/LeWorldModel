"""Independent checks of saved planner/feedback bindings and CEM arithmetic."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/pc_pointmaze_shadow_closed_loop_dev_20261103_CODEX.json"
ARCHIVE = ROOT / "results/a0_pointmaze_archive_bank_alignment_20261102_DEV01.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_result_sources_actions_rgb_and_cem_binding():
    doc = json.loads(OUT.read_text())
    for path, digest in doc["dependencies"].items():
        assert sha(path) == digest
    journal = OUT.with_name(OUT.stem+"_events.jsonl")
    assert sha(journal) == doc["journal_sha256"]
    events = [json.loads(line) for line in journal.read_text().splitlines()]
    assert sum(e["event"] == "predict_attempt" for e in events) == 40
    assert sum(e["event"] == "predict_returned" for e in events) == 40
    assert sum(e["event"] == "command_attempt" for e in events) == 34
    assert sum(e["event"] == "command_returned" for e in events) == 34
    assert doc["model_before"] == doc["model_after"]
    assert doc["counts"] == {"goal_encodes": 1, "feedback_encodes": 12, "predict_samples": 500,
                              "predict_batches": 40, "shadow_resets": 2, "legacy_resets": 2,
                              "shadow_commands": 34, "legacy_commands": 34}
    archive = json.loads(ARCHIVE.read_text())["provenance"]
    mean = torch.tensor(archive["action_mean"])
    std = torch.tensor(archive["action_std"])
    low, high, prior = ((-1-mean)/std).repeat(5), ((1-mean)/std).repeat(5), ((-mean)/std).repeat(5)
    for row in doc["rows"]:
        assert sha(row["trajectory_path"]) == row["trajectory_sha256"]
        with np.load(row["trajectory_path"], allow_pickle=False) as traj:
            assert traj["actions"].shape == (17, 2)
            assert traj["shadow_frames"].shape == traj["legacy_frames"].shape == (18, 224, 224, 3)
            for i, step in enumerate(row["steps"]):
                assert sha(step["decision_path"]) == step["decision_sha256"]
                with np.load(step["decision_path"], allow_pickle=False) as decision:
                    np.testing.assert_array_equal(decision["previous_rgb"], traj["shadow_frames"][11+i])
                    np.testing.assert_array_equal(decision["current_rgb"], traj["shadow_frames"][12+i])
                    np.testing.assert_array_equal(decision["plan_raw"][0], traj["actions"][12+i])
                    np.testing.assert_array_equal(decision["plan_raw"][0], step["action"])
                    generator = torch.Generator(device="cpu").manual_seed(step["seed"])
                    noise = torch.randn((3, 16, 10), generator=generator)
                    np.testing.assert_array_equal(noise.numpy(), decision["noise"])
                    expected_mean, expected_std = prior.clone(), torch.full_like(prior, 1/3)
                    for j in range(3):
                        candidates = (expected_mean + noise[j]*expected_std).clamp(low, high)
                        candidates[0] = expected_mean.clamp(low, high)
                        np.testing.assert_allclose(candidates.numpy(), decision[f"iteration{j}_candidates"],
                                                   rtol=1e-5, atol=2e-6)
                        scores = torch.from_numpy(decision[f"iteration{j}_scores"])
                        elite = torch.argsort(scores, stable=True)[:4]
                        np.testing.assert_array_equal(elite.numpy(), decision[f"iteration{j}_elite"])
                        selected = candidates[elite]
                        expected_mean = selected.mean(0)
                        expected_std = selected.std(0, correction=1).clamp_min(1e-5)
                        np.testing.assert_allclose(expected_mean.numpy(), decision[f"iteration{j}_mean"],
                                                   rtol=1e-5, atol=2e-6)
                        np.testing.assert_allclose(expected_std.numpy(), decision[f"iteration{j}_std"],
                                                   rtol=1e-5, atol=2e-6)
                    final = decision["final_candidates"]
                    np.testing.assert_allclose(final[0], prior.numpy(), rtol=1e-5, atol=2e-6)
                    np.testing.assert_allclose(final[1], expected_mean.numpy(), rtol=1e-5, atol=2e-6)
                    winner = int(decision["final_scores"][1] < decision["final_scores"][0])
                    assert winner == int(decision["winner"]) == step["winner"]
                    np.testing.assert_array_equal(decision["chosen"], final[winner])
                    np.testing.assert_allclose(decision["chosen"].reshape(5, 2)*std.numpy()+mean.numpy(),
                                               decision["plan_raw"], rtol=1e-5, atol=2e-6)
                    assert float(decision["realized_rgb_score"]) == step["rgb_score_before"]
