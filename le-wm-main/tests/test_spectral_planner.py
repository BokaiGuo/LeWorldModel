import numpy as np
import torch

from experiments.spectral_planner import build_spectral_planner_state, wrap_model_with_spectral_cost


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.zeros(()))

    def encode(self, info):
        pixels = info["pixels"].float()
        if pixels.ndim == 3:
            emb = pixels
        elif pixels.ndim == 4:
            emb = pixels.mean(dim=-1)
        else:
            emb = pixels.reshape(*pixels.shape[:2], -1)
        out = dict(info)
        out["emb"] = emb
        return out

    def rollout(self, info, action_sequence, history_size=3):
        emb = info["emb"]
        b, s = action_sequence.shape[:2]
        init = emb[:, :, -1:]
        steps = action_sequence[..., : emb.size(-1)]
        pred = init + torch.cumsum(steps, dim=2)
        info["predicted_emb"] = pred
        return info

    def criterion(self, info_dict):
        diff = info_dict["predicted_emb"][..., -1, :] - info_dict["goal_emb"][..., -1, :].detach()
        return diff.square().sum(dim=-1)

    def get_cost(self, info_dict, action_candidates):
        goal = {"pixels": info_dict["goal"][:, 0]}
        goal = self.encode(goal)
        init = {"pixels": info_dict["pixels"][:, 0], "emb": self.encode({"pixels": info_dict["pixels"][:, 0]})["emb"].unsqueeze(1).expand(-1, action_candidates.size(1), -1, -1)}
        init["goal_emb"] = goal["emb"]
        init = self.rollout(init, action_candidates)
        return self.criterion(init)


def test_build_spectral_planner_state_from_latent_npz(tmp_path):
    rng = np.random.default_rng(0)
    latents = rng.normal(size=(128, 4)).astype("float32")
    latents[:, 0] *= 5
    dynamics = np.zeros((128, 4), dtype="float32")
    dynamics[:, 0] = rng.normal(size=128)
    nuisance = np.zeros((128, 4), dtype="float32")
    nuisance[:, -1] = rng.normal(size=128)
    path = tmp_path / "latents.npz"
    np.savez_compressed(path, latents=latents, dynamics=dynamics, planning=dynamics, nuisance=nuisance)

    state = build_spectral_planner_state(
        type("Cfg", (), {
            "latents_path": str(path),
            "max_samples": 64,
            "seed": 1,
            "top_k": 2,
            "weight_mode": "reachability_controllability_nuisance",
            "normalize_weights": True,
            "vanilla_blend": 0.0,
        })()
    )

    assert state["num_loaded_samples"] == 64
    assert state["weights"][0] > state["weights"][-1]
    assert state["weight_summary"]["num_nonzero"] == 2


def test_spectral_cost_model_changes_online_candidate_ranking(tmp_path):
    latents = np.eye(4, dtype="float32").repeat(16, axis=0)
    dynamics = np.zeros_like(latents)
    dynamics[:, 0] = 1.0
    nuisance = np.zeros_like(latents)
    nuisance[:, 1] = 1.0
    path = tmp_path / "latents.npz"
    np.savez_compressed(path, latents=latents, dynamics=dynamics, planning=dynamics, nuisance=nuisance)

    model = wrap_model_with_spectral_cost(DummyModel(), str(path), max_samples=0)
    info = {
        "pixels": torch.zeros(1, 1, 1, 4),
        "goal": torch.tensor([[[[1.0, 1.0, 0.0, 0.0]]]]),
        "action": torch.zeros(1, 1, 1, 4),
    }
    candidates = torch.tensor([[[[1.0, 0.0, 0.0, 0.0]], [[0.0, 1.0, 0.0, 0.0]]]])

    costs = model.get_cost(info, candidates)

    assert costs.shape == (1, 2)
    assert costs[0, 0] < costs[0, 1]
