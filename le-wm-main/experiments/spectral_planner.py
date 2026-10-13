"""Online spectral planner cost wrappers.

These utilities turn the offline spectral audit into a planner-in-the-loop
intervention: CEM/gradient solvers still optimize action sequences online, but
the LeWM terminal latent distance is evaluated in an eigenspace-weighted metric
estimated from prior latent audit arrays.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.directional_metrics import EPS, compute_eigenspace
from experiments.empirical_spectral_audit.metrics import spectral_planner_weights
from experiments.io_utils import to_numpy


@dataclass(frozen=True)
class SpectralPlannerConfig:
    latents_path: str
    max_samples: int = 200_000
    seed: int = 3072
    top_k: int = 0
    weight_mode: str = "reachability_controllability_nuisance"
    normalize_weights: bool = True
    vanilla_blend: float = 0.0


def _sample_rows(n: int, max_samples: int, seed: int) -> np.ndarray | slice:
    if max_samples and n > int(max_samples):
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(n, size=int(max_samples), replace=False))
    return slice(None)


def _load_required(data: np.lib.npyio.NpzFile, key: str, rows: np.ndarray | slice) -> np.ndarray | None:
    if key not in data.files:
        return None
    return np.asarray(data[key][rows], dtype=np.float32)


def build_spectral_planner_state(config: SpectralPlannerConfig) -> dict[str, Any]:
    """Estimate an eigenspace and rank weights from a latent audit NPZ."""

    path = Path(config.latents_path)
    with np.load(path) as data:
        if "latents" not in data.files:
            raise ValueError(f"{path} does not contain required array 'latents'")
        n = int(data["latents"].shape[0])
        rows = _sample_rows(n, config.max_samples, config.seed)
        latents = _load_required(data, "latents", rows)
        dynamics = _load_required(data, "dynamics", rows)
        planning = _load_required(data, "planning", rows)
        nuisance = _load_required(data, "nuisance", rows)

    eig = compute_eigenspace(latents)
    eigenvectors = to_numpy(eig.eigenvectors).astype(np.float32, copy=False)
    if config.weight_mode == "uniform":
        weights = np.ones(eigenvectors.shape[1], dtype=np.float32)
    else:
        reach = planning if planning is not None else dynamics
        ctrl = dynamics if dynamics is not None else reach
        if reach is None or ctrl is None:
            raise ValueError("spectral planner weights require planning or dynamics tangents")
        weights = spectral_planner_weights(eigenvectors, reach, ctrl, nuisance).astype(np.float32, copy=False)
    top_k = int(config.top_k)
    if top_k > 0:
        mask = np.zeros_like(weights)
        mask[: min(top_k, weights.size)] = 1.0
        weights = weights * mask
    if config.normalize_weights:
        positive = weights[weights > 0]
        scale = float(positive.mean()) if positive.size else 1.0
        weights = weights / max(scale, EPS)
    return {
        "config": asdict(config) if is_dataclass(config) else dict(vars(config)),
        "num_source_samples": n,
        "num_loaded_samples": int(latents.shape[0]),
        "eigenvectors": eigenvectors,
        "weights": weights,
        "weight_summary": {
            "min": float(weights.min(initial=0.0)),
            "max": float(weights.max(initial=0.0)),
            "mean": float(weights.mean()),
            "num_nonzero": int((weights > 0).sum()),
        },
    }


class SpectralCostModel(torch.nn.Module):
    """Wrap a LeWM model and replace its terminal latent cost with d_spec."""

    def __init__(self, base_model: torch.nn.Module, state: dict[str, Any], vanilla_blend: float = 0.0) -> None:
        super().__init__()
        self.base_model = base_model
        self.vanilla_blend = float(vanilla_blend)
        self.register_buffer("spectral_eigenvectors", torch.as_tensor(state["eigenvectors"], dtype=torch.float32))
        self.register_buffer("spectral_weights", torch.as_tensor(state["weights"], dtype=torch.float32))
        self.state_metadata = {k: v for k, v in state.items() if k not in {"eigenvectors", "weights"}}

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            base = super().__getattr__("base_model")
            return getattr(base, name)

    def _spectral_terminal_cost(self, predicted_emb: torch.Tensor, goal_emb: torch.Tensor) -> torch.Tensor:
        if goal_emb.ndim == predicted_emb.ndim - 1:
            goal_emb = goal_emb.unsqueeze(1)
        goal = goal_emb[..., -1:, :].expand_as(predicted_emb)
        diff = predicted_emb[..., -1:, :] - goal[..., -1:, :].detach()
        diff = diff.squeeze(-2)
        basis = self.spectral_eigenvectors.to(device=diff.device, dtype=diff.dtype)
        weights = self.spectral_weights.to(device=diff.device, dtype=diff.dtype)
        proj = diff @ basis[:, : weights.numel()]
        spectral = (proj.square() * weights).sum(dim=-1)
        if self.vanilla_blend:
            vanilla = diff.square().sum(dim=-1)
            spectral = spectral + self.vanilla_blend * vanilla
        return spectral

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        if "goal" not in info_dict:
            return self.base_model.get_cost(info_dict, action_candidates)

        info = dict(info_dict)
        device = next(self.base_model.parameters()).device
        for key in list(info.keys()):
            if torch.is_tensor(info[key]):
                info[key] = info[key].to(device)

        goal = {key: value[:, 0] for key, value in info.items() if torch.is_tensor(value)}
        goal["pixels"] = goal["goal"]
        for key in list(info.keys()):
            if key.startswith("goal_") and key in goal:
                goal[key[len("goal_") :]] = goal.pop(key)
        goal.pop("action", None)
        goal = self.base_model.encode(goal)

        info["goal_emb"] = goal["emb"]
        if "emb" not in info:
            try:
                init = {key: value[:, 0] for key, value in info.items() if torch.is_tensor(value)}
                init = self.base_model.encode(init)
                emb = init["emb"].unsqueeze(1).expand(
                    action_candidates.size(0),
                    action_candidates.size(1),
                    *init["emb"].shape[1:],
                )
                info["emb"] = emb
            except Exception:
                info.pop("emb", None)
        info = self.base_model.rollout(info, action_candidates)
        return self._spectral_terminal_cost(info["predicted_emb"], info["goal_emb"])

    def criterion(self, info_dict: dict) -> torch.Tensor:
        if "predicted_emb" not in info_dict or "goal_emb" not in info_dict:
            return self.base_model.criterion(info_dict)
        return self._spectral_terminal_cost(info_dict["predicted_emb"], info_dict["goal_emb"])


def wrap_model_with_spectral_cost(
    model: torch.nn.Module,
    latents_path: str,
    *,
    max_samples: int = 200_000,
    seed: int = 3072,
    top_k: int = 0,
    weight_mode: str = "reachability_controllability_nuisance",
    normalize_weights: bool = True,
    vanilla_blend: float = 0.0,
) -> SpectralCostModel:
    config = SpectralPlannerConfig(
        latents_path=latents_path,
        max_samples=max_samples,
        seed=seed,
        top_k=top_k,
        weight_mode=weight_mode,
        normalize_weights=normalize_weights,
        vanilla_blend=vanilla_blend,
    )
    state = build_spectral_planner_state(config)
    return SpectralCostModel(model, state, vanilla_blend=vanilla_blend)
