"""Finite bounded model-response diversity for a fixed prefix bank."""
from __future__ import annotations

import torch

from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as p1
from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0


def shortlist(signatures: torch.Tensor, *, keep: int = 4) -> list[int]:
    """Preserve controls 0, 1, then select the most response-diverse entries."""
    if (signatures.ndim != 2 or signatures.shape[0] < 2 or signatures.shape[1] < 1
            or not torch.isfinite(signatures).all() or keep < 0):
        raise ValueError("invalid response signatures or shortlist size")
    chosen = [0, 1]
    remaining = list(range(2, signatures.shape[0]))
    for _ in range(min(keep, len(remaining))):
        distances = torch.cdist(signatures[remaining], signatures[chosen]).min(dim=1).values
        # Python's max is stable on exact ties; remaining is in original order.
        winner = max(range(len(remaining)), key=lambda i: float(distances[i]))
        chosen.append(remaining.pop(winner))
    return chosen


def screen(model, pixels, history, bank, *, lower, upper, keep=4, amplitude=0.3):
    """Evaluate equal bounded suffix probes for every candidate in the bank."""
    if (bank.ndim != 3 or bank.shape[0] < 2 or bank.shape[1] != 1
            or bank.shape[2] != 10 or amplitude <= 0 or not torch.isfinite(bank).all()):
        raise ValueError("expected finite five-step two-action packet prefix bank")
    probes = bank.new_tensor([[0., 0.], [amplitude, 0.], [-amplitude, 0.],
                              [0., amplitude], [0., -amplitude]]).repeat(1, 5).unsqueeze(1)
    signatures = []
    with torch.no_grad():
        for prefix in bank:
            embeddings, actions = p1._state_after_prefix(
                model, pixels, history, prefix.unsqueeze(0),
                action_lower=lower, action_upper=upper,
            )
            terminal = p1._terminal_from_state(
                model, embeddings.expand(len(probes), -1, -1),
                actions.expand(len(probes), -1, -1), probes,
                action_lower=lower, action_upper=upper,
            )
            coordinate = p0._planning_coordinate(model, terminal, role="predicted")
            if coordinate.ndim != 2 or coordinate.shape[0] != len(probes) or not torch.isfinite(coordinate).all():
                raise RuntimeError("invalid finite-response model output")
            signatures.append((coordinate[1:] - coordinate[:1]).flatten())
    values = torch.stack(signatures)
    indices = shortlist(values, keep=keep)
    return bank[indices].detach(), {
        "original_indices": indices,
        "signatures": values.cpu().tolist(),
        "probe_parameters": probes[:, 0].cpu().tolist(),
        "prefix_predict_samples": int(len(bank) * bank.shape[1]),
        "suffix_predict_samples": int(len(bank) * len(probes)),
        "scope": "model-only finite-response diversity; excludes encoder and wall costs",
    }
