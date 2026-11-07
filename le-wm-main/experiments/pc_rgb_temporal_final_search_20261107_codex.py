"""Keep endpoint-based retention; optionally choose final chain by temporal minimum."""
import numpy as np
import torch

from experiments import pc_rgb_hold_retention_search_20261107_codex as base
from experiments.pc_input_permissions import RGB, RGBLeWM
from experiments.pc_scored_candidate_selection_20261107_codex import ScoredCandidate, select_scored


def choose_with_temporal(result, temporal_scores, parents, probes, tails, prefix, remaining):
    if len(temporal_scores) != 3 or tuple(map(len, temporal_scores)) != (3, 9, 4):
        raise ValueError("temporal scores do not match search queries")
    candidates = [ScoredCandidate(source, i, float(v))
                  for source, scores in zip(("parent", "probe", "tail"), temporal_scores, strict=True)
                  for i, v in enumerate(scores)]
    chosen = select_scored(candidates)
    if chosen.source == "parent":
        actions = parents[chosen.index]
    elif chosen.source == "probe":
        i, j = divmod(chosen.index, 3)
        bank = np.concatenate((np.zeros_like(probes[:1]), probes))
        actions = np.concatenate((parents[i, :prefix], bank[j]))
    else:
        i, j = divmod(chosen.index, 2)
        actions = np.concatenate((parents[result["retained_prefixes"][i], :prefix], tails[j]))
    if actions.shape != (remaining, 2):
        raise ValueError("temporal selection clock mismatch")
    return chosen, actions.copy()


def compile_rgb(boundary, current, *, scaler, old_tail, candidate_seed, strategy_seed,
                strategy, remaining_steps, final_rule):
    if (not isinstance(boundary, RGBLeWM) or boundary.audit.track_id != RGB
            or set(current) != {"pixels", "goal", "action"} or strategy not in ("lookahead", "effect")
            or remaining_steps not in (5, 10, 15, 20, 25) or final_rule not in ("endpoint", "temporal")):
        raise ValueError("invalid RGB contract, clock or rule")
    calls, packet_steps = 0, 0
    temporal_scores = []

    def score(chains):
        nonlocal calls, packet_steps
        normal = np.asarray(scaler.transform(chains.reshape(-1, 2)), dtype=np.float32)
        normal = normal.reshape(1, len(chains), remaining_steps // 5, 10)
        value, context = boundary.score(current, torch.as_tensor(normal, device=current["pixels"].device))
        predicted, goal = context["predicted_emb"], context["goal_emb"][:, -1]
        if predicted.shape[2] != remaining_steps // 5 + 1:
            raise ValueError("unexpected model time dimension")
        time_cost = (predicted[:, :, 1:] - goal[:, None, None]).square().sum(-1)
        if not torch.isfinite(time_cost).all() or not torch.allclose(value, time_cost[:, :, -1], rtol=0, atol=1e-3):
            raise ValueError("invalid temporal model scores")
        temporal_scores.append(time_cost.min(dim=-1).values[0].detach().cpu().numpy())
        calls += 1
        packet_steps += len(chains) * (remaining_steps // 5)
        return value.detach().cpu().numpy().reshape(-1)

    if remaining_steps == 5:
        parents, _, _ = base.timed.build(candidate_seed, old_tail, 5, count=16)
        if old_tail is None:
            parents[0] = 0
        parents = parents[:, :5]
        costs = score(parents)
        idx = int(np.argmin(costs))
        result = {"actions": parents[idx].copy(), "choice": ScoredCandidate("fallback", idx, float(costs[idx])),
                  "retained_prefixes": [], "parent_cost": costs, "probe_cost": np.empty((0, 0)),
                  "tail_cost": np.empty(0), "effects": np.empty((0, 0)), "model_queries": 16,
                  "mode": "direct_fallback", "prefix_steps": None}
    else:
        prefix = min(10, remaining_steps - 5)
        parents, probes, tails = base.timed.build(candidate_seed, old_tail, prefix)
        parents, probes, tails = (parents[:, :remaining_steps], probes[:, :remaining_steps-prefix],
                                  tails[:, :remaining_steps-prefix])
        result = base.search(score, parents, probes, tails, prefix=prefix, remaining=remaining_steps,
                             strategy=strategy, seed=strategy_seed)
        if final_rule == "temporal":
            result["choice"], result["actions"] = choose_with_temporal(
                result, temporal_scores, parents, probes, tails, prefix, remaining_steps)
    result.update({"forward_calls": calls, "model_packet_steps": packet_steps, "remaining_steps": remaining_steps,
                   "temporal_scores": [x.tolist() for x in temporal_scores]})
    return result
