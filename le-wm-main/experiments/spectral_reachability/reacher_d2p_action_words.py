"""Finite, model-native ordered action words; not a Lie-bracket claim."""
from __future__ import annotations

import torch
import numpy as np

from experiments.spectral_reachability import reacher_d2p_context as execution
from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as p1
from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0
from experiments.spectral_reachability import reacher_d2p_iterated_suffix as solver
from experiments.spectral_reachability import reacher_d2p_suffix_step_rescue as rescue
from experiments.spectral_reachability import reacher_d2p_refined_dispatch as refinement

POLICY = "frozen_one_packet_svd_four_packet_ordered_words_fine8_suffix_v1"
AMPLITUDES = (.25, .5)


def build_words(response):
    if response.ndim != 2 or min(response.shape) < 2 or not torch.is_floating_point(response) or not torch.isfinite(response).all():
        raise ValueError("finite action response with at least two singular directions required")
    # A float32 SVD can invent a second direction for an exactly rank-one
    # matrix. Decompose on the SAME device in float64, then retain the action
    # dtype used by the frozen model. Never move the live Jacobian to CPU.
    _, singular, right = torch.linalg.svd(response.to(torch.float64), full_matrices=False)
    if singular[1] <= singular[0] * 1e-8 or singular[0] <= 0:
        raise ValueError("two nondegenerate model action directions required")
    directions = right[:2].to(response.dtype).clone()
    for row in directions:
        # SVD signs are arbitrary; fix them before assembling ordered words.
        row *= row[row.abs().argmax()].sign()
        row /= row.abs().max()
    commands = [torch.zeros((4, response.shape[1]), dtype=response.dtype, device=response.device)]
    labels = [{"index": 0, "amplitude": 0., "signs": [0, 0], "order": "neutral"}]
    for amplitude in AMPLITUDES:
        for sign_a, sign_b in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            a, b = amplitude * sign_a * directions[0], amplitude * sign_b * directions[1]
            for order, word in (("AB-A-B", (a, b, -a, -b)), ("BA-B-A", (b, a, -b, -a))):
                commands.append(torch.stack(word))
                labels.append({"index": len(labels), "amplitude": amplitude, "signs": [sign_a, sign_b], "order": order})
    commands = torch.stack(commands)
    return torch.atanh(commands), {"labels": labels, "singular_values": singular.tolist(),
        "canonical_unit_command_directions": directions.tolist(), "amplitudes": AMPLITUDES,
        "scope": "Finite ordered torque packets, no continuous-time commutator or inverse-dynamics guarantee."}


def compile_words(model, pixels, goal, history, *, lower, upper, progress=None):
    emb = p0._latent_context(model, pixels, history)
    goal_latent = p0._goal_latent(model, goal)

    def terminal(flat):
        value = p1._terminal_from_state(model, emb, history, flat.reshape(1, 1, history.shape[-1]), action_lower=lower, action_upper=upper)[0]
        return p0._planning_coordinate(model, value, role="predicted")

    origin = torch.zeros(history.shape[-1], device=history.device, dtype=history.dtype, requires_grad=True)
    response = torch.autograd.functional.jacobian(terminal, origin, vectorize=True)
    bank, support = build_words(response.detach())
    # Singleton scores avoid small batch-dependent margins changing eligibility.
    with torch.no_grad():
        immediate = [float(p0._cost_fn(model, pixels, goal, history, bank[i:i + 1], action_lower=lower, action_upper=upper)[0]) for i in range(len(bank))]
    rows, samples, vjp = [], 1 + 4 * len(bank), int(response.shape[0])
    partition_errors = []
    for index in range(len(bank)):
        next_emb, next_actions = p1._state_after_prefix(model, pixels, history, bank[index:index + 1], action_lower=lower, action_upper=upper)
        suffix, receipt = solver.compile_snapshots(model, goal_latent, next_emb, next_actions, future_horizon=2,
            action_lower=lower, action_upper=upper, damping=.01, checkpoints=(8,), accept_update=rescue.accept_with_fine_rescue)[8]
        with torch.no_grad():
            joined = torch.cat((bank[index:index + 1], suffix), dim=1)
            direct = p0._terminal_batch(model, pixels, goal, history, joined, action_lower=lower, action_upper=upper)[0]
            split = p1._terminal_from_state(model, next_emb, next_actions, suffix, action_lower=lower, action_upper=upper)[0]
            partition_errors.append(float((direct - split).abs().max()))
        rows.append({"proposal_index": index, "immediate_goal_cost": immediate[index],
            "refined_goal_cost": receipt["trace"][-1]["cost_after"], "suffix_receipt": receipt})
        samples += 4 + receipt["model_predict_sample_evaluations"] + 8
        vjp += receipt["backward_vjp_rows"]
        if progress is not None:
            progress(index + 1, len(bank))
    decision = refinement.select_refined_bank(rows, non_greedy_margin=1e-6)
    return bank, {"policy": POLICY, "dispatch": decision, "proposal_rows": rows, "word_support": support,
        "partition_max_abs_error": max(partition_errors), "model_predict_sample_evaluations": samples,
        "backward_vjp_rows": vjp, "physical_labels_used": False}


def advance_program(env, context, raw, *, verified_root_frame, action_block=5):
    """Advance every packet separately; never compress 20 steps to one token."""
    raw = np.asarray(raw, np.float32)
    if (not isinstance(action_block, int) or action_block < 1 or raw.ndim != 2 or raw.shape[1] < 1
            or raw.shape[0] < action_block or raw.shape[0] % action_block or not np.isfinite(raw).all()):
        raise ValueError("complete finite multi-packet raw program required")
    advanced, boundaries = context, []
    for start in range(0, len(raw), action_block):
        frame = verified_root_frame if start == 0 else advanced["root_frame"]
        advanced = execution.advance_executed_context(env, context=advanced, raw_action_block=raw[start:start + action_block],
            action_block=action_block, verified_root_frame=frame)
        advanced["execution_snapshot"]["observed_root_rgb"] = execution.pack_rgb_observation(advanced["root_frame"])
        boundaries.append({"raw_steps_executed": start + action_block, "root_rgb_sha256": advanced["root_rgb_sha256"],
            "history_rgb_sha256": advanced["history_rgb_sha256"], "action_sha256": advanced["advanced_action_sha256"]})
    return advanced, boundaries
