"""Full-bank consistent suffix and certified common-response witness contract."""
from experiments.spectral_reachability import reacher_d2p_refined_dispatch as base
from experiments.spectral_reachability import reacher_d2p_geometry_dispatch as geometry
from experiments.spectral_reachability import bounded_response_qp as qp

POLICY = "full_bank_fine8_certified_common_task_response_reconfiguration_v1"


def compile_refined_dispatch(model, pixels, goal, history, *, proposal_bank, action_lower,
                             action_upper, damping=.01, non_greedy_margin=1e-6, progress=None):
    _, receipt = base.compile_refined_dispatch(model, pixels, goal, history, proposal_bank=proposal_bank,
        action_lower=action_lower, action_upper=action_upper, damping=damping,
        non_greedy_margin=non_greedy_margin, progress=progress)
    greedy, winners = geometry.endpoint_winners(receipt)
    required = sorted(set(winners) | {0, greedy})
    metrics, residual, samples, vjp = geometry.response_bank_metrics(model, pixels, goal, history,
        proposal_bank=proposal_bank, action_lower=action_lower, action_upper=action_upper,
        required=required, greedy=greedy, damping=damping, gain_solver=qp.certified_response_gain)
    decision = qp.qualify_certified_decision(receipt, metrics)
    decision["policy"] = POLICY
    receipt.update({"program_type": POLICY, "endpoint_only_dispatch": receipt["dispatch"], "dispatch": decision,
        "common_task_response": {"anchor": "imagined_greedy_prefix_raw_zero_suffix", "residual": residual.tolist(),
            "damping": damping, "solver": "active_set_box_qp_with_primal_dual_gap", "candidate_metrics": metrics,
            "response_coordinates": "unit bounded actuator commands; task limits scale raw torques", "physical_labels_used": False},
        "model_predict_sample_evaluations": receipt["model_predict_sample_evaluations"] + samples,
        "backward_vjp_rows": receipt["backward_vjp_rows"] + vjp,
        "scope": "Certified model-side common-response witness, not physical controllability or P1 admission."})
    chosen = decision["selected_program_index"]
    return proposal_bank[chosen:chosen + 1], receipt
