"""Future no-padding H2 candidate queue, not an experiment launcher."""
from experiments.spectral_reachability import codex_p3_fixed_panel_quota_20261030_v1 as quota


def queue(policy, weights):
    quotas = quota.allocation(policy, weights)
    jobs = [{"root": root, "index": index}
            for root in (0, 1) for index in range(quotas[root])]
    if len(jobs) != 20:
        raise ValueError("fixed aggregate candidate cap")
    return {"policy": policy, "quotas": list(quotas), "jobs": jobs,
            "eligible": "all dispatched", "candidate_cap": 20,
            "predict_cap": 40, "action_encoder_cap": 40,
            "predictor_cap": 40, "pred_proj_cap": 40,
            "per_root_predict_caps": [2 * n for n in quotas],
            "cost_qualification": "H2 two predict calls per candidate; other axes separately measured",
            "shared_signal_cost": "requires separate accounting",
            "physical_release": False, "scientific_pass": False}
