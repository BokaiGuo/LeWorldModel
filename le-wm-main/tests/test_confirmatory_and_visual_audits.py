import json

import numpy as np

from experiments.audit_visual_only_planning import audit as audit_visual
from experiments.audit_pure_learned_closure import audit as audit_closure
from experiments.empirical_spectral_audit.run_confirmatory_diagnostic_audit import audit as audit_confirmatory


def _synthetic_diagnostic(path, delta_by_source):
    records = []
    for source, delta in delta_by_source.items():
        for split_seed in range(3):
            records.append({
                "label": source,
                "split_seed": split_seed,
                "alignment_minus_validation_regret": delta,
                "alignment_selection_regret": max(0.0, delta + 1.0),
                "validation_selection_regret": 1.0,
            })
    payload = {
        "protocol": {
            "downstream_horizon": 4,
            "candidate_count": 65,
            "prediction_error_tolerance": 0.05,
            "uncertainty_unit": "bootstrap over split seeds",
        },
        "records": records,
    }
    path.write_text(json.dumps(payload))


def test_confirmatory_audit_averages_nested_splits_and_declares_primary(tmp_path):
    primary = tmp_path / "primary.json"
    secondary = tmp_path / "secondary.json"
    _synthetic_diagnostic(primary, {"a": -1.0, "b": -0.8, "c": -0.6})
    _synthetic_diagnostic(secondary, {"a": 0.4, "b": -0.2, "c": 0.1})
    report = audit_confirmatory({"h4_k65": primary, "h2_k65": secondary})
    assert report["protocol"]["primary_endpoint"] == "h4_k65"
    assert report["endpoints"]["h4_k65"]["source_cluster_bootstrap"]["n_source_clusters"] == 3
    assert report["endpoints"]["h4_k65"]["split_seed_count_total"] == 9
    assert report["protocol"]["independent_unit"] == "source checkpoint/environment artifact, not split seed, episode, horizon, or candidate"


def _visual_artifact(path, seed):
    def episodes(offset):
        return [
            {
                "episode": 0,
                "final_distance": 10.0 + offset,
                "block_xy_distance": 4.0 + offset,
                "success": False,
            },
            {
                "episode": 1,
                "final_distance": 20.0 + offset,
                "block_xy_distance": 8.0 + offset,
                "success": False,
            },
        ]

    payload = {
        "protocol": {"seed": seed, "device": "cpu", "horizon": 2, "policies": ["zero", "vanilla"]},
        "reports": [
            {"policy": "zero", "episodes": episodes(0.0)},
            {"policy": "vanilla", "episodes": episodes(-1.0)},
        ],
    }
    path.write_text(json.dumps(payload))


def test_visual_audit_requires_paired_nonprivileged_cpu_artifacts(tmp_path):
    one = tmp_path / "one.json"
    two = tmp_path / "two.json"
    _visual_artifact(one, 1)
    _visual_artifact(two, 2)
    source = tmp_path / "source.py"
    source.write_text(
        "def make_policy(args, name):\n"
        "    return WorldModelPolicy(transform={'pixels': image_transform(), 'goal': image_transform()})\n"
    )
    report = audit_visual([one, two], source)
    assert report["aggregate"]["n_episode_pairs"] == 4
    assert report["aggregate"]["vanilla_success_total"] == 0
    assert report["sanity_checks"]["no_forbidden_policy_input_name_in_make_policy"]
    assert report["aggregate"]["vanilla_minus_zero_final_distance"]["mean"] < 0.0


def test_pure_closure_audit_rejects_verifier_contamination(tmp_path):
    rows = [
        {
            "model": "block_only_torque_readout",
            "strict_task_success": False,
            "block_xy_distance": 3.0,
            "angle_error": 0.2,
            "agent_xy_distance": 4.0,
            "route_info": [],
        }
    ]
    payload = {
        "protocol": {
            "seed": 1,
            "direct_goal_conditioned_feasibility": True,
            "planning_objective": "staged_contact_direct_torque_readout",
            "training_protocol": "contact_intervention",
        },
        "rows": rows,
    }
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps(payload))
    payload["protocol"]["seed"] = 2
    second.write_text(json.dumps(payload))
    report = audit_closure([first, second])
    assert report["success_count"] == 0
    assert report["sanity_checks"]["all_seeds_no_successor_verification"]
