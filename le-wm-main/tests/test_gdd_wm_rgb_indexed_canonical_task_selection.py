import json

from experiments.gdd_wm.audit_rgb_indexed_canonical_task_selection import audit


def _write_fixture(tmp_path):
    candidate_path = tmp_path / "candidate_dataset.json"
    records = []
    reports = {}
    split_roots = {
        "train": ("train0", "train1"),
        "calibration": ("cal0",),
        "outcome": ("out0", "out1"),
    }
    # Candidate 0 and candidate 1 have distinct train-derived canonical actions.
    # The RGB successor projections are deliberately identical so the baseline
    # cannot recover the action identity without the indexed bank.
    actions = {
        0: [[1.0, 0.0], [1.0, 0.0]],
        1: [[0.0, 1.0], [0.0, 1.0]],
    }
    for split, roots in split_roots.items():
        rows = []
        for root in roots:
            for candidate_id in (0, 1):
                # The train label makes candidate 1 optimal; held-out labels
                # preserve that relation and test root-disjoint transfer.
                cost = float(candidate_id == 0)
                records.append(
                    {
                        "instance_id": root,
                        "candidate_id": candidate_id,
                        "split": split,
                        "source_state_raw": [0.0, 0.0, 0.0, 0.0],
                        "goal_raw": [1.0, 0.0],
                        "action_sequence": actions[candidate_id],
                        "real_task_cost": cost,
                    }
                )
                rows.append(
                    {
                        "instance_id": root,
                        "candidate_id": candidate_id,
                        "split": split,
                        "source_projection": [0.0, 0.0],
                        "predicted_terminal_projection": [0.0, 0.0],
                    }
                )
        report = {
            "gates": {
                "renderer_consistency": True,
                "realized_rgb_latents_available": True,
                "pca_projected_successor_comparison": True,
            },
            "rows": rows,
        }
        reports[split] = report
    candidate_path.write_text(json.dumps({"records": records}), encoding="utf-8")
    paths = {}
    for split, report in reports.items():
        path = tmp_path / f"{split}.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        paths[split] = path
    return candidate_path, paths


def test_rgb_indexed_action_bank_transfers_to_root_disjoint_task_selection(tmp_path):
    candidate, reports = _write_fixture(tmp_path)
    result = audit(candidate, reports["train"], reports["calibration"], reports["outcome"])

    baseline = result["results"]["rgb_source_plus_delta"]
    indexed = result["results"]["rgb_indexed_canonical_action"]
    assert result["protocol"]["action_bank_fit_split"] == "train"
    assert result["protocol"]["outcome_used_for_fit_or_feature_selection"] is False
    assert result["bank"]["all_rows_canonical_deviation"] == 0.0
    assert indexed["candidate_regret"]["outcome"]["mean_regret"] < baseline["candidate_regret"]["outcome"]["mean_regret"]
    assert baseline["execution_consistency"]["outcome"]["pairs"]["execution_consistency"] is None
    assert indexed["execution_consistency"]["outcome"]["pairs"]["execution_consistency"] == 1.0


def test_rgb_indexed_audit_rejects_failed_successor_gate(tmp_path):
    candidate, reports = _write_fixture(tmp_path)
    failed = json.loads(reports["outcome"].read_text(encoding="utf-8"))
    failed["gates"]["renderer_consistency"] = False
    reports["outcome"].write_text(json.dumps(failed), encoding="utf-8")
    try:
        audit(candidate, reports["train"], reports["calibration"], reports["outcome"])
    except ValueError as error:
        assert "not all passed" in str(error)
    else:
        raise AssertionError("failed RGB successor gate must fail closed")
