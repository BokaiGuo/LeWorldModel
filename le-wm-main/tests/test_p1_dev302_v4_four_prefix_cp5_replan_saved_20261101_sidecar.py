import copy
import json

import numpy as np
import pytest

from experiments.spectral_reachability import (
    p1_dev302_v4_four_prefix_cp5_replan_20261101_sidecar as runner,
)
from experiments.spectral_reachability import (
    p1_dev302_v4_four_prefix_cp5_replan_saved_20261101_sidecar as saved,
)
from test_p1_dev302_v4_four_prefix_cp5_replan_20261101_sidecar import (
    write_physical_fixture,
)


def ledger():
    return {
        axis: {
            "attempted_calls": cap["calls"],
            "completed_calls": cap["calls"],
            "attempted_samples": cap["samples"],
            "completed_samples": cap["samples"],
        }
        for axis, cap in runner.AXIS_CAPS.items()
    }


def synthetic_candidates(protocol, source):
    seeds = np.asarray(protocol["seed_tails"], np.float32)
    mean = np.asarray(source["old_v4_tail"], np.float32)
    std = np.full_like(mean, runner.INITIAL_STD)
    rng = np.random.default_rng(runner.SEED)
    rows, iterations = [], []
    for iteration in range(runner.ITERATIONS):
        bank = runner.common.make_bank(rng, mean, std, seeds, iteration)
        base = np.mean(np.square(bank), axis=(1, 2, 3))
        variant_q = np.stack(
            [base + offset for offset in (0.0, 0.01, 0.02, 0.03, 0.04)], axis=1
        )
        robust = runner.common.robust_scores(variant_q)
        for candidate in range(runner.SAMPLES):
            rows.append(
                {
                    "iteration": iteration,
                    "candidate": candidate,
                    "nominal_q": float(variant_q[candidate, 0]),
                    "variant_q": variant_q[candidate].tolist(),
                    "robust_score": float(robust[candidate]),
                    "actions": bank[candidate].tolist(),
                }
            )
        mean, std, elite = runner.common.update(bank, robust)
        iterations.append(
            {
                "iteration": iteration,
                "elite_indices": elite,
                "mean": mean.tolist(),
                "std": std.tolist(),
            }
        )
    selected = min(
        rows,
        key=lambda item: (
            item["robust_score"],
            item["iteration"],
            item["candidate"],
        ),
    )
    return rows, iterations, selected


def payload(tmp_path, monkeypatch):
    physical, physical_sha = write_physical_fixture(tmp_path, monkeypatch)
    protocol = runner.protocol(physical, physical_sha)
    events = [
        {
            "kind": "run_started",
            "protocol_sha256": "protocol",
            "roles": list(runner.ROLES),
        }
    ]
    rows, tables = [], {}
    now = 10.0
    for source in protocol["roles"]:
        role = source["role"]
        events.append(
            {
                "kind": "arm_started",
                "role": role,
                "cp5_pixel_sha256": source["cp5_pixel_sha256"],
                "actual_prefix_sha256": source["actual_prefix_sha256"],
            }
        )
        for axis, cap in runner.AXIS_CAPS.items():
            per_call = cap["samples"] // cap["calls"]
            assert per_call * cap["calls"] == cap["samples"]
            for call_index in range(cap["calls"]):
                events.append(
                    {
                        "kind": "resource_attempt",
                        "role": role,
                        "axis": axis,
                        "call_index": call_index,
                        "samples": per_call,
                        "started_seconds": now,
                    }
                )
                events.append(
                    {
                        "kind": "resource_return",
                        "role": role,
                        "axis": axis,
                        "call_index": call_index,
                        "samples": per_call,
                        "started_seconds": now,
                        "ended_seconds": now + 0.01,
                    }
                )
                now += 0.02
        candidates, iterations, selected = synthetic_candidates(protocol, source)
        tables[role] = candidates
        row = {
            "role": role,
            "candidate_index": source["candidate_index"],
            "candidate_label": source["candidate_label"],
            "status": "replanned",
            "cp5_pixel_sha256": source["cp5_pixel_sha256"],
            "actual_prefix": source["actual_prefix"],
            "actual_prefix_sha256": source["actual_prefix_sha256"],
            "old_v4_tail": source["old_v4_tail"],
            "selected": selected,
            "iterations": iterations,
            "candidate_file": f"{role}.candidates.jsonl",
            "candidate_file_sha256": f"{role}-sha",
            "axis_ledger": ledger(),
        }
        rows.append(row)
        events.append(
            {
                "kind": "arm_complete",
                "role": role,
                "selected_iteration": selected["iteration"],
                "selected_candidate": selected["candidate"],
                "selected_robust_score": selected["robust_score"],
                "candidate_file_sha256": row["candidate_file_sha256"],
                "axis_ledger": ledger(),
            }
        )
    events.append(
        {
            "kind": "run_complete",
            "arms": len(runner.ROLES),
            "P1_admission": False,
        }
    )
    model = {"original_state_dict": runner.MODEL_SHA256}
    report = {
        "schema": runner.NAME + ".report.v1",
        "status": "model_only_complete_not_admitted",
        "error": None,
        "physical_report_sha256": physical_sha,
        "rows": rows,
        "model_before": model,
        "model_after": copy.deepcopy(model),
        "module_hooks": {"action_encoder": 40, "predict": 40},
        "equal_cem_and_robust_budget": True,
        "post_cp5_physical_outcomes_used": False,
        "physical_execution": False,
        "training": False,
        "independent_confirmation": False,
        "P1_admission": False,
    }
    return protocol, report, events, tables


def test_saved_payload_reconstructs_all_four_equal_budget_arms(tmp_path, monkeypatch):
    protocol, report, events, tables = payload(tmp_path, monkeypatch)
    result = saved.verify_payload(
        protocol, report, events, lambda row: tables[row["role"]]
    )
    assert result["status"].endswith("not_admitted")
    assert [row["role"] for row in result["rows"]] == list(runner.ROLES)
    assert result["equal_measured_axis_budget"] is True
    assert result["actual_cp5_rgb_and_action_history_bound"] is True
    assert result["physical_execution"] is False
    assert result["P1_admission"] is False


@pytest.mark.parametrize("kind", ["resource", "winner", "prefix", "gate"])
def test_saved_payload_rejects_tampering(tmp_path, monkeypatch, kind):
    protocol, report, events, tables = payload(tmp_path, monkeypatch)
    if kind == "resource":
        events.pop(
            next(
                index
                for index, event in enumerate(events)
                if event["kind"] == "resource_return"
            )
        )
    elif kind == "winner":
        report["rows"][0]["selected"]["robust_score"] += 1
    elif kind == "prefix":
        report["rows"][0]["actual_prefix"][0][0] += 0.1
    else:
        report["P1_admission"] = True
    with pytest.raises(ValueError):
        saved.verify_payload(protocol, report, events, lambda row: tables[row["role"]])


def test_run_journal_hash_chain_rejects_tamper(tmp_path):
    rows = []
    previous = None
    for index, kind in enumerate(("run_started", "run_complete")):
        body = {"index": index, "previous": previous, "kind": kind}
        row = {**body, "sha256": runner.digest(body)}
        rows.append(row)
        previous = row["sha256"]
    path = tmp_path / "events.jsonl"
    path.write_bytes(b"".join(runner.canonical(row) + b"\n" for row in rows))
    assert len(saved.read_journal(path, previous, 2)) == 2
    rows[0]["kind"] = "tampered"
    path.write_bytes(b"".join(runner.canonical(row) + b"\n" for row in rows))
    with pytest.raises(ValueError, match="hash chain"):
        saved.read_journal(path, previous, 2)
