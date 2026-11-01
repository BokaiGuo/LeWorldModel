"""Synthetic saved summaries only; no running results, simulator or GPU."""
import copy
import json
import math

import pytest

from experiments.spectral_reachability import codex_p1_three_prefix_h10_analysis_20261030 as analysis
from experiments.spectral_reachability import codex_p1_three_prefix_h10_prepare_20261030 as prep


@pytest.fixture
def fixture():
    banks = [prep.bank(bytes(80)) for _ in range(8)]
    anchors, rows = [], []
    goal = [0., 0., 10., 10., 0., 0., 0.]
    for i in range(8):
        for j, role in enumerate(analysis.ROLES):
            state = [0., 0., float(j), 0., 0., 0., 0.]
            anchors.append(dict(index=3*i+j, logical_pair=i, role=role, root_slot=i//4,
                                condition=("off", "on")[(i//2)%2], arm=str(i%2),
                                task_id=str(i//2), physical_root_sha256=str(i//4),
                                goal=goal, checkpoint={"state": state},
                                prefix5_float32_hex=(bytes([j])*40).hex(),
                                duplicate_prefix_task_group=j))
            endpoints = [[0., 0., float(j)+k, 0., .1*k, 0., 0.] for k in range(42)]
            rows.append(dict(anchor=3*i+j, statuses=["probe_completed"]*42, endpoints=endpoints))
    original = dict(anchors=anchors, probe_banks=banks, fixed_pairs=8, fixed_branches=1008)
    verified = dict(rows=rows, fixed_branches=1008, saved_checks_passed=True, P1_admission=False,
                    independent_confirmation=False, counts=dict(probe_completed=1008, native_stopped=0,
                                                               censored_out_of_domain=0))
    return original, verified


def test_full_roster_same_suffix_response_and_comparisons(fixture):
    original, verified = fixture
    result = analysis.analyze(original, verified)
    assert len(result["groups"]) == 8 and sum(len(g["probes"]) for g in result["groups"]) == 336
    row = result["groups"][0]["probes"][2]
    assert row["roles"]["selected"]["metrics"]["object_displacement_xy"] == [2., 0.]
    assert row["roles"]["greedy"]["response_vs_own_zero"]["object_response_xy"] == [2., 0.]
    assert row["comparisons"]["selected_minus_greedy"]["endpoint_metrics"]["object_position_error"] > 0
    assert row["comparisons"]["selected_minus_greedy"]["response_vs_own_zero"]["object_response_xy"] == [0., 0.]
    assert result["P1_admission"] is result["freshness"] is result["independent_confirmation"] is False


@pytest.mark.parametrize("status", ["native_stopped", "censored_out_of_domain"])
def test_missing_endpoint_and_missing_zero_baseline_are_not_imputed(fixture, status):
    original, verified = fixture
    verified["rows"][0]["statuses"][0] = status
    verified["rows"][0]["endpoints"][0] = None
    verified["counts"]["probe_completed"] -= 1
    verified["counts"][status] += 1
    result = analysis.analyze(original, verified)
    zero, other = result["groups"][0]["probes"][:2]
    assert zero["roles"]["selected"]["status"] == status
    assert zero["roles"]["selected"]["metrics"] is None
    assert zero["comparisons"]["selected_minus_zero"]["endpoint_metrics"] is None
    assert other["roles"]["selected"]["metrics"] is not None
    assert other["roles"]["selected"]["response_vs_own_zero"] is None
    assert other["comparisons"]["selected_minus_greedy"]["response_vs_own_zero"] is None


@pytest.mark.parametrize("fault", ["task", "role", "sha", "missing", "count", "nan", "false_complete"])
def test_inconsistent_inputs_fail_closed(fixture, fault):
    original, verified = copy.deepcopy(fixture)
    if fault == "task":
        original["anchors"][1]["task_id"] = "wrong"
    elif fault == "role":
        original["anchors"][1]["role"] = "greedy"
    elif fault == "sha":
        original["probe_banks"][0][2]["raw10_sha256"] = "0"*64
    elif fault == "missing":
        verified["rows"].pop()
    elif fault == "count":
        verified["counts"]["probe_completed"] -= 1
    elif fault == "nan":
        verified["rows"][0]["endpoints"][1][2] = float("nan")
    else:
        verified["rows"][0]["endpoints"][1] = None
    with pytest.raises((ValueError, KeyError)):
        analysis.analyze(original, verified)


def test_wrapped_angles_and_duplicate_groups(fixture):
    original, verified = fixture
    for role in range(3):
        original["anchors"][3+role] = dict(original["anchors"][role], index=3+role, logical_pair=1,
                                            arm=original["anchors"][3]["arm"])
    verified["rows"][0]["endpoints"][1][4] = 2*math.pi-.1
    result = analysis.analyze(original, verified)
    assert result["groups"][0]["duplicate_group"] == result["groups"][1]["duplicate_group"]
    assert result["groups"][0]["probes"][1]["roles"]["selected"]["metrics"]["object_angle_error"] == pytest.approx(.1)


def test_consume_requires_saved_consumer_and_source_reverification(fixture, monkeypatch, tmp_path):
    original, verified = fixture
    called = []
    monkeypatch.setattr(analysis.saved, "consume", lambda *args: (called.append("saved") or verified))
    monkeypatch.setattr(analysis.runner, "verify_prepare", lambda *args: (called.append("prepare") or
                                                                        {"original": original}))
    shards = []
    for i in range(24):
        directory = tmp_path / f"anchor_{i:02d}"
        directory.mkdir()
        path = directory / "report.json"
        path.write_text(json.dumps(dict(rows=[dict(index=j, status="probe_completed", responses=[{}]*10)
                                               for j in range(42)])))
        shards.append(dict(anchor=i, report_sha256=analysis.runner.runner.physical.sha(path)))
    report = tmp_path / "run_report.json"
    report.write_text(json.dumps(dict(shards=shards)))
    digest = analysis.runner.runner.physical.sha(report)
    result = analysis.consume(str(tmp_path), digest, "/fixture", "a"*64)
    assert called == ["saved", "prepare"]
    assert result["run_report_sha256"] == digest and result["fixed_branches"] == 1008
    assert result["groups"][0]["probes"][0]["roles"]["selected"]["observation"]["accepted_probe_steps"] == 10


def test_preserves_stopped_and_censored_detail(fixture):
    original, verified = fixture
    verified["rows"][0]["statuses"][1] = "native_stopped"
    verified["rows"][0]["endpoints"][1] = None
    verified["rows"][1]["statuses"][1] = "censored_out_of_domain"
    verified["rows"][1]["endpoints"][1] = None
    verified["counts"].update(probe_completed=1006, native_stopped=1, censored_out_of_domain=1)
    details = [[dict(status=verified["rows"][i]["statuses"][j], native_stop=None, censor=None)
                for j in range(42)] for i in range(24)]
    details[0][1]["native_stop"] = dict(step=7, terminated=False, truncated=True)
    details[1][1]["censor"] = dict(step=8, out_of_bounds=[{"coordinate": 0, "value": 520.}])
    row = analysis.analyze(original, verified, details)["groups"][0]["probes"][1]
    assert row["roles"]["selected"]["observation"]["native_stop"]["step"] == 7
    assert row["roles"]["zero"]["observation"]["censor"]["step"] == 8
    assert row["comparisons"]["selected_minus_zero"]["endpoint_metrics"] is None
