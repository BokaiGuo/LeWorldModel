"""Synthetic-only contract and native guarded branch tests; no environment creation."""
from copy import deepcopy
from pathlib import Path

import pytest

from experiments.spectral_reachability import codex_p2_bank_physical_contract_20261030_v1 as m
from test_p2_composition_predict_frontier_physical_20261027_v1 import Env, fixture_plan, source_fixture


@pytest.fixture
def material(monkeypatch, tmp_path):
    body, exported, compilation = source_fixture()
    old_roots = [c["root_sha256"] for c in body["context"]["cases"]]
    roots = list(m.frontier.ROOTS)
    packets, refs = [], {}
    policy = tmp_path / "policy"
    models = {str(policy / n): "model" for n in ("weights.pt", "config.json")}
    for i, root in enumerate(roots):
        case, c = body["context"]["cases"][i], compilation["rows"][i]["compiled"]
        case["root_sha256"] = case["root"]["root_id"] = root
        compilation["rows"][i]["root_sha256"] = root
        packets.append(dict(root_sha256=root, root_context=case["root"]["context"],
                            goal=c["goal_vectors"][0], projection=c["scoring"]["metric"]["projection"]))
        refs[root] = dict(history_rgb_sha256="a"*64, goal_rgb_sha256="b"*64)
        for e in exported["plans"]:
            if e["root_sha256"] == old_roots[i]:
                e["root_sha256"] = e["root"]["root_id"] = root
    protocol = dict(context=dict(body["context"], compilation="fixture", policy=str(policy)),
                    source_paths=[], bindings={})
    prepared = {}
    monkeypatch.setattr(m.frontier, "load_prepared", lambda *a: (None, protocol, prepared))
    monkeypatch.setattr(m.frontier, "read", lambda *a: compilation)
    monkeypatch.setattr(m.saved.bridge, "check_sources", lambda *a: None)
    monkeypatch.setattr(m.physical.helpers, "validate_prepared",
                        lambda *a: dict(packets=packets, references=refs, root_seed=123, model_bindings=models))
    def consume(**kw):
        case = kw["ready"]
        plans = []
        for e in exported["plans"]:
            if e["root_sha256"] == case["root_sha256"]:
                p = dict(root_sha256=e["root_sha256"], root_index=e["root_index"], arm=e["arm"],
                         root=deepcopy(case["root"]), action_domain=case["action_domain"],
                         commands=e["raw_commands"], witness=e["witness"],
                         selection_uses_physical_outcome=False)
                p["selection_sha256"] = m.frontier.digest(p)
                plans.append(p)
        return dict(plans=plans, journal_sha256="journal")
    monkeypatch.setattr(m.saved, "consume", consume)
    bindings = dict(models)
    for mod in (m, m.saved, m.physical, m.saved.bridge, m.saved.bound):
        bindings[str(Path(mod.__file__).resolve())] = "fixture"
    reports = {root: (str(tmp_path / str(i)), str(i)*64) for i, root in enumerate(roots)}
    return protocol, compilation, exported, reports, bindings


def test_four_by_two_and_report_binding(material):
    _, _, _, reports, bindings = material
    c = m.build("fixture", "sha", reports, bindings)
    assert len(c["plans"]) == 8
    assert [p["method"] for p in c["plans"][:4]] == list(m.frontier.ARMS) + list(m.frontier.ARMS[::-1])
    assert all(p["planner_report_sha256"] == reports[p["root_sha256"]][1] for p in c["plans"])
    assert c["contract_sha256"] == m.frontier.digest({k: v for k, v in c.items() if k != "contract_sha256"})
    assert not c["physical_authorized"] and not c["P2_admission"]


@pytest.mark.parametrize("damage", ["missing", "duplicate", "goal", "rgb", "action"])
def test_bad_material_fails_closed(material, damage):
    protocol, compilation, exported, reports, bindings = material
    if damage == "missing":
        reports.pop(next(iter(reports)))
    elif damage == "duplicate":
        exported["plans"][1] = deepcopy(exported["plans"][0])
    elif damage == "goal":
        compilation["rows"][0]["compiled"]["goal_vectors"][0][0] = 1
    elif damage == "rgb":
        protocol["context"]["cases"][0]["root"]["root_rgb_sha256"] = "wrong"
    else:
        exported["plans"][0]["raw_commands"][0][0] = 2
    with pytest.raises((ValueError, KeyError)):
        m.build("fixture", "sha", reports, bindings)


@pytest.mark.parametrize("mode", ["complete", "terminated", "warmup", "rgb", "step"])
def test_guarded_synthetic_branch(tmp_path, mode):
    p = fixture_plan()
    counts = dict(attempts=0, returns=0)
    env = Env(terminate=3 if mode == "terminated" else None, fail=2 if mode == "step" else None)
    def warm():
        counts.update(attempts=15, returns=14 if mode == "warmup" else 15)
        return None, None, None
    def rgb(*args):
        if mode == "rgb":
            raise ValueError("RGB mismatch")
        return {}
    row = m.checked_branch(plan=p, output=tmp_path, reset_and_warmup=warm, warmup_counts=lambda: counts,
                           references={p["root_sha256"]: {}}, verify_rgb_context=rgb,
                           validate_reset_history=lambda *a: {}, env=env,
                           state_of=lambda e: [float(e.n)] * 7, evaluate=lambda state: {})
    if mode in ("warmup", "rgb"):
        assert env.n == 0 and row["status"] == "branch_failed"
    elif mode == "complete":
        assert env.n == 10 and row["full_horizon_completed"]
    else:
        assert env.n in (2, 3) and not row["full_horizon_completed"]


def test_no_production_dispatch():
    with pytest.raises(RuntimeError, match="BLOCKED"):
        m.execute_physics()


def test_history_mismatch_blocks_and_journal_tamper_fails(tmp_path):
    p = fixture_plan()
    counts = {"attempts": 0, "returns": 0}
    env = Env()
    def warm():
        counts.update(attempts=15, returns=15)
        return None, None, None
    def reject(*args):
        raise ValueError("history mismatch")
    row = m.checked_branch(plan=p, output=tmp_path, reset_and_warmup=warm, warmup_counts=lambda: counts,
                           references={p["root_sha256"]: {}}, verify_rgb_context=lambda *a: {},
                           validate_reset_history=reject, env=env,
                           state_of=lambda e: [0.] * 7, evaluate=lambda s: {})
    assert env.n == 0 and row["failures"]
    Path(row["journal_path"]).write_bytes(b"tampered\n")
    with pytest.raises(ValueError, match="journal bytes"):
        m.physical.verify_branch(p, row)
