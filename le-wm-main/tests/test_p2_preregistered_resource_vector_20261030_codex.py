import copy
import hashlib
import json

import pytest

from experiments.spectral_reachability import p2_preregistered_resource_vector_20261030_codex as gate


H = "a" * 64


def contract():
    return {
        "schema": gate.NAME + ".contract.v1",
        "arms": ["composition", "joint"],
        "scope": "per_root_per_arm_standalone_planning_and_selected_physical_execution",
        "identity": {k: H for k in gate.IDENTITY},
        "axes": {k: {"unit": unit, "aggregation": "max" if k == "peak_device_bytes" else "sum",
                     "cap": 100} for k, unit in gate.AXES.items()},
    }


def receipt(c, arm, n=10):
    axes = {}
    for key, unit in gate.AXES.items():
        entry = {"unit": unit, "coverage": "complete", "evidence_sha256": H}
        entry.update({"attempted": n, "completed": n} if key in gate.COUNTS else {"value": n})
        axes[key] = entry
    return {"schema": gate.NAME + ".receipt.v1", "contract_sha256": gate.sha(gate.canonical(c)),
            "arm": arm, "run_id": arm, "identity": copy.deepcopy(c["identity"]),
            "status": "complete", "axes": axes}


def fixture():
    c = contract()
    return c, [receipt(c, arm) for arm in c["arms"]]


def test_equal_and_unequal_within_common_cap_are_distinct():
    c, rows = fixture()
    digest = gate.sha(gate.canonical(c))
    equal = gate.compare(c, digest, rows)
    assert equal["common_cap_satisfied"] and equal["actual_vector_equal"]
    assert not equal["equal_total_compute"] and not equal["P2_admission"]
    rows[1]["axes"]["vjp_directions"].update(attempted=11, completed=11)
    unequal = gate.compare(c, digest, rows)
    assert unequal["common_cap_satisfied"] and not unequal["actual_vector_equal"]
    assert unequal["unequal_axes"]["vjp_directions"] == {"composition": 10, "joint": 11}


@pytest.mark.parametrize("change", [
    lambda c, r: c["axes"].pop("flops"),
    lambda c, r: c["axes"]["peak_device_bytes"].update(aggregation="sum"),
    lambda c, r: c["axes"]["flops"].update(cap=None),
    lambda c, r: c["axes"]["flops"].update(cap=True),
    lambda c, r: c.update(scope="planning_only"),
    lambda c, r: r[0]["axes"].pop("solve_calls"),
    lambda c, r: r[0]["axes"]["flops"].update(coverage="unknown"),
    lambda c, r: r[0]["axes"]["flops"].update(evidence_sha256=""),
    lambda c, r: r[0]["axes"]["flops"].update(attempted=None),
    lambda c, r: r[0]["axes"]["flops"].update(attempted=True),
    lambda c, r: r[0]["axes"]["flops"].update(attempted=11, completed=10),
    lambda c, r: r[0]["axes"]["flops"].update(attempted=10, completed=11),
    lambda c, r: r[0]["axes"]["flops"].update(attempted=101, completed=101),
    lambda c, r: r[0]["axes"]["peak_device_bytes"].update(value=101),
    lambda c, r: r[0]["axes"]["flops"].update(unit="calls"),
    lambda c, r: r[0]["identity"].update(task_sha256="b" * 64),
    lambda c, r: r[0].update(contract_sha256="b" * 64),
    lambda c, r: r[0].update(status="partial"),
    lambda c, r: r[1].update(arm=r[0]["arm"]),
    lambda c, r: r[1].update(run_id=r[0]["run_id"]),
    lambda c, r: r.pop(),
])
def test_reject_incomplete_incomparable_or_over_cap(change):
    c, rows = fixture()
    change(c, rows)
    with pytest.raises(ValueError):
        gate.compare(c, gate.sha(gate.canonical(c)), rows)


def test_contract_mutation_rejected_against_committed_hash():
    c, rows = fixture()
    digest = gate.sha(gate.canonical(c))
    c["axes"]["flops"]["cap"] = 200
    with pytest.raises(ValueError, match="canonical contract bytes"):
        gate.compare(c, digest, rows)


def test_exact_file_digest_and_duplicate_keys(tmp_path):
    path = tmp_path / "contract.json"
    data = b'{"a":1,"a":2}'
    path.write_bytes(data)
    with pytest.raises(ValueError, match="duplicate"):
        gate.read_bound(path, hashlib.sha256(data).hexdigest())
    path.write_text(json.dumps(contract()))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        gate.read_bound(path, H)


def test_historical_diagnostic_is_not_prospective_admission():
    report = gate.historical_diagnostic()
    assert len(report["roots"]) == 2
    assert report["roots"][0]["predict_samples"] == {
        "h15": 132, "cem": 132, "gn": 117, "gradient": 117}
    assert report["actual_vector_equal"] is None
    assert not report["prospective_contract_bound"] and not report["P2_admission"]
