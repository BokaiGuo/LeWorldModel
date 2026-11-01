import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/spectral_reachability/p3_actual_resource_audit_codex_20261101.py"
CONSUMER = ROOT / "results/codex_p3_online_budget_host_20261030_v1_CONSUME_01.json"
SHA = "03a3eaa5f7cce26298ab578440c02dac87b7d984ab0e2cb3eccfdc07c28adc67"
spec = importlib.util.spec_from_file_location("p3_actual_resource_audit", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_saved_real_consumer():
    process = subprocess.run([sys.executable, str(SCRIPT), str(CONSUMER), "--sha256", SHA],
                             check=True, capture_output=True, text=True)
    result = json.loads(process.stdout)
    assert not result["equal_total_compute_certified"]
    assert not result["scientific_pass"]
    assert len(result["comparisons"]) == 4
    for row in result["comparisons"]:
        assert not row["same_recorded_vector"]
        assert row["actual_axis_delta"]["predict"] == 0
        assert row["actual_axis_delta"]["grad_api"] == 384
        assert row["actual_axis_delta"]["metric"] == -4


def test_wrong_hash_rejected():
    process = subprocess.run([sys.executable, str(SCRIPT), str(CONSUMER), "--sha256", "0" * 64],
                             capture_output=True, text=True)
    assert process.returncode != 0
    assert "SHA-256 mismatch" in process.stderr


@pytest.mark.parametrize("mutation", ["duplicate", "missing_axis", "bool", "negative", "admission"])
def test_invalid_consumer_rejected(mutation):
    data = json.loads(CONSUMER.read_bytes())
    if mutation == "duplicate":
        data["rows"][0] = copy.deepcopy(data["rows"][1])
    elif mutation == "missing_axis":
        del data["rows"][0]["attempted"]["svd"]
    elif mutation == "bool":
        data["rows"][0]["attempted"]["svd"] = False
    elif mutation == "negative":
        data["rows"][0]["attempted"]["svd"] = -1
    else:
        data["scientific_pass"] = True
    with pytest.raises(ValueError):
        module.audit(data, SHA)
