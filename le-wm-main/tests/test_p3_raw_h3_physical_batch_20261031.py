import copy
import json
from pathlib import Path

import pytest

from experiments.spectral_reachability import p3_raw_h3_physical_batch_20261031 as batch


def test_sealed_batch_plan_binding_and_budget():
    root = Path(__file__).resolve().parents[1]
    result = json.loads((root / "results/p3_raw_h3_physical_prepare_20261031_RUN01.json").read_bytes())
    plans = result["plans"][:6]
    assert batch.validate_plans(plans)
    assert batch.contract()["caps"]["public_step"] == 6 * (10 + 20)
    assert batch.contract()["caps"]["explicit_render"] == 6 * (11 + 21)
    bad = copy.deepcopy(plans)
    bad[1]["commands"][0][0] += 0.1
    with pytest.raises(ValueError):
        batch.validate_plans(bad)
    bad = copy.deepcopy(plans)
    bad[1]["task_id"] = "wrong"
    with pytest.raises(ValueError, match="pair identity"):
        batch.validate_plans(bad)


def test_constructor_failures_keep_all_six_rows_and_never_certify(tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = json.loads((root / "results/p3_raw_h3_physical_prepare_20261031_RUN01.json").read_bytes())
    prepared = {"schema": batch.NAME + ".prepare", "contract": batch.contract(),
                "plans": result["plans"][:6], "bindings": {}}

    class Store:
        path = tmp_path
        previous = None

        def __init__(self):
            (self.path / "journal.jsonl").write_text("fixture")
            self.records = []

        def check(self, *args, **kwargs):
            return None

        def emit(self, kind, **payload):
            self.records.append((kind, payload))

        def put(self, name, value, **kwargs):
            assert name == "report.json"
            self.report = value

    def fail():
        raise RuntimeError("constructor incident")

    store = Store()
    actual = batch.execute(prepared, {"references": [None, None]}, store, fail)
    assert actual["status"] == "partial_or_failed"
    assert len(actual["rows"]) == 6
    assert all(row["status"] == "failed" and row["own_returned"] == 0
               and row["first_error"]["message"] == "constructor incident"
               for row in actual["rows"])
    assert actual["counts"]["environment_construct"] == {"attempted": 6, "completed": 0}
    assert actual["scientific_pass"] is False
