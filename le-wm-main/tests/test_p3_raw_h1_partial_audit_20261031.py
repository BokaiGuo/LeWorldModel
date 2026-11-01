import copy
import json

import numpy as np
import pytest

from experiments.spectral_reachability import p3_raw_h1_partial_audit_20261031 as audit


def test_saved_partial_batch_and_reject_corruption():
    plans, bindings, contract, data = audit.prior.initial.prepare()
    records = json.loads((audit.prior.continuation.PREVIOUS / "batch_report.json").read_bytes())["batches"]
    record = records[1]
    directory = audit.prior.continuation.PREVIOUS / "batch_01"
    result = audit.audit_batch(directory, record, plans[6:12], bindings, contract, data)
    assert len(result) == 6
    assert sum(row["status"] == "failed" and row["endpoint"] is None for row in result) == 2
    changed = copy.deepcopy(plans[6:12])
    changed[0]["commands"][0][0] += 0.01
    with pytest.raises(Exception):
        audit.audit_batch(directory, record, changed, bindings, contract, data)
    with pytest.raises(Exception):
        audit.audit_batch(directory, {**record, "report_sha256": "0"*64},
                          plans[6:12], bindings, contract, data)


def test_guard_independently_checks_canvas_pixels_and_receipt():
    pixels = np.zeros((224, 224, 3), dtype=np.uint8)
    pixels[0, 0] = [65, 105, 225]
    image = {"state": [513, 1, 2, 3, 0, 0, 0], "observation_state": [513, 1, 2, 3, 0, 0, 0]}
    guard = {"canonical_agent_pixels": 1, "render_preserved_dynamics": True, "accepted": False}
    assert audit.verify_guard(pixels, image, guard)["in_canvas"] is False
    with pytest.raises(Exception):
        audit.verify_guard(pixels, image, {**guard, "accepted": True})
    with pytest.raises(Exception):
        audit.verify_guard(pixels, image, {**guard, "canonical_agent_pixels": 2})
