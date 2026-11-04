import json

import numpy as np
import pytest

from experiments import a0_pointmaze_frame25_audit_20261102_codex_exclusive as audit
from experiments import a0_pointmaze_frame25_contact_audit_20261102_codex_exclusive as contact


def test_real_frame25_versions_and_state_restoration():
    expected = json.loads((audit.ROOT / "results" /
                           (audit.NAME + "_RESULT01.json")).read_bytes())
    result = audit.build()
    assert result == expected
    old, modern = result["versions"]
    assert (old["runtime"]["mujoco"], modern["runtime"]["mujoco"]) == ("2.2.0", "3.1.6")
    assert old["runtime"]["mjtNum_bytes"] == modern["runtime"]["mjtNum_bytes"] == 8
    for version in (old, modern):
        for row in version["runs"]:
            p = row["predictions"]
            assert p["continuous"] == p["full_clone"] == p["qv_from_continuous"]
    first = old["runs"][0]["predictions"]
    assert first["continuous"]["26"]["state_l2"] < audit.prior.STATE_TOLERANCE
    assert first["qv_from_archive"]["26"]["state_l2"] > audit.prior.STATE_TOLERANCE
    assert modern["runs"][-1]["predictions"]["qv_from_archive"]["26"]["state_l2"] > 0.04
    assert not result["P1_gate_changed"]


def test_contact_trace_and_archive_are_reproducible():
    expected = json.loads((audit.ROOT / "results" /
                           (contact.NAME + "_RESULT01.json")).read_bytes())
    result = contact.build()
    assert result == expected
    old = result["results"][0]
    a, b = old["traces"]["continuous"], old["traces"]["archive"]
    assert len(a) == len(b) == 7
    assert a[4]["ncon"] == b[4]["ncon"] == 1
    assert a[5]["ncon"] == b[5]["ncon"] == 2
    before = np.asarray(a[4]["state"]) - np.asarray(b[4]["state"])
    after = np.asarray(a[5]["state"]) - np.asarray(b[5]["state"])
    assert abs(before[3]) < 1e-6 and abs(after[3]) > 2e-6


def test_wrong_audit_anchor_rejected(monkeypatch):
    monkeypatch.setattr(contact, "RESULT_SHA", "0" * 64)
    with pytest.raises(ValueError, match="frame25 audit identity"):
        contact.build()
