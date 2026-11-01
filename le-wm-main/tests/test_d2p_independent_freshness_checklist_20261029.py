"""Saved-only freshness guardrails; synthetic identities are not research roots."""
import hashlib
import json
import math
from pathlib import Path

import pytest

from experiments.spectral_reachability.moving_root_exclusion_manifest_20261027_v1 import (
    same_pose,
    state_identity,
)


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "le-wm-main/results/agent_p4_fresh_certificate_audit_20261028_v2_report_MAIN01.json"
PROTOCOL = ROOT / "docs/D2P_MOVING_FRESH_ROOT_PROTOCOL_20261027.md"
COVERAGE = ROOT / "le-wm-main/results/agent_freshroots_resume_20261027_coverage.json"


@pytest.mark.parametrize("path,expected", [
    (PROTOCOL, "42f1bcbf517dfd46b8ba05525663d82efac32dbb5fb24f87a8ca572bef507a2f"),
    (COVERAGE, "00a3fffeaaa8943e851d129b941a7f07feaa1db387117beb98110e4ec95f5eeb"),
    (REPORT, "15e3ece87e36eba469a0cc2188efbb7edaa94fa00f1562f9e6adc2052e4e4ae0"),
])
def test_saved_evidence_pins(path, expected):
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected


def test_saved_certificate_remains_blocked():
    report = json.loads(REPORT.read_bytes())
    assert report["status"] == "blocked_no_legal_fresh_certificate"
    assert report["freshness"]["certified_fresh_roots"] == []
    assert report["freshness"]["source_run_authorized"] is False
    assert report["history"]["coverage_complete"] is False
    assert report["history"]["original_execution_bounds_certified"] == 0
    assert report["history"]["history_certificate"] is None
    assert report["reserved_roots_opened"] is False


def test_pose_dedup_ignores_velocity_angle_wrap_and_signed_zero():
    a = state_identity([0., 2., 3., 4., 0., 0., 0.])
    b = state_identity([-0., 2., 3., 4., 2 * math.pi, 4., 5.])
    assert a["root_state_sha256"] != b["root_state_sha256"]
    assert a["physical_pose_key"] == b["physical_pose_key"]
    assert same_pose(a["physical_pose"], b["physical_pose"])


def test_pose_numeric_alias_and_non_alias():
    a = [0., 0., 0., 0., 0.]
    assert same_pose(a, [5e-10, 0., 0., 0., 5e-13])
    assert not same_pose(a, [2e-9, 0., 0., 0., 0.])
    assert not same_pose(a, [0., 0., 0., 0., 2e-12])


@pytest.mark.parametrize("state", [[0.] * 6, [0.] * 6 + [float('nan')], [0.] * 6 + [float('inf')]])
def test_invalid_state_is_rejected(state):
    with pytest.raises((ValueError, AssertionError)):
        state_identity(state)
