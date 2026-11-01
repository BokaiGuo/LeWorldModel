"""Offline plan/source closure checks; never enter CUDA or native runtime."""
import hashlib
import json
from pathlib import Path

import pytest

from experiments.spectral_reachability import p3_decision_host_cuda_driver_20261030_codex_v1 as d

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "results/p3_decision_host_cuda_resource_plan_20261030_codex_PREP01.json"
CANONICAL_SHA = "4eae543e09ea63aec94814f394205ffd3ff4ddc71025d0eb544bd39a5722719f"
FILE_SHA = "0fa0ef09f27b7f29558c10014e64cc3d940cd208a2cc0819d56163786953e541"


def load():
    return json.loads(PLAN.read_text())


def test_frozen_plan_and_real_source_closure():
    p = load()
    assert d.file_sha(PLAN) == FILE_SHA
    assert hashlib.sha256(d.control.core.canonical(p)).hexdigest() == CANONICAL_SHA
    _, prepared, source = d.validate(p)
    assert len(prepared["tasks"]) == len(source["roots"]) == 2
    assert sum(p["native_limits"].values()) <= p["native_cap"] - p["native_reserve"]
    assert p["native_limits"]["physical_png"] >= 192 * 256 * 1024


@pytest.mark.parametrize("field", ["driver", "adapter", "decision_journal", "native_transport"])
def test_each_pinned_implementation_rejects_tampering(field):
    p = load()
    p["files"][field] = "0" * 64
    with pytest.raises(ValueError, match="file pin"):
        d.validate(p)


def test_source_contract_and_budget_tamper_rejected():
    p = load()
    p["native_limits"]["physical_png"] = p["native_cap"]
    with pytest.raises(ValueError, match="consistent limits"):
        d.validate(p)
    p = load()
    p["source_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="native source contract pin"):
        d.validate(p)
