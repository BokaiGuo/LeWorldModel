import json
from pathlib import Path

import pytest

from experiments.spectral_reachability.pc_atlas_injection_pruning_audit_20261103_codex import audit_row, ranks


def test_ranking_exact_ties_and_invalid_input():
    assert ranks([2., 1., 1., 3.], 2) == {"strict_rank": 1, "stable_rank": 2}
    with pytest.raises(ValueError):
        ranks([float("nan")], 0)


def test_saved_injection_replay_and_pruning_accounting():
    path = Path(__file__).resolve().parents[1] / "results/pc_atlas_oracle_injection_20261103.json"
    saved = json.loads(path.read_text())["rows"][0]
    row = audit_row(saved)
    assert row["direct_sha256"] == saved["direct_sha256"]
    assert row["actual_v2_suffix_survived_beam"] and row["actual_v2_suffix_branch_winner"] and row["actual_v2_selected"]
    assert not row["actual_v2_prefix_pruned"]
    assert row["external_diagnostic_rollouts"] == len(row["prefix_candidates"]) + len(row["suffix_candidates"]) + 3
    assert row["hypothetical_global_prefix_retained"]["2"] == (row["prefix_rank"]["stable_rank"] <= 2)
