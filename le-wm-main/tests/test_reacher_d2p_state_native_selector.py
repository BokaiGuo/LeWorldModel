"""Identity/seal contract fixtures, not neural experiments."""
import copy

import pytest

from experiments.spectral_reachability.audit_reacher_d2p_state_native_selector import join_reference


def fixture():
    seeds = list(range(8))
    rows = [{"seed": seed, "source_episode": seed + 10, "source_start_row": 10,
        "branches": {name: {"seed": seed, "family": name} for name in ("dispatch", "greedy", "zero", "cem")}} for seed in seeds]
    report = {"schema": "reacher_d2p_suffix_multistart_confirmation.v1", "status": "completed",
        "gates": dict.fromkeys(("cuda_only", "weights_unchanged", "all_declared_roots", "source_episodes_disjoint", "all_execution_gates",
            "all24_branches_per_root", "prefix_seal_unchanged", "suffix_seal_unchanged", "implementation_unchanged"), True),
        "sealed_protocol": {"root_seeds": seeds}, "rows": rows}
    prefixes = {"rows": [{key: row[key] for key in ("seed", "source_episode", "source_start_row")} for row in reversed(rows)]}
    suffixes = {"rows": [copy.deepcopy(branch) for row in reversed(rows) for branch in row["branches"].values()]}
    return report, prefixes, suffixes


def test_all_root_family_pairs_join_by_identity():
    joined = join_reference(*fixture())
    assert [r["seed"] for r, _ in joined] == list(range(8))
    assert all(r["seed"] == p["seed"] for r, p in joined)


@pytest.mark.parametrize("kind", ["truthy", "missing_gate", "status", "duplicate_prefix", "duplicate_suffix", "missing_root", "source", "suffix_record"])
def test_fail_closed_reference_contract(kind):
    report, prefixes, suffixes = fixture()
    if kind == "truthy":
        report["gates"]["cuda_only"] = 1
    elif kind == "missing_gate":
        del report["gates"]["weights_unchanged"]
    elif kind == "status":
        report["status"] = "running"
    elif kind == "duplicate_prefix":
        prefixes["rows"][0] = copy.deepcopy(prefixes["rows"][1])
    elif kind == "duplicate_suffix":
        suffixes["rows"][0] = copy.deepcopy(suffixes["rows"][1])
    elif kind == "missing_root":
        report["rows"].pop()
    elif kind == "source":
        prefixes["rows"][0]["source_episode"] += 1
    else:
        suffixes["rows"][0]["receipt"] = "changed"
    with pytest.raises(ValueError):
        join_reference(report, prefixes, suffixes)
