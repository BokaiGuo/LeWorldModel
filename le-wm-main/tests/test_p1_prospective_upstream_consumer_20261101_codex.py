from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from experiments.spectral_reachability import p1_prospective_upstream_consumer_20261101_codex as subject


def root_events():
    initial = np.zeros((8, 4, 5, 2), np.float32)
    for i in range(8):
        initial[i, 0] = i / 10
    bank = initial.copy()
    scores = np.ones((8, 8), dtype=float)
    np.fill_diagonal(scores, 0.)
    selected = dict(raw=bank[0].tolist(), scores=[0.] * 8, index=0)
    events = [dict(kind=k, payload=v) for k, v in (
        ("initial_candidates", {"raw": initial.tolist()}),
        ("optimized_candidates_before_exchange", {"raw": bank.tolist()}),
        ("suffix_exchange", {"scores": scores.reshape(-1).tolist(), "tail_indices": list(range(8))}),
        ("selected_full_plan", selected))]
    receipt = dict(scores=selected["scores"], selected_index=0,
                   selected_full_plan=selected["raw"], commands=bank[0, 0].tolist())
    return events, receipt


def test_root_selection_and_prefix_protection():
    events, receipt = root_events()
    initial, selected, index = subject.root_selection(events, receipt)
    assert initial.shape == (8, 5, 2) and index == 0
    np.testing.assert_array_equal(initial[0], selected)
    bad = deepcopy(events)
    bad[1]["payload"]["raw"][1][0][0][0] += .01
    with pytest.raises(ValueError, match="protected-prefix"):
        subject.root_selection(bad, receipt)
    bad = deepcopy(events)
    bad[2]["payload"]["tail_indices"][0] = 1
    with pytest.raises(ValueError, match="argmin"):
        subject.root_selection(bad, receipt)


def test_greedy_is_same_bank_first_argmin_and_finite():
    events, receipt = root_events()
    initial, _, _ = subject.root_selection(events, receipt)
    scores = [4., 1., 1., 2., 3., 4., 5., 6.]
    binding = {"prefixes": initial.tolist(), "tie": "first_index"}
    result = dict(prefixes=initial.tolist(), scores=scores,
                  greedy_index=1, selected_prefix=initial[1].tolist())
    _, index = subject.greedy_selection(result, binding, initial)
    assert index == 1
    bad = deepcopy(result)
    bad["greedy_index"], bad["selected_prefix"] = 2, initial[2].tolist()
    with pytest.raises(ValueError, match="argmin"):
        subject.greedy_selection(bad, binding, initial)
    bad = deepcopy(result)
    bad["prefixes"][0][0][0] = .5
    with pytest.raises(ValueError, match="same-bank"):
        subject.greedy_selection(bad, binding, initial)
    bad = deepcopy(result)
    bad["scores"][0] = float("nan")
    with pytest.raises(ValueError, match="same-bank"):
        subject.greedy_selection(bad, binding, initial)


def test_source_drift_and_conflicting_bindings_fail_closed(tmp_path):
    path = tmp_path / "source"
    path.write_text("first")
    digest = subject.file_sha(path)
    assert subject.source_drift([{str(path): digest}]) == []
    path.write_text("changed")
    assert subject.source_drift([{str(path): digest}])[0]["actual"] == subject.file_sha(path)
    with pytest.raises(ValueError, match="conflicting"):
        subject.source_drift([{str(path): digest}, {str(path): "0" * 64}])
    path.unlink()
    assert subject.source_drift([{str(path): digest}])[0]["actual"] is None


def test_physical_input_binding_rejects_changed_suffix(tmp_path):
    path = tmp_path / "report.json"
    path.write_text("{}")
    digest = subject.file_sha(path)
    role = dict(prefix_raw=np.zeros((5, 2)).tolist(), own_suffix_raw=np.zeros((15, 2)).tolist(),
                cp5_pixel_sha256="a" * 64, cp5_state_identity_sha256="b" * 64)
    audit = {"role_results": {k: deepcopy(role) for k in ("S", "G", "Z")}}
    checkpoint = {"state": [1.] * 7, "goal": [2.] * 7, "pixel_sha256": "c" * 64}
    manifest = dict(roles=deepcopy(audit["role_results"]), sources={str(path): digest},
                    planning_evidence=[str(path)], root=dict(root_state=checkpoint["state"],
                    goal_state=checkpoint["goal"], warmup_rgb=[{"pixel_sha256": checkpoint["pixel_sha256"]}]))
    result = subject.bind_physical_input(audit, manifest, [(path, digest)], checkpoint)
    assert not result["confirmation_ready"] and result["cp5_dynamics_binding"] == "not_independently_verified"
    manifest["roles"]["G"]["own_suffix_raw"][0][0] = 1.
    with pytest.raises(ValueError, match="own_suffix_raw"):
        subject.bind_physical_input(audit, manifest, [(path, digest)], checkpoint)


def test_real_saved_development_bundle_never_admits_confirmation():
    base = Path(__file__).resolve().parents[1] / "results"
    items = [
        ("p1_centered_h4_coherent_20261101_launcher_structured_RUN_COHERENTMAIN01", "5be4ad99b9140af2f3e0d73389f411fb08007eb71518f1b52d9984a6aeb43a1f"),
        ("p1_coherent_same_bank_greedy_20261101_codex_RUN_MAIN01", "df39207feda1a2239c2b2bcee120a78680e99a7c91b9393c7572b53731e9cc76"),
        ("p1_coherent_own_suffix_20261101_codex_RUN_MAIN01", "e14a4e701ed4983e48801b2ed813f9daf8c0373db1771af4415118690b19cdc7"),
        ("p1_coherent_own_suffix_samebank_20261101_codex_RUN_MAIN01", "7402564db52792ed3a7027b63aa7328d6c9c7b59b0e7ca881a000a16515919e0"),
    ]
    paths = [(base / name / "report.json", sha) for name, sha in items]
    if not all(path.is_file() for path, _ in paths):
        pytest.skip("known development bundle not available")
    result = subject.inspect(*paths)
    assert result["root_selected_index"] == 6 and result["greedy_index"] == 3
    assert result["checks"]["saved_own_argmin_and_ledgers"]
    assert result["confirmation_ready"] is False and result["P1_admission"] is False
    assert result["no_physical_outcome_leakage"] == "unknown"
