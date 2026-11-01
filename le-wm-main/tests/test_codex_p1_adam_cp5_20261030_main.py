import hashlib
import json

import numpy as np
import pytest

from experiments.spectral_reachability import codex_p1_adam_cp5_20261030_main as adam
from experiments.spectral_reachability import codex_p1_multistart_cp5_20261030_main as old


def test_fixed_adam_result_matches_initialization_and_action_contract():
    root = old.base.ROOT / "results"
    path = root / "codex_p1_adam_cp5_20261030_main_RUN01.json"
    if not path.is_file():
        pytest.skip("host CUDA result unavailable")
    result = json.loads(path.read_bytes())
    baseline = json.loads((root / "codex_p1_multistart20_cp5_20261030_main_RUN01.json").read_bytes())
    assert len(result["rows"]) == len(baseline["rows"]) == 24
    assert result["sources"][str(adam.Path(adam.__file__).resolve())] == hashlib.sha256(
        adam.Path(adam.__file__).read_bytes()).hexdigest()
    for i, (row, previous) in enumerate(zip(result["rows"], baseline["rows"], strict=True)):
        assert row["index"] == previous["index"] == i
        assert row["initial_f32_sha256"] == previous["initial_f32_sha256"]
        assert row["initial_f32_sha256"] == hashlib.sha256(old.initial_actions(row["logical_pair"]).tobytes()).hexdigest()
        actions = np.asarray(row["selected_actions"], dtype="<f4")
        assert actions.shape == (2, 5, 2) and np.isfinite(actions).all() and (np.abs(actions) <= 1).all()
        assert hashlib.sha256(actions.tobytes()).hexdigest() == row["selected_action_sha256"]
        assert row["selected_index"] == old.pick(row["candidate_scores"])
