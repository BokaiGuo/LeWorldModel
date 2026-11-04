import json
from pathlib import Path

import numpy as np
import pytest

from experiments.spectral_reachability import reacher_actual_rgb_pixel_metric_20261102 as metric


def test_metrics_exact_and_invalid():
    zeros = np.zeros((2, 2, 3), dtype=np.uint8)
    white = np.full_like(zeros, 255)
    assert metric.metrics(zeros, white) == {"pixel_mse": 1.0, "pixel_mae": 1.0}
    with pytest.raises(Exception):
        metric.metrics(zeros.astype(float), white)


def test_ranking_tie_and_json_serialization():
    rows = [{"order": i, "role": str(i), "test": x, "physical_final": y}
            for i, (x, y) in enumerate([(0, 1), (1, 0), (1, 2)])]
    result = metric.summarize(rows, "test")
    assert (result["concordant"], result["discordant"], result["ties"]) == (1, 1, 1)
    assert result["chosen_role"] == "0" and result["physical_regret"] == 1
    json.dumps(result)


def test_saved_result_recomputes_from_sources():
    path = metric.ROOT / "results/reacher_actual_rgb_pixel_metric_20261102_v2.json"
    if not path.exists():
        pytest.skip("development result not present")
    assert json.loads(Path(path).read_text()) == metric.run()
