import json

import numpy as np

from experiments import pc_discovery_loss_attribution_posthoc as analysis


def test_loss_attribution_has_separate_generation_and_refinement_fields():
    # Contract-level test uses a saved development result when available.
    record = json.loads((analysis.panel.OUT / "plan_00_spectral.json").read_text())
    assert record["status"] == "completed"
    child = record["result"]["child"]
    layer = child["layers"][0]
    scores = np.asarray([item["score"] for item in layer["candidates"]], float)
    assert "retained" in layer and len(layer["retained"]) > 0
    assert np.isfinite(scores).all()
    assert all("source" in item and "best_score" in item
               for item in child["refinement_starts"])
