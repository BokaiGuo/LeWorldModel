import numpy as np
import pytest

from experiments.audit_pc_action_effect_refinement_lineage_20261104_codex import reconstruct


def test_reconstruct_matches_refinement_start_and_rejects_wrong_action():
    seed = np.zeros((15, 2), dtype=np.float32)
    winning = np.ones((15, 2), dtype=np.float32)
    pair = np.stack([winning, seed])
    row = {
        "root": 0, "arm": "random_min", "result": {
            "selected": winning.tolist(), "selected_score": 0.2, "abstained": False,
            "child": {
                "incumbent": winning.tolist(), "incumbent_source": "full_refine",
                "paired_scores": [0.2, 0.9], "abstained": False,
                "refinement_starts": [{"source": "carry", "seed_score": 0.9,
                                       "best_score": 0.2, "samples": 2,
                                       "generations": [{"samples": 1}]}],
            },
        },
    }
    calls = [
        (seed[None], None, np.array([0.9])),
        (winning[None], None, np.array([0.2])),
        (pair, None, np.array([0.2, 0.9])),
    ]
    result = reconstruct(row, calls)
    assert result["starts"][0]["incumbent_matches"] == ["generation_0"]
    calls[1] = (seed[None], None, np.array([0.2]))
    with pytest.raises(ValueError, match="incumbent not found"):
        reconstruct(row, calls)
