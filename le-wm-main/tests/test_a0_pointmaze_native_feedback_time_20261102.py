import json
from pathlib import Path

import numpy as np

from experiments import a0_pointmaze_native_feedback_time_20261102 as run
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import sha256_file


def test_saved_feedback_decisions_and_accounting():
    result = json.loads(run.OUTPUT.read_text())
    assert result["script_sha256"] == sha256_file(Path(run.__file__))
    assert result["protocol_sha256"] == sha256_file(run.PROTOCOL)
    assert result["model_before"] == result["model_after"]
    assert [row["root"] for row in result["rows"]] == list(run.ROOTS)
    for row in result["rows"]:
        remaining, extended = row["branches"]
        assert (remaining["method"], extended["method"]) == ("remaining", "extended")
        assert remaining["first"] == extended["first"]
        assert remaining["rgb5_sha256"] == extended["rgb5_sha256"]
        for branch in (remaining, extended):
            first, initial_tail = np.unravel_index(np.argmin(branch["initial_costs"]), (8, 8))
            assert (branch["first"], branch["initial_tail"]) == (int(first), int(initial_tail))
            assert branch["physical_actions"] == 10
        assert remaining["chosen_tail"] == int(np.argmin(remaining["feedback_costs"]))
        assert extended["chosen_tail"] == int(np.unravel_index(np.argmin(extended["feedback_costs"]), (8, 8))[0])
        assert (remaining["prediction_samples"], extended["prediction_samples"]) == (72, 128)
        assert (remaining["predicted_block_transitions"], extended["predicted_block_transitions"]) == (136, 256)
