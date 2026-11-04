import json
from pathlib import Path

import numpy as np

from experiments import pc_root3_fixed_chain_diagnostic_20261103_codex as run


def test_fixed_chain_rgb_physics_and_shared_search_prefix():
    sha = run.old.prior.sha
    result = json.loads(run.OUTPUT.read_text())
    original = json.loads(run.base.ROOT_PLAN.read_text())
    assert result["prior_sha256"] == sha(run.base.ROOT_PLAN)
    assert result["protocol_sha256"] == sha(run.PROTOCOL)
    assert result["runner_sha256"] == sha(Path(run.__file__))
    assert result["images_sha256"] == sha(run.IMAGES)
    with np.load(run.IMAGES, allow_pickle=False) as images:
        for row in result["rows"]:
            plan = next(x for x in original["rows"] if x["root"] == 3 and x["method"] == row["method"])
            np.testing.assert_allclose(row["actions"], plan["chosen"]["prefix"]+plan["chosen"]["suffix"])
            assert row["model_terminal_score"] == plan["chosen"]["terminal_score"]
            assert row["rgb_sha256"] == [run.old.digest_image(x) for x in images[row["method"]]]
            assert len(row["trace"]) == 31
            np.testing.assert_allclose(row["physical_terminal_distance"],
                                       np.linalg.norm(np.asarray(row["trace"][-1]["state"][:2])-[1.8, 2.8]))
    spec = next(x for x in original["rows"] if x["root"] == 3 and x["method"] == "spectral")
    plain = next(x for x in original["rows"] if x["root"] == 3 and x["method"] == "no_spectral")
    s = next(x for x in spec["candidates"] if x["name"] == "direction_2")
    n = next(x for x in plain["candidates"] if x["name"] == "direction_2")
    assert s["scores"] == n["scores"][:24]
    assert n["terminal_score"] < s["terminal_score"]
