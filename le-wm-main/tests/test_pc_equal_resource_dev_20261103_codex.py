import json
from pathlib import Path

import numpy as np
import pytest

from experiments import pc_equal_resource_dev_20261103_codex as run
from experiments import pc_equal_resource_physical_20261103_codex as physical


def test_resource_meter_counts_tokens_and_fails_closed():
    meter = run.Meter(4)
    model = lambda x: x
    meter.predict(np.zeros((2, 10, 2)), model)
    assert meter.as_dict()["prediction_token_samples"] == 4
    with pytest.raises(RuntimeError, match="budget"):
        meter.predict(np.zeros((1, 5, 2)), model)
    assert meter.calls == 1


def test_sealed_plans_feedback_binding_and_physical_results():
    sha = run.old.prior.sha
    root, cp5, execution, replay = (json.loads(p.read_text()) for p in
                                    (run.ROOT_PLAN, run.CP_PLAN, run.EXEC, physical.OUTPUT))
    assert root["runner_sha256"] == sha(Path(run.__file__))
    assert root["protocol_sha256"] == sha(run.PROTOCOL)
    assert cp5["root_plan_sha256"] == sha(run.ROOT_PLAN)
    assert cp5["context_sha256"] == sha(run.CONTEXT)
    assert execution["cp5_sha256"] == sha(run.CP_PLAN)
    assert execution["images_sha256"] == sha(run.IMAGES)
    assert replay["execution_sha256"] == sha(run.EXEC)
    assert len(root["rows"]) == len(cp5["rows"]) == len(execution["rows"]) == len(replay["rows"]) == 16
    with np.load(run.CONTEXT, allow_pickle=False) as contexts, np.load(run.IMAGES, allow_pickle=False) as images:
        for row in root["rows"]:
            root_id, method = row["root"], row["method"]
            assert row["counts"]["prediction_token_samples"] <= 1300
            assert row["encoding_calls"] == 3
            assert row["chosen"] == min(row["candidates"], key=lambda x: x["terminal_score"])
            if method == "greedy":
                assert row["chosen"]["name"] == min(row["prefix_evaluations"], key=lambda x: x["score"])["name"]
            next_row = next(x for x in cp5["rows"] if x["root"] == root_id and x["method"] == method)
            np.testing.assert_allclose(next_row["prefix"], row["chosen"]["prefix"])
            assert next_row["counts"]["prediction_token_samples"] == 192
            assert next_row["encoding_calls"] == 3
            key = f"root{root_id}_{method}"
            assert run.old.digest_image(contexts[key][20]) == next_row["cp5_rgb_sha256"]
            actual = next(x for x in execution["rows"] if x["root"] == root_id and x["method"] == method)
            np.testing.assert_allclose(actual["prefix"], next_row["prefix"])
            np.testing.assert_allclose(actual["suffix"], next_row["suffix"])
            assert [run.old.digest_image(x) for x in images[key]] == actual["rgb_sha256"]
            result = next(x for x in replay["rows"] if x["root"] == root_id and x["method"] == method)
            assert len(result["trace"]) == 31
            for index, name in ((15, "initial_distance"), (20, "prefix_distance"), (30, "terminal_distance")):
                np.testing.assert_allclose(result[name], np.linalg.norm(np.asarray(result["trace"][index]["state"][:2])-[1.8, 2.8]))
