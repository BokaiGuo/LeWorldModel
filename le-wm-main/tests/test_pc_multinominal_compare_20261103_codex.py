import numpy as np
import json
from pathlib import Path

from experiments import pc_multinominal_compare_20261103_codex as run
from experiments import pc_multinominal_physical_20261103_codex as physical


def test_common_bank_design_and_total_horizon():
    def feature(action):
        return np.array([np.sum(action[:, 0]), np.sum(action[:, 1])])
    spectral, evidence = run.make_bank("spectral", feature, np.linalg.svd, 1)
    random, _ = run.make_bank("random", feature, np.linalg.svd, 1)
    nominal, _ = run.make_bank("nominal", feature, np.linalg.svd, 1)
    assert len(spectral) == len(random) == 14
    assert len(nominal) == len(evidence) == 6
    for bank in (spectral, random, nominal):
        assert sum(x["horizon"] == 5 for x in bank) == len(bank)//2
        assert sum(x["horizon"] == 10 for x in bank) == len(bank)//2
        for item in bank:
            assert np.asarray(item["prefix"]).shape == (item["horizon"], 2)
            chain, _, _ = run.base.optimize(item["prefix"], 15, 12, np.random.default_rng(5),
                                            lambda x: np.sum(x*x, axis=(1, 2)))
            assert chain.shape == (15, 2)
            np.testing.assert_allclose(chain[:item["horizon"]], item["prefix"])


def test_equal_budget_allocation_and_spectral_generation_cost():
    meter = run.base.Meter(900)
    def feature(action):
        return meter.predict(np.asarray(action)[None], lambda a: np.array([[a.sum(), a[..., 1].sum()]]))[0]
    bank, _ = run.make_bank("spectral", feature, np.linalg.svd, 1)
    assert meter.token_samples == 318
    count = ((900-meter.token_samples)//(len(bank)*36))*12
    assert count == 12
    assert meter.token_samples + len(bank)*count*3 <= 900


def test_sealed_equal_horizon_execution_and_common_suffix():
    sha = run.old.prior.sha
    root, cp10, execution, replay = (json.loads(path.read_text()) for path in
                                     (run.ROOT_PLAN, run.CP_PLAN, run.EXEC, physical.OUTPUT))
    assert root["runner_sha256"] == sha(Path(run.__file__))
    assert root["generator_sha256"] == sha(Path(run.compiler.__file__))
    assert root["protocol_sha256"] == sha(run.PROTOCOL)
    assert cp10["root_plan_sha256"] == sha(run.ROOT_PLAN)
    assert cp10["context_sha256"] == sha(run.CONTEXT)
    assert execution["cp10_sha256"] == sha(run.CP_PLAN)
    assert execution["images_sha256"] == sha(run.IMAGES)
    assert replay["execution_sha256"] == sha(run.EXEC)
    assert len(root["rows"]) == len(cp10["rows"]) == len(execution["rows"]) == 32
    assert len(replay["rows"]) == 232
    with np.load(run.CONTEXT, allow_pickle=False) as contexts, np.load(run.IMAGES, allow_pickle=False) as images:
        for row in root["rows"]:
            root_id, budget, method = row["root"], row["budget"], row["method"]
            assert row["counts"]["prediction_token_samples"] <= budget
            assert row["chosen"] == min(row["programs"], key=lambda x: x["terminal_score"])
            assert len(row["chosen"]["chain"]) == 15
            match = next(x for x in cp10["rows"] if (x["root"], x["budget"], x["method"]) == (root_id, budget, method))
            assert match["counts"]["prediction_token_samples"] == 96
            np.testing.assert_allclose(match["first10"], row["chosen"]["chain"][:10])
            assert len(match["suffix"]) == 5
            key = f"root{root_id}_b{budget}_{method}"
            assert run.old.digest_image(contexts[key][25]) == match["cp10_rgb_sha256"]
            actual = next(x for x in execution["rows"] if (x["root"], x["budget"], x["method"]) == (root_id, budget, method))
            np.testing.assert_allclose(actual["first10"], match["first10"])
            np.testing.assert_allclose(actual["suffix"], match["suffix"])
            assert actual["rgb_sha256"] == [run.old.digest_image(x) for x in images[key]]
    for row in replay["rows"]:
        assert len(row["trace"]) == (31 if row["kind"] == "feedback" else 21+row["prefix_length"])
        for tick, field in ((15, "initial_distance"), (15+row["prefix_length"], "prefix_distance"), (-1, "terminal_distance")):
            np.testing.assert_allclose(row[field], np.linalg.norm(np.asarray(row["trace"][tick]["state"][:2])-[1.8, 2.8]))
