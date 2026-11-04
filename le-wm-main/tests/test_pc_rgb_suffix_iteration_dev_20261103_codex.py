import json

import numpy as np

from experiments import pc_budget_allocation_20261103_codex as old_optimizer
from experiments import pc_rgb_suffix_iteration_dev_20261103_codex as exp


def test_original_twelve_reproduced_and_second_batch_uses_elite_mean():
    prefix = np.tile([.1, -.1], (5, 1))
    def score(actions):
        return np.square(actions - .2).sum(axis=(1, 2))
    original = old_optimizer.optimize(prefix, 15, 12, np.random.default_rng(7), score)
    same = exp.optimize(prefix, 12, 12, 4, np.random.default_rng(7), score)
    np.testing.assert_array_equal(same["chain"], original[0])
    assert same["terminal_score"] == original[1]
    divided = exp.optimize(prefix, 12, 6, 2, np.random.default_rng(7), score)
    assert [len(x["scores"]) for x in divided["batches"]] == [6, 6]
    np.testing.assert_allclose(np.asarray(divided["batches"][1]["chains"])[0, 5:],
                               divided["batches"][0]["post_mean"])


def test_saved_model_and_physical_binding_and_budget():
    model, physical = json.loads(exp.MODEL.read_text()), json.loads(exp.PHYSICAL.read_text())
    assert model["protocol_sha256"] == exp.old.prior.sha(exp.PROTOCOL)
    assert physical["model_sha256"] == exp.old.prior.sha(exp.MODEL)
    assert len(model["rows"]) == len(physical["rows"]) == 36
    for row, replay in zip(model["rows"], physical["rows"], strict=True):
        assert (row["root"], row["candidate"], row["config"]) == (replay["root"], replay["candidate"], replay["config"])
        expected = {"one12": (12, 1), "two6": (12, 2), "four12": (48, 4)}[row["config"]]
        assert row["samples"] == sum(len(b["scores"]) for b in row["batches"]) == expected[0]
        assert len(row["batches"]) == expected[1]
        assert len(replay["trace"]) == 31
        positions = np.asarray([x["state"][:2] for x in replay["trace"]])
        actual = np.linalg.norm(positions - [1.8, 2.8], axis=1)[[15, 20, 25, 30]]
        np.testing.assert_allclose(actual, replay["distances"], atol=1e-12)
