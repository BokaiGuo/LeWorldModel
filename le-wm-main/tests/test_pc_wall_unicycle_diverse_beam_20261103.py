import json

from experiments.spectral_reachability import pc_wall_unicycle_diverse_beam_20261103 as exp


def test_select_diverse_keeps_best_then_farthest_with_stable_ties():
    def node(x, y, cost):
        return {"context": {"latent_history": [[x, y, 0.]]}, "cost": cost}

    nodes = [node(0, 0, 0), node(1, 0, 1), node(-1, 0, 2), node(.1, 0, .5)]
    cost = lambda context: next(n["cost"] for n in nodes if n["context"] is context)
    beam, count = exp.select_beam(nodes, cost, "diverse")
    assert beam == [nodes[0], nodes[1]] and count == 3
    beam, count = exp.select_beam(nodes, cost, "cost")
    assert beam == [nodes[0], nodes[3]] and count == 0


def test_cost_variant_matches_real_atlas_and_saved_output():
    original, cost = exp.make(20261103, "original"), exp.make(20261103, "cost")
    for key in ("programs", "selected_index", "selected", "counts", "backend"):
        assert original[key] == cost[key]
    result = json.loads((exp.ROOT / "results/pc_wall_unicycle_diverse_beam_20261103_DEV01.json").read_text())
    assert original["selected"] == result["rows"][0]["original"]["selected"]
