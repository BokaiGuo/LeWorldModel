import numpy as np
import pytest

from experiments.spectral_reachability.d2p_atlas_search_v2 import AtlasConfig
from experiments.spectral_reachability.d2p_composition_witness_v2 import digest
from experiments.spectral_reachability.d2p_local_compiler import CompilerConfig
from experiments.spectral_reachability.pc_atlas_depth3_pruning_alternative_20261103_codex import (
    IntermediatePolicyAtlas, ROOT, choose_intermediate, new_compiler, run_pair,
)
from experiments.spectral_reachability.pc_exact_unicycle_development import TURN


def cand(cost, lookahead, action):
    return {"cost": cost, "lookahead_cost": lookahead, "last_action": action}


def test_policy_ties_diversity_threshold_and_fallback():
    candidates = [cand(1., 1., [0., 0.]), cand(1., 1.005, [.1, 0.]),
                  cand(2., 1.006, [.9, 0.]), cand(3., 1.02, [1., 1.])]
    rng = np.random.default_rng(5)
    assert choose_intermediate(candidates, "v2", rng, [2., 2.], .01) == [0, 1]
    assert choose_intermediate(candidates, "lookahead", rng, [2., 2.], .01) == [0, 1]
    assert choose_intermediate(candidates, "lookahead_diverse", rng, [2., 2.], .01) == [0, 2]
    assert choose_intermediate(candidates, "lookahead_diverse", rng, [2., 2.], .001) == [0, 1]
    tie = [cand(1., 1., [0., 0.]), cand(1., 1., [1., 0.]), cand(1., 1., [-1., 0.])]
    assert choose_intermediate(tie, "lookahead_diverse", rng, [2., 2.], 0.) == [0, 1]
    with pytest.raises(ValueError):
        choose_intermediate(candidates, "lookahead", rng, [0., 2.], .01)
    with pytest.raises(ValueError):
        choose_intermediate([cand(1., None, [0., 0.])] * 2, "lookahead", rng, [2., 2.], .01)


@pytest.fixture(scope="module")
def pair():
    return run_pair(2, 44)


def test_real_v2_parity_and_matched_all_call_and_step_budgets(pair):
    assert pair["original_v2_parity"]
    first = pair["arms"]["v2"]
    for result in pair["arms"].values():
        assert result["operation_counts"] == first["operation_counts"]
        assert result["backend_action_steps"] == first["backend_action_steps"]
        assert result["operation_counts"]["rollout_attempted"] == result["operation_counts"]["rollout_returned"]
        assert result["backend_action_steps"]["jacobian_attempted"] == result["backend_action_steps"]["jacobian_returned"]
    assert pair["original_v2_counts"]["rollout_attempted"] < first["operation_counts"]["rollout_attempted"]


def test_original_bank_only_bounded_actions_and_full_witness(pair):
    compiler, backend = new_compiler(pair["goal"])
    for result in pair["arms"].values():
        for trace in result["intermediate_trace"]:
            assert len(trace["selected_indices_in_beam_order"]) == 2
            for node in trace["candidates"]:
                assert all(not name.startswith("audit_") and not name.startswith("injected")
                           for name in node["names"])
                actions = np.asarray(node["actions"])
                assert np.all(actions >= [-1., -TURN]) and np.all(actions <= [1., TURN])
        for program in result["programs"]:
            actions = np.asarray(program["actions"])
            assert actions.shape == (3, 2)
            assert np.all(actions >= compiler.low) and np.all(actions <= compiler.high)
            assert digest(backend.rollout(ROOT, actions)) == digest(program["output_context"])


def test_seed_reproducibility_and_unpadded_control(pair):
    assert run_pair(2, 44) == pair
    c, _ = new_compiler(pair["goal"])
    unpadded = IntermediatePolicyAtlas(c, AtlasConfig(depth=3, beam_width=2, seed=44),
                                       mode="v2", pad_lookahead=False).search(ROOT)
    assert unpadded["operation_counts"] == pair["original_v2_counts"]
    assert all(n["lookahead_cost"] is None for t in unpadded["intermediate_trace"] for n in t["candidates"])


def test_budget_exhaustion_fails_without_partial_result():
    compiler, _ = new_compiler([1., 0.])
    compiler.config = CompilerConfig(packet_steps=1, max_rollout_operations=1)
    search = IntermediatePolicyAtlas(compiler, AtlasConfig(depth=3, beam_width=2),
                                    mode="lookahead_diverse", pad_lookahead=True)
    with pytest.raises(RuntimeError, match="cap exhausted"):
        search.search(ROOT)


def test_rejects_unsupported_depth_or_policy():
    compiler, _ = new_compiler([1., 0.])
    with pytest.raises(ValueError):
        IntermediatePolicyAtlas(compiler, AtlasConfig(depth=2), mode="v2")
    with pytest.raises(ValueError):
        IntermediatePolicyAtlas(compiler, AtlasConfig(depth=3), mode="injected")
