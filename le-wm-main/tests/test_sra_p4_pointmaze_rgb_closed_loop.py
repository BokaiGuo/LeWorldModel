import torch

from experiments.spectral_reachability.run_sra_p4_pointmaze_rgb_closed_loop_cuda import (
    select_fixed_tree_action,
)


def test_p4_scores_both_arms_on_the_same_two_step_candidate_tensor() -> None:
    class FakeModel:
        def unroll(self, _context, act_suffix):
            first = act_suffix[0, :, 0]
            continuation = act_suffix[1, :, 0]
            # Immediate arm prefers first action 3, while two-step reachable
            # cost prefers first action 1 after optimizing its continuation.
            immediate = first - 3.0
            spectral = (first - 1.0) + 0.1 * (continuation - 4.0)
            shape = (-1, 1, 1, 1, 1)
            return {"visual": [immediate.reshape(shape), spectral.reshape(shape)]}

    candidates = torch.arange(5, dtype=torch.float32).unsqueeze(1).repeat(1, 10)
    result = select_fixed_tree_action(FakeModel(), context=None, goal=torch.zeros(1), candidates=candidates)

    assert result["candidate_rollouts"] == 25
    assert result["horizon"] == 2
    assert result["immediate_index"] == 3
    assert result["spectral_index"] == 1
