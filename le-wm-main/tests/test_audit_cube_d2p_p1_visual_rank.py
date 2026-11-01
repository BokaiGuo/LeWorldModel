from experiments.spectral_reachability.audit_cube_d2p_p1_visual_rank import visual_ranking_gates


def _row(*, rho: float, selected: bool = True, observed_optimum: bool = True) -> dict:
    return {
        "spearman_predicted_vs_observed_visual": rho,
        "positive_control_selected_by_model": selected,
        "positive_control_observed_visual_optimum": observed_optimum,
    }


def test_visual_rank_gates_accept_the_visual_contract_only() -> None:
    gates, aggregate = visual_ranking_gates([_row(rho=0.8) for _ in range(4)])
    assert all(gates.values())
    assert aggregate["mean_spearman"] == 0.8


def test_visual_rank_gates_reject_a_reversed_root() -> None:
    gates, _ = visual_ranking_gates([_row(rho=0.8) for _ in range(3)] + [_row(rho=-0.1)])
    assert not gates["no_per_root_order_reversal"]


def test_visual_rank_gates_require_the_observed_visual_positive_control() -> None:
    gates, _ = visual_ranking_gates([_row(rho=0.8, observed_optimum=False) for _ in range(4)])
    assert not gates["positive_control_is_observed_visual_optimum_each_root"]
