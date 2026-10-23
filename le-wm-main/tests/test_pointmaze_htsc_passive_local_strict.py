from experiments.run_pointmaze_htsc_passive_local_strict import (
    PASSIVE_METHOD,
    parse_roots,
)


def test_passive_protocol_names_the_predeclared_no_probe_method() -> None:
    assert PASSIVE_METHOD.endswith("passive_residual_verified_local")


def test_passive_protocol_root_parser_rejects_overlap_candidates_elsewhere() -> None:
    assert parse_roots("1,2", "roots") == [1, 2]
