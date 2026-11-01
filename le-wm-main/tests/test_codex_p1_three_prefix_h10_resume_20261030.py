"""Regression tests for the H10 continuation's environment factory."""
from experiments.spectral_reachability import codex_p1_three_prefix_h10_resume_20261030 as resume


def test_factory_returns_callable_without_constructing_environment():
    factory = resume._factory()
    assert callable(factory)


def test_write_new_is_exclusive_and_returns_digest(tmp_path):
    path = tmp_path / "receipt.json"
    digest = resume._write_new(path, {"status": "test"})
    assert digest == resume.runner.runner.physical.sha(path)
    try:
        resume._write_new(path, {"status": "replacement"})
    except ValueError:
        pass
    else:
        raise AssertionError("existing receipt must not be overwritten")


def test_aggregate_recovery_compares_fixed_facts_not_resume_invocation():
    report = dict(schema="run", status="recorded_not_consumed", shards=[{"anchor": 0}],
                  run_binding_sha256="abc", fixed_branches=1008,
                  independent_confirmation=False, P1_admission=False,
                  created_anchors=[8], resumed_existing=list(range(8)))
    restarted = dict(report, created_anchors=[], resumed_existing=list(range(9)))
    assert resume._same_aggregate(report, restarted)
    assert not resume._same_aggregate(report, dict(restarted, shards=[]))
    assert not resume._same_aggregate(report, dict(restarted, P1_admission=True))
