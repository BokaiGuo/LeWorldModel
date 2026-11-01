import pytest

from experiments.spectral_reachability import p1_full_bank_physical_runner_20261030_main as subject


def test_source_code_or_selection_drift_fails(monkeypatch):
    monkeypatch.setattr(subject.full_source, "configure", lambda: None)
    monkeypatch.setattr(subject.diverse, "material", lambda sha: ({"bindings": {}}, {}))
    monkeypatch.setattr(subject.diverse.runner.native, "read", lambda *args, **kwargs: {
        "full_bank_selection_sha256": "wrong", "full_bank_source_code_sha256": "wrong"})
    with pytest.raises(Exception, match="full-bank source closure"):
        subject.material("irrelevant")


def test_material_binds_runner_and_protocol(monkeypatch):
    monkeypatch.setattr(subject.full_source, "configure", lambda: None)
    monkeypatch.setattr(subject.diverse, "material", lambda sha: ({"bindings": {}}, {}))
    monkeypatch.setattr(subject.diverse.runner, "sha", lambda path: "digest")
    monkeypatch.setattr(subject.diverse.runner.native, "read", lambda *args, **kwargs: {
        "full_bank_selection_sha256": subject.full_source.full.SELECTED_SHA,
        "full_bank_source_code_sha256": "digest"})
    result, _ = subject.material("irrelevant")
    assert str(subject.PROTOCOL.resolve()) in result["bindings"]
    assert str(subject.full_source.__file__) in result["bindings"]
