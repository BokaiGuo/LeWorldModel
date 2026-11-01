import json

from experiments.spectral_reachability import codex_p1_cp5_twenty_step_cuda_20261030_main as trial


def test_twenty_step_contract_and_source_binding(monkeypatch):
    base = trial.base
    monkeypatch.setattr(base, "NAME", base.NAME)
    monkeypatch.setattr(base, "CONFIG", base.CONFIG)
    monkeypatch.setattr(base, "bindings", base.bindings)
    monkeypatch.setattr(base, "make_contract", base.make_contract)
    trial.install()
    bundle = json.loads(base.BUNDLE.read_bytes())
    sources = base.bindings()
    policy, contract = base.make_contract(bundle["rows"][0], sources)
    caps = contract["levels"][0]["caps"]
    assert base.CONFIG == trial.CONFIG
    assert contract["identity"]["optimizer_config"] == base.spec.digest(trial.CONFIG)
    assert caps["action_encode_calls"] == caps["predict_calls"] == caps["predict_samples"] == 42
    assert caps["backward_calls"] == caps["vjp_rows"] == 20
    assert caps["replay_calls"] == caps["selection_evaluations"] == 1
    assert contract["levels"][1]["caps"] == {k: v + 1 for k, v in caps.items()}
    assert policy["metric"] == "terminal_latent_mean_squared_error"
    assert str(trial.Path(trial.__file__).resolve()) in sources
