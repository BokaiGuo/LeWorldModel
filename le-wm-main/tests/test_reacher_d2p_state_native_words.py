"""Small affine algebra fixtures, no neural experiment or P1 efficacy."""
import pytest
import torch

from experiments.spectral_reachability import reacher_d2p_state_native_words as compiler


class Affine:
    def __init__(self):
        self.samples = 0

    def action_encoder(self, value):
        return value

    def predict(self, emb, actions):
        self.samples += emb.shape[0]
        return emb + actions


def test_exact_legacy_compiler_and_model_accounting(monkeypatch):
    model, old_model = Affine(), Affine()
    emb = torch.zeros(1, 3, 2)
    target = torch.tensor([.3, -.2])
    history = torch.zeros(1, 3, 2)
    pixels = torch.zeros(1, 3, 1)
    lower, upper = torch.full((2,), -1.), torch.ones(2)
    monkeypatch.setattr(compiler.words.p0, "_latent_context", lambda *args: emb)
    monkeypatch.setattr(compiler.words.p0, "_goal_latent", lambda *args: target)

    def cost(model, pixels, goal, history, params, **bounds):
        return compiler.native.state_costs(model, target, emb, history, params, **bounds)

    def terminal(model, pixels, goal, history, params, **bounds):
        n = params.shape[0]
        return compiler.words.p1._terminal_from_state(model, emb.expand(n, -1, -1), history.expand(n, -1, -1), params, **bounds)

    monkeypatch.setattr(compiler.words.p0, "_cost_fn", cost)
    monkeypatch.setattr(compiler.words.p0, "_terminal_batch", terminal)
    old_bank, old = compiler.words.compile_words(old_model, pixels, None, history, lower=lower, upper=upper)
    progress = []
    bank, tails, receipt = compiler.compile_words(model, pixels, None, history, lower=lower, upper=upper,
        seed=31, progress=lambda i, n: progress.append((i, n)))
    assert torch.equal(bank, old_bank)
    assert receipt["original_rows"] == old["proposal_rows"]
    assert receipt["historical_decision"] == old["dispatch"]
    assert receipt["model_predict_sample_evaluations"] == model.samples
    assert old["model_predict_sample_evaluations"] == old_model.samples
    assert progress == [(i, 17) for i in range(1, 18)]
    assert len(tails) == len(receipt["word_records"]) == 17
    assert all(torch.equal(tails[i], torch.tensor(word["planned_commands"])) for i, word in enumerate(receipt["word_records"]))
    assert all(s["refined_goal_cost"] <= o["refined_goal_cost"] for s, o in zip(receipt["score_rows"], receipt["original_rows"]))
    assert receipt["physical_labels_used"] is False and receipt["trained_modules"] is False
    assert receipt["partition_max_abs_error"] <= 2e-6


def test_nonfinite_model_costs_are_rejected():
    with pytest.raises(ValueError):
        compiler.native.choose_state_candidate(dict.fromkeys(compiler.native.historical.NAMES, float("nan")))
