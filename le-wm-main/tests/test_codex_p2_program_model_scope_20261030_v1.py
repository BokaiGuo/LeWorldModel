import pytest
from experiments.spectral_reachability import codex_p2_program_model_scope_20261030_v1 as s
from experiments.spectral_reachability import codex_p2_program_independent_budget_20261030_v1 as budget


class Component:
    def forward(self,x):return x+1


class Model:
    def __init__(self):
        self.action_encoder=Component();self.predictor=Component();self.pred_proj=Component()
    def predict(self,x):
        return self.pred_proj.forward(self.predictor.forward(self.action_encoder.forward(x)))


def test_real_entry_wrapping_and_restoration():
    model=Model();events=[];meter=budget.Budget(events.append)
    with s.model_scope(model,meter):
        assert model.predict(1)==4
        assert all(meter.returned[k]==1 for k in ("action_encoder","predict","predictor","pred_proj"))
    assert all("forward" not in vars(x) for x in (model.action_encoder,model.predictor,model.pred_proj))
    assert "predict" not in vars(model)


def test_failure_restores_every_bound_method():
    model=Model();meter=budget.Budget(lambda _:None)
    with pytest.raises(RuntimeError):
        with s.model_scope(model,meter):
            raise RuntimeError("callback")
    assert "predict" not in vars(model) and "forward" not in vars(model.pred_proj)


def test_partial_construction_restores_previous_methods():
    model=Model();model.pred_proj.forward=None
    with pytest.raises(ValueError):
        with s.model_scope(model,budget.Budget(lambda _:None)):
            pass
    assert "predict" not in vars(model) and "forward" not in vars(model.predictor)
    assert model.pred_proj.forward is None
