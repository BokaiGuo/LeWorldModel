from copy import deepcopy
import pytest

from experiments.spectral_reachability import codex_p3_standalone_cost_kernel_20261030_v1 as kernel
from experiments.spectral_reachability import codex_p3_standalone_cost_saved_20261030_v1 as saved
from experiments.spectral_reachability import codex_p3_standalone_cost_batch_20261030_v1 as batch
from test_codex_p3_online_budget_20261030_v1 import Backend, fake_inputs


@pytest.mark.parametrize("point,arm", batch.SLOTS)
def test_single_process_journal_and_caps(point, arm, monkeypatch):
    class Fake(Backend):
        def jacobian(self, task, incumbent, meter):
            terminal, matrix = super().jacobian(task, incumbent, meter)
            for i in range(192):
                meter.emit("jacobian_row", row=i, array=kernel.native.blob(matrix[i]))
            return terminal, matrix
    tasks, banks, _ = fake_inputs()
    monkeypatch.setattr(kernel, "POINTS", (point,))
    monkeypatch.setattr(kernel, "ARMS", (arm,))
    monkeypatch.setattr(saved.native, "check_path", lambda *args: None)
    monkeypatch.setattr(saved.native, "signal", lambda task, *_: {"w": task["weight"]})
    monkeypatch.setattr(saved.native, "cost", lambda *args: 0.)
    events = []
    def sink(e):
        events.append(dict(e, wall_mono_ns=1000+len(events)))
    meter = kernel.Meter(sink, lambda: None, lambda: None)
    kernel.execute(tasks, banks, [Fake(), Fake()], meter, check_path=lambda *args: None,
                   signal_fn=lambda task, *_: {"w": task["weight"]}, cost=lambda *args: 0.)
    checked = saved.verify_events(events, tasks, banks, point, arm)
    report = dict(point=point, arm=arm, elapsed_seconds=1.,
                  peak_allocated_bytes=1, peak_reserved_bytes=1)
    caps = batch.check_caps(point, checked["rows"][0], report)
    assert caps["predict"] == 2*point+8
    intervals = batch.intervals(events, point, arm)
    assert intervals["search_seconds_including_journal"] > 0
    assert (intervals["signal_seconds_including_journal"] == 0) == (arm == "uniform")
    bad = deepcopy(events)
    next(e for e in bad if e["kind"] == "allocation_frozen")["quota"] = [0, 0]
    with pytest.raises(ValueError):
        saved.verify_events(bad, tasks, banks, point, arm)


def test_fairness_does_not_certify_overbudget():
    row = dict(point=20, arm="uniform", attempted={k: 0 for k in
               ("action_encoder", "predict", "predictor", "pred_proj", "jacobian",
                "vjp_directions", "grad_api", "raw_backward_hook", "svd", "metric")})
    report = dict(point=20, arm="uniform", elapsed_seconds=1., peak_allocated_bytes=1, peak_reserved_bytes=1)
    row["attempted"]["predict"] = 49
    with pytest.raises(ValueError, match="axis caps"):
        batch.check_caps(20, row, report)
