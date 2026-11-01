from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from experiments.spectral_reachability import p1_common_budget_node_20261101_leibniz as m
from experiments.spectral_reachability import codex_four_arm_common_suffix_control_20261030 as structured
from experiments.spectral_reachability import codex_p1_variable_cem_20261101_exclusive as cem


class Analytic:
    def encode(self, info):
        return info

    def action_encoder(self, x):
        return x

    def predict(self, x, actions):
        return x + actions


@pytest.fixture
def inputs(tmp_path):
    root = Path(__file__).resolve().parents[1]
    prepare = root / "results/p1_first_feedback_carry_20261101_main_root0_PREP_MAIN02.json"
    task = json.loads(prepare.read_bytes())["source"]["tasks"][0]
    _, s, _ = structured.contract_for(task, "a" * 64, "b" * 64)
    templates = {"structured": {b: deepcopy(s) for b in m.BOUNDARIES},
                 "variable_cem": {b: cem.node_contract(task, "a" * 64, "b" * 64, b, seed=20261101)
                                  for b in m.BOUNDARIES}}
    extra = tmp_path / "source"
    extra.write_text("frozen")
    paths = [Path(x.__file__).resolve() for x in (m, m.spec, m.durable, m.leafmod)] + [extra]
    sources = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    protocol = m.make_protocol(templates, sources)
    return templates, protocol, extra


def test_both_real_contracts_equal_cap_all_boundaries(inputs):
    templates, protocol, _ = inputs
    original = deepcopy(templates)
    for arm in m.ARMS:
        for b in m.BOUNDARIES:
            result = m.bind(protocol, m.spec.digest(protocol), arm, b, templates[arm][b])
            assert result["levels"][0]["caps"] == m.CAPS
            assert result["identity"] == templates[arm][b]["identity"]
    assert templates == original


@pytest.mark.parametrize("arm", m.ARMS)
def test_actual_leaf_dispatch_blocks_before_extra_call(tmp_path, inputs, arm):
    templates, protocol, _ = inputs
    driver, leaf, contract = m.open_node(tmp_path / "ledger", protocol, m.spec.digest(protocol),
                                         arm, 0, templates[arm][0], Analytic(), analytic_fixture=True)
    # Real durable reservation and real leaf predict dispatch, no neural inference.
    reservation = dict.fromkeys(m.spec.AXES, 0)
    reservation.update(predict_calls=195, predict_samples=195)
    driver.dispatch("verification", "predict", reservation, lambda: None)
    with pytest.raises(ValueError, match="budget exceeded"):
        leaf.predict(torch.zeros(1, 1), torch.zeros(1, 1), phase="verification")
    receipt = driver.finish()
    assert receipt["status"] == "failed_or_blocked"
    events = [json.loads(line) for line in driver.path.read_bytes().splitlines()]
    assert [e["kind"] for e in events] == ["header", "intent", "return", "blocked", "terminal"]
    assert events[0]["payload"]["contract"] == contract


@pytest.mark.parametrize("defect", ["identity", "optimizer", "roster", "protocol", "source"])
def test_reject_drift(inputs, defect):
    templates, protocol, extra = inputs
    expected = m.spec.digest(protocol)
    t = deepcopy(templates["structured"][0])
    if defect == "identity":
        t["identity"]["root"] = "f" * 64
    elif defect == "optimizer":
        t["identity"]["optimizer"] = "f" * 64
    elif defect == "roster":
        with pytest.raises(ValueError):
            m.bind(protocol, expected, "structured", 20, t)
        return
    elif defect == "protocol":
        protocol["caps"]["predict_calls"] = 196
    else:
        extra.write_text("changed")
    with pytest.raises(ValueError):
        m.bind(protocol, expected, "structured", 0, t)


def test_cpu_module_forbidden_and_driver_closed(tmp_path, inputs):
    templates, protocol, _ = inputs
    class CPUModel(torch.nn.Module):
        def encode(self, x): return x
        def action_encoder(self, x): return x
        def predict(self, x, y): return x
    # A parameter makes the CPU device violation observable.
    model = CPUModel()
    model.register_parameter("weight", torch.nn.Parameter(torch.ones(1), requires_grad=False))
    model.eval()
    with pytest.raises(ValueError, match="CUDA"):
        m.open_node(tmp_path / "ledger", protocol, m.spec.digest(protocol), "structured", 0,
                    templates["structured"][0], model)
    events = [json.loads(line) for line in (tmp_path / "ledger/events.jsonl").read_bytes().splitlines()]
    assert events[-1]["payload"]["status"] == "failed_or_blocked"


def test_realized_partial_stop_does_not_pad_or_replan():
    episode = m.EpisodeOrder()
    episode.begin(0)
    episode.returned(5, False)
    episode.begin(5)
    episode.returned(1, True)
    assert episode.steps == 6 and episode.stopped
    with pytest.raises(ValueError):
        episode.begin(10)


def test_bad_partial_receipt_fails_stopped():
    episode = m.EpisodeOrder()
    episode.begin(0)
    with pytest.raises(ValueError):
        episode.returned(2, False)
    assert episode.steps == 0 and episode.stopped
    with pytest.raises(ValueError):
        episode.begin(0)


def test_unsolicited_receipt_fails_stopped():
    episode = m.EpisodeOrder()
    with pytest.raises(ValueError):
        episode.returned(5, False)
    assert episode.stopped
    with pytest.raises(ValueError):
        episode.begin(0)
