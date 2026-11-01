from copy import deepcopy
import numpy as np
import pytest
from experiments.spectral_reachability import p1_uphill_nonlinear_suffix_host_20261029 as producer
from experiments.spectral_reachability import p1_uphill_nonlinear_suffix_consume_20261029 as consumer

CAP, proposal, summarize = producer.CAP, producer.proposal, producer.summarize


def test_projected_step_uses_tanh_chain_and_ball():
    x = np.zeros(10)
    raw = np.full(10, .5, np.float32)
    updated = proposal(x, np.ones(10), raw)
    assert np.linalg.norm(updated) <= CAP
    assert (updated < 0).all()


def test_fixed_denominator_is_required():
    with pytest.raises(ValueError, match='denominator'):
        summarize({}, [])


def test_consumer_rejects_saved_objective_tamper():
    goal = np.zeros(192).tolist()
    end = np.zeros(192).tolist()
    action = np.zeros((5, 2)).tolist()
    candidate = dict(start=0, iteration=0, stage_uphill=False, suffix_solve={'params':np.zeros(10).tolist()},
                     end={'latent_history':[end]})
    report = {'rows':[dict(slot=i//2, arm=str(i%2), result=dict(candidates=[deepcopy(candidate) for _ in range(20)], baseline_terminal_Q=0.)) for i in range(8)]}
    prepared = {'requests':[dict(task=dict(goal=goal)) for _ in range(4)]}
    evaluations = [dict(iteration=i, logits=np.zeros(10).tolist(), raw=action, endpoint=end,
                        terminal_q=0., regularized=0., raw_gradient=action if i<8 else None) for i in range(9)]
    rows = [dict(slot=i//40,arm=str((i//20)%2),start=0,iteration=0,stage_uphill=False,
                 best_iteration=0,best_regularized=0.,best_terminal_q=0.,evaluations=deepcopy(evaluations)) for i in range(160)]
    result = producer.summarize(report,rows)
    result.update(source_report_sha256=consumer.source.REPORT_SHA,source_receipt_sha256=consumer.source.RECEIPT_SHA,
                  updates=8,calls={'predict':1440,'action_encoder':1440,'reverse':1280})
    assert consumer.verify(result,report,prepared)['verified_saved_arithmetic']
    result['rows'][0]['evaluations'][1]['regularized'] = 1.
    with pytest.raises(ValueError,match='objective mismatch'):
        consumer.verify(result,report,prepared)
