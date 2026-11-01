"""Non-neural RAM callback and float32 boundary tests."""
from copy import deepcopy
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p1_fixed_prefix_fd_20261029_v1 as m
import test_agent_p1_fixed_prefix_saved_audit_20261029_v1 as fixture
import test_agent_p1_nonlinear_suffix_library_20261028_v2 as native

inputs=fixture.inputs
complete=fixture.complete
group=fixture.group
saved=fixture.saved


def test_schedule_and_actual_float32_denominators(saved):
    q,r,_,_=saved; p=m.pairs(q,r)
    assert len(p)==40 and all(v['valid'] for v in p)
    assert [(v['space'],v['requested_step'],v['coordinate']) for v in p] == [
        (space,h,j) for space in ('raw','chart_logit') for h in m.STEPS for j in range(10)]
    assert all(v['denominator']==v['plus_step']+v['minus_step'] and v['denominator']>0 for v in p)
    assert m.contract()['max_new_predict']==5120 and m.contract()['new_reverse']==0


def test_callback_real_paid_rows_and_gradient(saved):
    q,r,_,_=saved; events=native.Events(); ledger=m.core.Ledger(m.F.caps(80,0),sink=events.event)
    b=native.RAM(q['task'],events,ledger); sink=[]; out=m.evaluate(q,r,b,sink.append)
    assert out['valid_pairs']==40 and out['predict_calls']==80 and ledger.completed==m.F.caps(80,0)
    assert all(abs(p['error_vs_saved_AD'])<1e-5 for p in out['pairs'])
    assert len([e for e in sink if e['kind']=='fd_intent'])==80


def test_projection_outside_is_null_not_clipped(saved):
    q,r,_,_=saved; q=deepcopy(q); r=deepcopy(r)
    x=np.zeros(10,np.float64); x[0]=m.F.CAP
    q['suffix_logits']=x.tolist(); r['iterates'][0]['logits']=x.tolist()
    r['iterates'][0]['raw_bits']=m.F.bits(m.F.chart(x))
    ps=m.pairs(q,r); v=next(x for x in ps if x['space']=='chart_logit' and x['coordinate']==0)
    assert not v['valid'] and v['plus_commands'] is None and v['plus_norm']>m.F.CAP


def test_context_mismatch_rejected(saved):
    q,r,_,_=saved; r=deepcopy(r); r['gradients'][0]['context']['latent_history'][-1][0]+=1
    with pytest.raises(ValueError):m.pairs(q,r)
