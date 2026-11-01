"""Analytic callback arithmetic tests; not neural/physical efficacy evidence."""
from copy import deepcopy
import numpy as np
import pytest
from experiments.spectral_reachability import pusht_geometry_prefix_20261027_v3 as p


def records(center=None, varying=True, residual=False, epsilons=p.EPSILONS):
    center=np.zeros(10,dtype=np.float32) if center is None else np.asarray(center,dtype=np.float32)
    probes=[center.copy()]
    for epsilon in epsilons:
        for axis in range(10):
            for sign in (-1,1):
                x=center.copy();x[axis]+=sign*epsilon;probes.append(x)
    return [dict(prefix_params=x.astype(float).tolist(),predicted_prefix=[0.],
        nominal_terminal=[float(x[0]) if residual else 0.],
        jacobian=np.asarray([[1+float(x[0]) if varying else 1.]+[0.]*9]).tolist()) for x in probes]


def test_geometry_envelope_term_is_present_and_stopk_omits_only_it():
    result=p.gradients_from_records(records(),np.ones(1),np.eye(1))
    exact=-2*p.packet.DAMPING/(1+p.packet.DAMPING)**2
    for grid in result:
        assert grid['gradients']['geometry'][0]==pytest.approx(exact,rel=.001)
        assert grid['gradients']['stopK']==[0.]*10
        assert grid['gradients']['geometry'][1:]==[0.]*9


def test_constant_response_control_keeps_residual_derivative_in_both():
    result=p.gradients_from_records(records(varying=False,residual=True),np.ones(1),np.eye(1))
    assert result[0]['gradients']['geometry']==result[0]['gradients']['stopK']
    assert result[0]['gradients']['stopK'][0]!=0


def test_center_response_is_refreshed_not_always_zero_response():
    center=[.3]+[0.]*9
    result=p.gradients_from_records(records(center),np.ones(1),np.eye(1),center=center)
    exact=-2*p.packet.DAMPING*1.3/(1.3**2+p.packet.DAMPING)**2
    assert result[0]['gradients']['geometry'][0]==pytest.approx(exact,rel=.001)


@pytest.mark.parametrize('fault',['missing','swap','wrong_center','off_axis','nonfinite'])
def test_finite_difference_identity_cannot_be_forged(fault):
    value=records()
    if fault=='missing':value.pop()
    elif fault=='swap':value[1],value[2]=value[2],value[1]
    elif fault=='wrong_center':value[0]['prefix_params'][0]=.2
    elif fault=='off_axis':value[1]['prefix_params'][1]=.01
    else:value[1]['predicted_prefix'][0]=float('nan')
    with pytest.raises(ValueError):p.gradients_from_records(value,np.ones(1),np.eye(1))


def test_normalized_update_and_fixed_starts_stay_in_ball():
    center=[p.packet.CAP/2]+[0.]*9
    for i in range(2):center=p.update(center,[-1]+[0]*9)
    assert np.linalg.norm(center)<=p.packet.CAP+1e-6
    assert p.update([0]*10,[0]*10)==[0.]*10
    with pytest.raises(ValueError):p.update([p.packet.CAP]*10,[-1]*10)


def test_optimization_trace_is_recomputed_before_bank_acceptance():
    common=dict(params=[[0.]*10,[0.]*10],kinds=['zero','dls'])
    runs=[]
    for center in p.starts(common):
        trace=[]
        for i in range(p.ITERATIONS):
            value=records(center,epsilons=p.EPSILONS[:1])
            gradients=p.gradients_from_records(value,np.ones(1),np.eye(1),center=center,epsilons=p.EPSILONS[:1])
            nxt=p.update(center,gradients[0]['gradients']['geometry'])
            trace.append(dict(center_params=center,probe_records=value,gradients=gradients,next_params=nxt));center=nxt
        runs.append(trace)
    bank=p.augmented_bank(common,runs,np.ones(1),np.eye(1),'geometry')
    assert len(bank['params'])==4
    corrupt=deepcopy(runs);corrupt[0][0]['next_params'][0]+=.1
    with pytest.raises(ValueError):p.augmented_bank(common,corrupt,np.ones(1),np.eye(1),'geometry')


def test_saved_cuda_commands_expose_numpy_f32_reference_error_not_plan_mismatch():
    import json
    from pathlib import Path
    receipt=Path(__file__).resolve().parents[1]/'results/pusht_geometry_prefix_20261027_v2_cuda_v1/report.json'
    report=json.loads(receipt.read_text())
    errors32=[];errors64=[]
    for plan in report['plans']:
        for arm in plan['arms']:
            for c in arm['candidates']:
                stored=np.asarray(c['commands'])
                ref32=np.tanh(np.concatenate([c['prefix_params'],c['suffix_params']]).astype(np.float32)).reshape(10,2)
                ref64=p.reference_commands(c['prefix_params'],c['suffix_params'])
                errors32.append(float(np.max(np.abs(stored-ref32))))
                errors64.append(float(np.max(np.abs(stored-ref64))))
    assert len(errors64)==96 and max(errors32)>1e-7 and max(errors64)<1e-7
    assert report['status']=='integrity_fail'  # original failure is never edited


def test_high_precision_reference_rejects_invalid_action_parameters():
    with pytest.raises(ValueError):p.reference_commands([float('nan')]*10,[0]*10)
    with pytest.raises(ValueError):p.reference_commands([0]*2,[0]*10)

