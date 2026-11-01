"""Array arithmetic and immutable real saved-evidence tests. Never neural/physics."""
from copy import deepcopy
import json

import numpy as np
import pytest

from experiments.spectral_reachability import p2_composition_paired_code_audit_20261027_v1 as a

@pytest.fixture(scope='module')
def saved():
    report = json.loads((a.PROJECT/'results/p2_composition_paired_20261027_v2_prepared01/report.json').read_text())
    protocol = json.loads((a.PROJECT/'results/p2_composition_paired_20261027_v2_prepared01/protocol.json').read_text())
    source = json.loads(a.Path(protocol['context']['compilation']).read_text())
    compiled = a.keyed(source['rows'],('root_sha256',))
    row = report['rows'][0]
    candidate = row['results'][1]['attempts'][7]
    return candidate,compiled[(row['root_sha256'],)]['compiled'],protocol['context']['normalization']

def test_original_q_orientation_and_exactly_once_mean():
    q = np.eye(192)
    q[0,1] = 2
    z = np.zeros(192); z[1] = 1
    assert a.metric(z,np.zeros(192),q)==5/192
    assert a.metric(z,np.zeros(192),q)!=float(np.mean((q.T@z)**2))

def test_time_major_xy_normalization_not_repeat_interleaved():
    raw = np.arange(10,dtype=np.float32).reshape(5,2)/10
    stats = dict(mean=[.1,-.1],std=[.5,.25])
    expected = ((raw-np.array(stats['mean'],dtype=np.float32))/np.array(stats['std'],dtype=np.float32)).reshape(10)
    assert a.normalized(raw,stats)==expected.tolist()

@pytest.mark.parametrize('value',[float('nan'),float('inf'),True,'1'])
def test_reject_nonfinite_nonnumeric_arrays(value):
    with pytest.raises(ValueError):
        a.array([value],(1,))

def test_reject_duplicate_stable_root():
    with pytest.raises(ValueError,match='duplicate'):
        a.keyed([dict(root='r'),dict(root='r')],('root',))

def test_positive_improvement_and_distinct_ranking_gaps():
    gain = a.gains(dict(predicted_q_mse=3.,encoded_q_mse=1.,pose_cost=4.),
        dict(predicted_q_mse=2.,encoded_q_mse=2.,pose_cost=5.))
    assert gain['predicted_q_mse']==1
    assert gain['encoded_q_mse']==-1
    assert gain['model_vs_encoding_rank_reversal'] and gain['model_vs_pose_rank_reversal']

def test_saved_suffix_independent_kkt_and_local_diagnostic(saved):
    candidate,compiled,_ = saved
    result = a.local_diagnostic(candidate,compiled)
    assert result['scaled_kkt_residual']<1e-10
    assert result['nonlinear_suffix_q_mse']==pytest.approx(candidate['cost'],abs=1e-12)
    assert result['projected_linearization_error_mse']>0

@pytest.mark.parametrize('defect',['residual','jacobian','params','radius','normalization',
    'tanh_coordinates','nonlinear_cost','negative_multiplier','joint_prefix'])
def test_local_saved_contract_tampering_refused(saved,defect):
    candidate,compiled,_ = deepcopy(saved)
    local = candidate['suffix_compilation']
    if defect=='residual': local['nominal'][0]+=1
    if defect=='jacobian': local['jacobian'][0][0]+=1
    if defect=='params': local['solve']['params'][0]+=.1
    if defect=='radius': local['remaining_radius']+=1
    if defect=='normalization': local['solve']['ridge_penalty']*=192
    if defect=='tanh_coordinates': candidate['actions'][0][0]+=.01
    if defect=='nonlinear_cost': candidate['cost']+=1
    if defect=='negative_multiplier': local['solve']['ball_multiplier']=-1
    if defect=='joint_prefix': candidate['prefix_logits'][0]+=.1
    with pytest.raises(ValueError):
        a.local_diagnostic(candidate,compiled)

@pytest.mark.parametrize('defect',['latent','history','xy_swap','wrong_stats'])
def test_rolling_context_corruption_refused(saved,defect):
    candidate,_,stats = deepcopy(saved)
    edge = candidate['witness']['edges'][0]
    if defect=='latent': edge['output_context']['latent_history'][0][0]+=.1
    if defect=='history': edge['output_context']['action_history'][0][0]+=.1
    if defect=='xy_swap':
        edge['actions'] = [row[::-1] for row in edge['actions']]
    if defect=='wrong_stats': stats['mean'][0]+=.1
    with pytest.raises(ValueError):
        a.rolling(edge['input_context'],edge['output_context'],edge['actions'],stats)

def test_linearization_has_no_nonlinear_improvement_guarantee():
    # An exact derivative at zero can still overpredict progress for a finite step.
    # Analytic fixture only: f(x)=x-3*x*x, goal=1.
    x = 1/(1+.002)
    assert (1-x)**2<1
    assert (1-(x-3*x*x))**2>1

def test_zero_radius_branch(saved):
    candidate,compiled,_ = deepcopy(saved)
    local = candidate['suffix_compilation']
    prefix = np.zeros(10); prefix[0]=a.planner.sra.CAP
    x = np.zeros(10)
    candidate['prefix_logits']=local['prefix_logits']=prefix.tolist()
    candidate['logits']=np.r_[prefix,x].tolist()
    candidate['actions']=np.tanh(np.r_[prefix,x]).astype(np.float32).reshape(10,2).tolist()
    local['remaining_radius']=0.
    local['suffix_logits']=x.tolist()
    local['solve']=dict(params=x.tolist(),radius=0.,norm=0.,singleton_zero=True)
    assert a.local_diagnostic(candidate,compiled)['scaled_kkt_residual']==0

def test_real_evidence_no_execution_and_no_admission(monkeypatch):
    def forbidden(*args,**kw):
        pytest.fail('neural/physics execution forbidden')
    for name in ('cuda','suffix_cuda','run_arm'):
        monkeypatch.setattr(a.planner,name,forbidden)
    monkeypatch.setattr(a.planner.compiler,'make_transition',forbidden)
    monkeypatch.setattr(a.physical,'physical_branch',forbidden)
    reader = a.evidence.EvidenceReader()
    result = a.compute(a.DEFAULT_PHYSICAL,reader)
    assert result['new_neural_calls']==result['new_physical_steps']==0
    assert result['P2_admission'] is False and result['equal_total_compute'] is False
    assert result['proposed_new_planner_version'] is None
    assert len(result['roots'])==4 and sum(len(r['candidate_diagnostics']) for r in result['roots'])==160
    assert result['evidence_diagnosis']['model_to_real_q_reversal_root_indices']==[0,2]
    assert result['evidence_diagnosis']['real_q_better_but_pose_worse_root_indices']==[1,3]
    assert [r['composition_improvement']['pose_cost']>0 for r in result['roots']]==[True,False,False,False]
    assert all(r['composition_improvement']['predicted_q_mse']>0 for r in result['roots'])
    assert len(reader.hashes)>100
    reader.unchanged()

def test_cli_writes_new_seal_and_refuses_overwrite(tmp_path,monkeypatch):
    monkeypatch.setattr(a,'compute',lambda directory,reader:dict(status='fixture_only',**a.BOUNDARY))
    output = tmp_path/(a.TAG+'_fixture')
    result = a.run(a.DEFAULT_PHYSICAL,output)
    assert (output/'report.json').is_file() and result['source_seals']
    with pytest.raises(FileExistsError):
        a.run(a.DEFAULT_PHYSICAL,output)

