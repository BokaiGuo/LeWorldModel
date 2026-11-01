"""RAM-only metric/matching tests, no env construction or model."""
from copy import deepcopy
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p1_teacher_library40_task_metric_20261028_v1 as m


def metrics(agent=0.,obj=0.):
    s=np.asarray([agent,0.,obj,0.,0.,0.,0.]);g=np.zeros(7)
    distance=float(np.linalg.norm(s-g));success=bool(np.linalg.norm(s[:4])<20)
    return m.task_metrics(s,g,[obj,0.,0.],[256.,256.,np.pi/4],success,distance,-distance)


def test_whole_body_success_not_object_only():
    a=metrics(agent=100.,obj=0.)
    assert a['object_xy_error']==0 and a['task_success'] is False
    assert a['native_coverage'] is a['object_only_coverage'] is None
    assert a['object_only_coverage_is_wrapper_success'] is False and a['outline_is_taskgoal'] is False


def test_partial_RGB_can_have_complete_physical_metrics():
    a=metrics(agent=514.7,obj=70.)
    assert a['returned_object_pose']==[70.,0.,0.]
    assert a['native_distance']>0 and a['task_success'] is False


@pytest.mark.parametrize('mutation',['pose','success','reward','distance','shape'])
def test_false_native_contract_rejected(mutation):
    s=np.zeros(7);g=np.zeros(7);b=[0.,0.,0.];success=True;distance=0.;reward=0.
    if mutation=='pose':b=[1.,0.,0.]
    elif mutation=='success':success=False
    elif mutation=='reward':reward=1.
    elif mutation=='distance':distance=1.
    else:s=s[:-1]
    with pytest.raises(ValueError):m.task_metrics(s,g,b,[256.,256.,np.pi/4],success,distance,reward)


def test_original_negative_angle_term_preserved_not_repaired():
    s=np.zeros(7);g=np.zeros(7);g[4]=10.
    out=m.task_metrics(s,g,[0.,0.,0.],[256.,256.,np.pi/4],True,10.,-10.)
    assert out['native_angle_term']<0 and out['task_success'] is True
    assert out['object_wrapped_angle_error']>0


def rows():
    out=[]
    for i in range(40):
        task,rem=divmod(i,10);role,word=divmod(rem,5)
        r=dict(index=i,root_slot=task//2,condition=('off','on')[task%2],role=('Z','S')[role],library_index=word,
            library_label=m.teacher.LIBRARY[word],anchor_id=str(i//5),task_id=str(task),goal192=[0.]*192,
            goal_binding=dict(task=task),suffix_raw_bits=m.canonical(word).hex(),physical_goal=[0.]*7,outline_pose=[256.,256.,np.pi/4],
            physical_metrics=metrics(agent=10*word,obj=0.))
        r.update(m.rank.scalar_terms(np.full(192,word+1.),None if i==12 else np.full(192,5.-word),np.zeros(192)));out.append(r)
    return out


def test_pair_axes_and_unknown_ties_are_separate():
    a=m.assemble(rows());s=a['within_role_summary']
    assert s['predicted_to_actual_latent']['fixed']==80 and s['predicted_to_actual_latent']['unknown']==4
    assert s['actual_latent_to_physical_by_metric']['native_coverage']==dict(fixed=80,available=0,unknown=80,discordant=0,ties=0)
    assert s['actual_latent_to_physical_by_metric']['object_only_coverage']['unknown']==80
    assert s['actual_latent_to_physical_by_metric']['object_xy_error']['ties']==76
    assert s['actual_latent_to_physical_by_metric']['all_xy_error']['discordant']==76
    assert a['same_word_ZS_summary']['predicted_to_actual_latent']['fixed']==20
    assert a['same_word_ZS_summary']['predicted_to_actual_latent']['unknown']==1
    assert a['rows'][12]['actual_Q'] is None and a['rows'][12]['physical_metrics']['native_distance'] is not None
    assert all(p['C2_failure_inferred'] is False for g in a['groups'] for p in g['pairs'])


def test_zero_tie_not_reversal_success_orientation():
    r=rows();r[0]['actual_Q']=r[1]['actual_Q']=1.
    p=m.compare(r[0],r[1],'within_role_library')
    assert p['predicted_to_actual_latent']['reversal'] is False
    assert p['actual_latent_to_physical_by_metric']['task_success']['orientation']==-1


@pytest.mark.parametrize('mutation',['root','condition','goal','outline','task','drop','fill12'])
def test_fixed_matching_not_relaxed(mutation):
    r=rows()
    if mutation=='root':r[1]['root_slot']=1
    elif mutation=='condition':r[1]['condition']='on'
    elif mutation=='goal':r[1]['physical_goal'][0]=1.
    elif mutation=='outline':r[1]['outline_pose'][0]=0.
    elif mutation=='task':r[1]['task_id']='other'
    elif mutation=='drop':r.pop()
    else:r[12]['actual_Q']=0.
    with pytest.raises(ValueError):m.assemble(r)
