import copy
import math
import pytest
from experiments.spectral_reachability import agent_p1_native7_pair_decomposition_20261029_v1 as m


def fixture():
    rows=[];slots=[];pairs=[]
    for i in range(40):
        a=i//5; g=[0.,0.,0.,0.,0.,2.,-3.]; s=[float(i%5),0.,0.,0.,0.,float(i%5),0.]
        d=math.sqrt(sum((x-y)**2 for x,y in zip(s,g))); root=a//4;cond=('off','on')[(a//2)%2];role=('Z','S')[a%2]
        common=dict(index=i,root_slot=root,condition=cond,role=role,task_id=str(a),anchor_id=str(a),job_digest=str(i),plan_digest=str(i),actual_return_event_sha256=str(i))
        rows.append(dict(**common,returned_state=s,physical_goal=g,goal_binding={'state':g},boundary_trace={'20':{'state':s,'goal':g}},
            physical_metrics={'native_distance':d,'task_success':False},native_success=False,returned_reward=-d))
        slots.append(dict(**common,anchor_index=a,word=m.WORDS[i%5]))
    for a in range(8):
        for i in range(5):
            for j in range(i+1,5):
                ii=5*a+i;jj=5*a+j;delta=rows[ii]['physical_metrics']['native_distance']-rows[jj]['physical_metrics']['native_distance']
                sg=m.sign(delta); latent=sg if sg else 1
                pairs.append(dict(i=ii,j=jj,anchor_index=a,physical_lower_is_better_deltas={'native_distance':delta},
                    observed_to_physical={'native_distance':{'available':True,'sign_a':latent,'reversal':False}}))
    return dict(fixed_pairs=80,fixed_slots=40,P1_admission=False,slots=slots,pairs=pairs),dict(fixed_slots=40,P1_admission=False,rows=rows)

def test_full_80_and_nonzero_goal():
    a,b=fixture();out=m.analyze(a,b)
    assert len(out['pairs'])==80 and len(out['slots'])==40
    assert out['slots'][0]['goal7'][-2:]==[2.,-3.]
    assert out['slots'][0]['agent_velocity_squared']['numerator']=='13'

def test_missing_latent_preserves_physical_pair():
    a,b=fixture();p=a['pairs'][0];p['observed_to_physical']['native_distance']={'available':False,'sign_a':None,'reversal':None}
    out=m.analyze(a,b); assert out['pairs'][0]['original_reversal'] is None and out['pairs'][0]['native_squared'] is not None

@pytest.mark.parametrize('kind',['goal','roster','event','distance','flags','pair'])
def test_mismatch_rejected(kind):
    a,b=fixture()
    if kind=='goal': b['rows'][0]['physical_goal']=[0.]*7
    if kind=='roster': a['slots'][0]['word']='uZ'
    if kind=='event': b['rows'][0]['actual_return_event_sha256']='bad'
    if kind=='distance': b['rows'][0]['physical_metrics']['native_distance']=0.
    if kind=='flags': a['P1_admission']=True
    if kind=='pair': a['pairs'][0]['j']=4
    with pytest.raises(ValueError):m.analyze(a,b)

def test_fraction_sign_and_extreme_tie():
    assert m.unpack(m.pack(m.Fraction(1,3)))==m.Fraction(1,3)
    assert m.sign(m.Fraction(0))==0

def test_velocity_necessity_logic():
    a,b=fixture();p=a['pairs'][0]; p['observed_to_physical']['native_distance']['sign_a']=-m.sign(p['physical_lower_is_better_deltas']['native_distance'])
    p['observed_to_physical']['native_distance']['reversal']=True
    out=m.analyze(a,b)['pairs'][0]
    assert out['original_reversal'] is True and type(out['velocity_necessary_for_existing_reversal']) is bool
