import json
import subprocess
import sys
import pytest
from experiments.spectral_reachability import agent_p3_c3_delivery_budget_exploratory_20261029_v1 as b

def test_all_axes_and_fixed_roster():
    p=b.proof(); assert len(p['slots'])==6 and len(p['common_cap_per_slot'])==27
    assert p['common_cap_per_slot']['program_compile_calls']==170
    assert p['common_cap_per_slot']['predict_API']==1072
    assert p['common_cap_per_slot']['VJP_directions']==35904
    assert p['common_cap_per_slot']['replay_calls']==308
    assert not p['guarantees_model_witness'] and not p['physical_release']

@pytest.mark.parametrize('h,w',[(1,4),(1,8),(3,4),(3,8)])
def test_atomic_reserve_all_prefixes(h,w):
    s,r,v,c=b.signal(),b.reconfiguration(3),b.verification(9),b.cap()
    for n in range(17):
        spent=b.add({k:n*(s[k]+r[k]) for k in b.AXES},s)
        for q in (b.progress(h,w),b.reconfiguration(h)):
            assert b.fits(spent,q,v,c)=={}

def test_tampered_cap_and_missing_axis_rejected():
    c=b.cap(); c['program_compile_calls']=169
    s,r=b.signal(),b.reconfiguration(3)
    spent=b.add({k:16*(s[k]+r[k]) for k in b.AXES},s)
    assert b.fits(spent,r,b.verification(9),c)=={'program_compile_calls':1}
    c.pop('predict_API')
    with pytest.raises(ValueError): b.fits(b.vector(),b.vector(),b.vector(),c)

def test_invalid_shape_and_drift(tmp_path):
    with pytest.raises(ValueError): b.progress(2,4)
    with pytest.raises((ValueError,FileNotFoundError)): b.proof(tmp_path)

def test_cli():
    out=subprocess.check_output([sys.executable,'-m',b.__name__,'proof'],cwd=b.ROOT,text=True)
    assert json.loads(out)['atomic_queue_budget_proved'] is True
