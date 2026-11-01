from copy import deepcopy
import numpy as np
import pytest
from experiments.spectral_reachability import p1_conditional_history_arrays_main_20261027_v1 as m


def sample():
    raw=np.arange(20,dtype=np.float32).reshape(10,2)/20
    mean=np.asarray([-.0087,.0068],dtype=np.float32)
    std=np.asarray([.2019,.2002],dtype=np.float32)
    c=dict(m.META,latent_history=np.zeros((3,192)).tolist(),
        action_history=((raw-mean)/std).reshape(2,10).astype(float).tolist())
    return raw,mean,std,c


def test_float32_exact_and_no_runtime_imports():
    raw,mean,std,c=sample()
    assert len(m.check_context(c,raw,mean,std))==64
    import subprocess,sys
    subprocess.run([sys.executable,'-B','-c',
        'import sys; from experiments.spectral_reachability import '+m.NAME+
        '; assert not ({"torch","gymnasium","pymunk"}&set(sys.modules))'],check=True)


@pytest.mark.parametrize('mutation',[
    lambda c:c['action_history'][0].__setitem__(0,12345.),
    lambda c:c['action_history'].reverse(),
    lambda c:c.update(layout='wrong'),
    lambda c:c.update(context_length=2),
    lambda c:c.update(latent_history=[[0.]*192]),
])
def test_content_and_metadata_corruption_rejected(mutation):
    raw,mean,std,c=sample();mutation(c)
    with pytest.raises(ValueError):m.check_context(c,raw,mean,std)


def test_extra_div100_and_repeat_interleaving_rejected():
    raw,mean,std,c=sample()
    changed=deepcopy(c);changed['action_history']=((raw/100-mean)/std).reshape(2,10).tolist()
    with pytest.raises(ValueError):m.check_context(changed,raw,mean,std)
    changed['action_history']=((raw.reshape(2,10)-np.repeat(mean,5))/np.repeat(std,5)).tolist()
    with pytest.raises(ValueError):m.check_context(changed,raw,mean,std)
