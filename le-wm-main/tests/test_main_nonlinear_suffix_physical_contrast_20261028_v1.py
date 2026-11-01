from copy import deepcopy
import pytest
from experiments.spectral_reachability import main_nonlinear_suffix_physical_contrast_20261028_v1 as m


def saved():
    a=m.base.read(m.ROOT/'results/agent_allmode_physical_20261028_v1_MAIN01/report.json',m.OLD_SHA)
    b=m.base.read(m.ROOT/'results/agent_nonlinear_suffix_physical_20261028_v1_MAIN01/report.json',m.NEW_SHA)
    return a['rows'],b['rows']


@pytest.mark.parametrize('index',range(4))
def test_real_fixed_saved_native_metrics(index):
    a,b=saved();r=m.compare(a[index],b[index])
    assert r['predicted_Q_delta']<0
    assert not r['new']['native_success']


@pytest.mark.parametrize('damage',['goal','time','label','task'])
def test_wrong_task_time_and_label_rejected(damage):
    a,b=saved();x=deepcopy(b[0])
    if damage=='goal':x['goal'][0]+=1
    elif damage=='time':x['own_returned']=5
    elif damage=='label':x['native_success']=True
    else:x['plan']['task_sha256']='0'*64
    with pytest.raises(ValueError):m.compare(a[0],x)
