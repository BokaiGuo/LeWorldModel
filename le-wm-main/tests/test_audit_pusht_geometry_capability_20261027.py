import json
from pathlib import Path
import pytest
from experiments.spectral_reachability.audit_pusht_geometry_capability_20261027 import audit


def load():
    p=Path(__file__).resolve().parents[1]/'results/pusht_geometry_capability_20261027_v1_cuda_v1'
    return [json.loads((p/name).read_text()) for name in ('report.json','protocol.json')]


def test_actual_complete_diagnostic_has_zero_selected_object_response():
    r=audit(*load())
    assert r['actual_steps']==1230 and r['P1_admission'] is False
    selected=next(v for v in r['findings'] if v['role']=='selected' and v['space']=='physical_phi')
    assert selected['raw_response_norms']==[0.,0.]
    assert all(all(n>0 for n in v['raw_response_norms']) for v in r['findings'] if v is not selected)


@pytest.mark.parametrize('mutation',['duplicate','fake_command','state_drift','feature_drift','fit_drift','fd_drift','missing_grid','count_drift'])
def test_corrupted_receipt_or_algebra_fails(mutation):
    r,p=load()
    if mutation=='duplicate':r['rows'].append(r['rows'][0])
    elif mutation=='fake_command':r['rows'][0]['trace'][0]['command'][0]+=.1
    elif mutation=='state_drift':r['endpoint_receipts'][0]['state'][0]+=1
    elif mutation=='feature_drift':r['endpoint_receipts'][0]['phi'][0]+=.1
    elif mutation=='fit_drift':r['analysis']['responses'][0]['grids'][0]['own_fit']['params'][0]+=.1
    elif mutation=='fd_drift':r['analysis']['responses'][0]['grids'][0]['raw_jacobian'][0][0]+=.1
    elif mutation=='missing_grid':r['analysis']['responses'][0]['grids'].pop()
    else:r['physical_env_step_attempts']+=1
    with pytest.raises(ValueError):audit(r,p)
