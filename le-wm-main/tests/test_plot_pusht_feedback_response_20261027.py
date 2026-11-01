import json
from pathlib import Path
import pytest
from experiments.spectral_reachability.plot_pusht_feedback_response_20261027 import summarize


def load():
    d=Path(__file__).resolve().parents[1]/'results/pusht_geometry_feedback_response_20261027_v1_prepare_v1'
    return [json.loads((d/name).read_text()) for name in ('report.json','protocol.json')]


def test_nominal_repairs_not_universal_response_repairs():
    result=summarize(*load())
    idx={(r['role'],r['arm']):r for r in result['rows']}
    assert all(idx[r,'actual']['zero_prediction_mse']<idx[r,'imagined']['zero_prediction_mse'] for r in ('selected','greedy','zero'))
    assert idx['selected','actual']['response_cosines'][0]<0
    assert result['P1_admission'] is False


@pytest.mark.parametrize('mutation',['missing_arm','fake_metric','fake_physical','fake_counter'])
def test_bad_receipt_refused(mutation):
    r,p=load()
    if mutation=='missing_arm':r['rows'].pop()
    elif mutation=='fake_metric':r['rows'][0]['analysis']['probes'][0]['projected_mse']+=1
    elif mutation=='fake_physical':r['rows'][0]['analysis']['probes'][0]['physical'][0]+=1
    else:r['image_work']['frames']+=1
    with pytest.raises(ValueError):summarize(r,p)
