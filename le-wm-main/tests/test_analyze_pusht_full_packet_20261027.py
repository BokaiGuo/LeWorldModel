import json
from copy import deepcopy
from pathlib import Path
import pytest
from experiments.spectral_reachability.analyze_pusht_full_packet_20261027 import summarize


def actual():
    root=Path(__file__).resolve().parents[1]/'results'
    return [json.loads((root/p/'report.json').read_text()) for p in (
        'pusht_full_packet_compiler_20261027_cuda_v1','pusht_full_packet_physical_20261027_cuda_v1')]


def test_actual_saved_census_retains_original_failed_gate():
    result=summarize(*actual())
    assert result['original_physical_status']=='integrity_fail'
    assert result['original_gates']['encodings_complete'] is False
    assert len(result['missing_endpoints_explained_by_guard'])==3
    assert result['actual_steps']==791 and result['encoder_samples']==157
    assert result['all_selected_equal_greedy'] is True
    assert result['P1_admission'] is False


@pytest.mark.parametrize('fault',['drop_branch','drop_encoding','runtime_error','encoding_error','unexplained_endpoint'])
def test_incomplete_or_unexplained_receipts_rejected(fault):
    plan,physical=actual()
    if fault=='drop_branch':physical['rows'].pop()
    elif fault=='drop_encoding':physical['encodings'].pop()
    elif fault=='runtime_error':physical['failures'].append({'error':'render'})
    else:
        receipt=next(e for e in physical['encodings'] if e['status']=='blocked')
        if fault=='encoding_error':receipt['status']='encoding_failed'
        else:
            next(r for r in physical['rows'] if (r['root_sha256'],r['method'])==(
                receipt['root_sha256'],receipt['method']))['status']='completed'
    with pytest.raises(ValueError):summarize(plan,physical)
