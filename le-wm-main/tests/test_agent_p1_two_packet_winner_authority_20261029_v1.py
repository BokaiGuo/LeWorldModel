from copy import deepcopy
from functools import lru_cache
import pytest
from experiments.spectral_reachability import agent_p1_two_packet_winner_authority_20261029_v1 as m


@lru_cache(maxsize=1)
def fixture():
    # Saved metadata ABI only: simulated accepted object, never claim producer strict.
    folder=m.ROOT/'results/agent_mixedmode_prefix_20261028_v1_cuda_MAIN01'
    r=m.mixed.overlay.read(folder/'report.json',m.mixed.CAP)
    p=m.mixed.overlay.read(r['prepared'],m.mixed.PREP_CAP);report_sha=m.sha(folder/'report.json')
    a=dict(schema=m.mixed.NAME+'.consume',status='all4_mixedmode_commands_verified_not_admitted',errors=[],
           P1_admission=False,plans=[dict(row['plan'],whole_run_report_sha256=report_sha) for row in r['rows']])
    tasks=[m.F.task_from(s,p['expected_model_hash']) for s in p['slots']]
    return p,r,a,tasks,report_sha


def test_real_saved_winner_chart_and_solver_domains():
    p,r,a,tasks,h=fixture();rows=m.join(p,r,a,tasks,h)
    assert [x['winner_index'] for x in rows]==[51,51,23,23]
    assert all(x['prefix_norm']<=m.F.CAP and x['suffix_norm']<=m.F.CAP for x in rows)
    assert all(m.F.bits(x['request']['winner_commands'])==x['request']['winner_raw_bits'] for x in rows)


@pytest.mark.parametrize('which',['prefix','suffix','solver','plan','schema','index'])
def test_winner_tampering_is_rejected(which):
    p,r,a,tasks,h=fixture();p=deepcopy(p);r=deepcopy(r);a=deepcopy(a)
    w=r['rows'][0]['result']['winner'];index=w['index']
    if which=='prefix':w['prefix_params'][0]+=.01
    elif which=='suffix':w['suffix_params'][0]+=.01
    elif which=='solver':w['suffix_solve']['radius']+=1
    elif which=='plan':a['plans'][0]['commands'][0][0]+=.01
    elif which=='schema':a['status']='partial'
    else:w['index']=True
    if which in ('prefix','suffix','solver'):r['rows'][0]['result']['candidates'][index]=deepcopy(w)
    with pytest.raises((ValueError,KeyError)):m.join(p,r,a,tasks,h)
