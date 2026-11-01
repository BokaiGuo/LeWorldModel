"""Pure saved arrays and tamper tests: no model/environment creation."""
from copy import deepcopy
import json
import pytest
from experiments.spectral_reachability import p2_composition_contract_audit_20261027_v1 as a


@pytest.fixture(scope='module')
def saved():
    run=a.PROJECT/a.resources.PLANNER
    read=lambda p:json.loads(p.read_text())
    report=read(run/'report.json'); p=read(a.Path(report['prepared'])/'protocol.json')
    row=read(run/'root_0.json'); root=row['root_sha256']
    c=next(x['compiled'] for x in read(a.Path(p['context']['compilation']))['rows'] if x['root_sha256']==root)
    case=next(x for x in p['context']['cases'] if x['root_sha256']==root)
    return row,c,case['root']['context'],p['context']['normalization'],read(run/'root_0_bank.json')['expanded']


@pytest.mark.parametrize('arm',[0,1])
def test_actual_saved_candidates(saved,arm):
    row,c,initial,norm,bank=saved
    assert a.candidate_contract(row['results'][arm],c,initial,norm,bank)['candidates']==(80,40)[arm]


@pytest.mark.parametrize('damage',['raw','splice','history','cost','winner','denominator','kkt'])
def test_reject_saved_tamper(saved,damage):
    row,c,initial,norm,bank=saved
    result=deepcopy(row['results'][1]); candidate=result['attempts'][0]
    if damage=='raw': candidate['actions'][0][0]+=.01
    if damage=='splice': candidate['witness']['edges'][1]['actions'][0][0]+=.01
    if damage=='history': candidate['witness']['edges'][0]['output_context']['action_history'][-1][0]+=.01
    if damage=='cost': candidate['cost']+=.01
    if damage=='winner': result['winner']['candidate_id']='wrong'
    if damage=='denominator': result['attempts'].pop()
    if damage=='kkt': candidate['suffix_compilation']['solve']['ball_multiplier']=-1
    with pytest.raises(ValueError): a.candidate_contract(result,c,initial,norm,bank)


def test_stable_join_order_duplicates_and_freeze():
    plans=[dict(root_sha256=str(i),method=arm,selection_sha256=str(i)+arm) for i in range(4) for arm in ('j','c')]
    rows=[dict(p,terminal_rgb_sha256='rgb') for p in plans]
    assert len(a.frozen_join(plans,list(reversed(plans)),list(reversed(rows)),rows)[0])==8
    with pytest.raises(ValueError): a.frozen_join(plans,plans,rows[:-1]+[rows[0]],rows)
    changed=deepcopy(plans);changed[0]['selection_sha256']='new'
    with pytest.raises(ValueError): a.frozen_join(plans,changed,rows,rows)


def test_output_exclusive_cap_and_reserve(tmp_path,monkeypatch):
    path=tmp_path/(a.TAG+'.json')
    with pytest.raises(ValueError): a.write_new(path,{'large':'x'*a.CAP})
    monkeypatch.setattr(a.shutil,'disk_usage',lambda _:type('Disk',(),{'free':a.FLOOR-1})())
    with pytest.raises(ValueError): a.write_new(path,{})
    assert not path.exists()


def test_full_vector_rejects_forward_only_match():
    j=a.resources.vector(predict_calls=168)
    c=a.resources.vector(predict_calls=168,jacobian_192x10_calls=40)
    assert a.resources.validate_claim(j,c,'predict_frontier_equal')['predict_frontier_equal']
    with pytest.raises(ValueError,match='168/168'):
        a.resources.validate_claim(j,c,'full_resource_equal')


def test_actual_complete_saved_audit():
    report=a.audit()
    assert report['status']=='saved_contract_checks_completed'
    assert len(report['roots'])==4
    assert sum(sum(c['candidates'] for c in r['contracts'].values()) for r in report['roots'])==480
    assert report['earlier_paired40']['suffix_KKT_checks']==160
    assert not report['P2_admission']
