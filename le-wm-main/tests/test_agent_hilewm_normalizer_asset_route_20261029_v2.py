"""Offline metadata fixtures only, never network/stats/torch/pickle."""
from pathlib import Path
from types import SimpleNamespace
import json
import pytest
from experiments.spectral_reachability import agent_hilewm_normalizer_asset_route_20261029_v2 as m


@pytest.mark.parametrize('url',[
    'http://hf-mirror.com/api/datasets/quentinll/lewm-pusht',
    'https://evil.example/api/datasets/quentinll/lewm-pusht',
    'https://hf-mirror.com/datasets/quentinll/lewm-pusht/resolve/main/a.h5',
    'https://hf-mirror.com/api/datasets/other/repo',
    'https://user@hf-mirror.com/api/datasets/quentinll/lewm-pusht',
    'https://hf-mirror.com/api/datasets/quentinll/lewm-pusht/tree/main',
    'https://hf-mirror.com/api/datasets/quentinll/lewm-pusht#fragment'])
def test_nonmetadata_and_unpinned_routes_rejected(url):
    with pytest.raises(ValueError):m.allowed(url,'hf-mirror.com')


@pytest.mark.parametrize('body',[b'{"x":1,"x":2}',b'{"x":NaN}',b'['+b' '*m.BODY_CAP+b']'])
def test_bad_json(body):
    with pytest.raises(ValueError):m.parse(body)


@pytest.mark.parametrize('name',['../scaler.json','/a','a//b','a\\b','a\x00b'])
def test_unsafe_names(name):
    with pytest.raises(ValueError):m.member_name(name)


def test_tree_preserves_claims_not_certification():
    a=m.tree_summary([dict(type='file',path='dataset_info.json',size=123,oid='gitblob',lfs=None),
        dict(type='file',path='process/scaler.npy',size=24,oid=None,lfs=dict(oid='mirrorclaims')),
        dict(type='file',path='pusht_expert_train.h5',size=4_000_000_000),dict(type='directory',path='process')])
    assert [x['metadata_candidate'] for x in a]==[True,True,False]
    assert all(x['authority']=='unverified_provider_claim' for x in a)
    assert a[1]['lfs']==dict(oid='mirrorclaims')
    with pytest.raises(ValueError):m.tree_summary([dict(type='file',path='a.json')])
    with pytest.raises(ValueError):m.tree_summary([dict(type='file',path='a',size=False)])
    with pytest.raises(ValueError):m.tree_summary([dict(type='file',path='a',size=1)]*2)


def test_pagination_pins_host_and_revision():
    url='https://hf-mirror.com/api/datasets/quentinll/lewm-pusht/tree/'+'a'*40+'?limit=1000'
    assert m.next_page('<'+url+'&cursor=2>; rel="next"',url,'hf-mirror.com')==url+'&cursor=2'
    for nxt in (url.replace('hf-mirror.com','evil.example'),url.replace('a'*40,'b'*40)):
        with pytest.raises(ValueError):m.next_page('<'+nxt+'>; rel="next"',url,'hf-mirror.com')


def setup_output(tmp_path,monkeypatch):
    root=tmp_path/'project';(root/'results').mkdir(parents=True)
    monkeypatch.setattr(m,'ROOT',root)
    monkeypatch.setattr(m.shutil,'disk_usage',lambda p:SimpleNamespace(free=100*m.OUTPUT_CAP+m.FLOOR))
    return root/'results'/(m.TAG+'_fixture')


def test_authorization_before_io(monkeypatch):
    monkeypatch.setattr(m,'fetch',lambda *a:pytest.fail('network'))
    with pytest.raises(ValueError,match='authorization'):m.collect('missing','hf-mirror.com')


def test_actual403_preserved_not_absence(tmp_path,monkeypatch):
    output=setup_output(tmp_path,monkeypatch);calls=[]
    def fetch(url,host):
        calls.append(url);body=b'Access denied'
        return dict(url=url,status=403,error=None,headers={},body_bytes=len(body),body_sha256=m.sha(body)),body
    monkeypatch.setattr(m,'fetch',fetch)
    report,result=m.collect(output,'hf-mirror.com',network_authorized=True)
    assert len(calls)==2 and all('/tree/' not in u for u in calls)
    assert report['status']=='metadata_blocked_unknown' and report['raw_inverse_authority'] is None
    assert all(x['candidate_revision'] is None and not x['tree_complete'] and x['error'] for x in report['repositories'])
    assert (output/'response_00.bin').read_bytes()==b'Access denied'
    assert m.sha(Path(result['report']).read_bytes())==result['sha256']
    with pytest.raises(FileExistsError):m.collect(output,'hf-mirror.com',network_authorized=True)


def test_claimed_revision_tree_and_sidecar_remain_unknown_authority(tmp_path,monkeypatch):
    output=setup_output(tmp_path,monkeypatch);calls=[]
    def fetch(url,host):
        calls.append(url)
        if '/tree/' not in url:data=dict(id=m.REPO,sha='a'*40,siblings=[dict(rfilename='dataset_info.json')],cardData=dict(dataset_info=dict(features=[])))
        else:data=[dict(type='file',path='dataset_info.json',size=100,oid='b'*40)]
        body=m.encode(data);return dict(url=url,status=200,error=None,headers={},body_bytes=len(body),body_sha256=m.sha(body)),body
    monkeypatch.setattr(m,'fetch',fetch)
    report,_=m.collect(output,'hf-mirror.com',network_authorized=True)
    assert len(calls)==4 and all('/resolve/' not in u for u in calls)
    assert report['status']=='metadata_recorded_not_scaler_certified'
    assert report['raw_inverse_authority'] is None and not report['official_equivalence'] and not report['physical_release']
    assert not report['author_frozen_scaler_obtained'] and not report['stats_fitted']
    assert all(x['candidate_revision']=='a'*40 and len(x['metadata_candidates'])==1 for x in report['repositories'])


def test_empty_tree_does_not_assert_exhaustive(tmp_path,monkeypatch):
    output=setup_output(tmp_path,monkeypatch)
    def fetch(url,host):
        body=m.encode([] if '/tree/' in url else dict(id=m.REPO,sha='b'*40))
        return dict(status=200,error=None,headers={},url=url,body_sha256=m.sha(body)),body
    monkeypatch.setattr(m,'fetch',fetch)
    report,_=m.collect(output,'hf-mirror.com',network_authorized=True)
    assert all(x['tree_complete'] for x in report['repositories']) and not report['exhaustive']
    assert report['raw_inverse_authority'] is None


def test_raw_response_integrity_and_own_json(tmp_path,monkeypatch):
    output=setup_output(tmp_path,monkeypatch)
    def fetch(url,host):return dict(url=url,status=None,error='URLError network',headers={}),b''
    monkeypatch.setattr(m,'fetch',fetch)
    report,result=m.collect(output,'hf-mirror.com',network_authorized=True)
    saved=json.loads(Path(result['report']).read_bytes());assert saved==report
    assert len(list(output.iterdir()))==3
    assert saved['neural_calls']==saved['physics_calls']==0
