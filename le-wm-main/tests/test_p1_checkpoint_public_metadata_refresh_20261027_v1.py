"""RAM-only network/filesystem mocks. No live public requests or model imports."""
import hashlib
import io
import json
from pathlib import Path
import socket
import stat
from types import SimpleNamespace
import pytest
from experiments.spectral_reachability import p1_checkpoint_public_metadata_refresh_20261027_v1 as p

COMMIT='a'*40
WEIGHTS='b'*64
CONFIG=b'{"architecture":"fixture"}\n'


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(socket,'create_connection',lambda *a,**k:pytest.fail('live network forbidden'))


class MemoryStore:
    def __init__(self):self.files={};self.created=False
    def begin(self):
        if self.created:raise FileExistsError('NEW directory required')
        self.created=True
    def write(self,name,raw):
        if name in self.files:raise FileExistsError(name)
        self.files[name]=raw


def metadata(**updates):
    data=dict(id=p.REPO,sha=COMMIT,siblings=[
        dict(rfilename='weights.pt',lfs=dict(sha256=WEIGHTS,size=10000000)),
        dict(rfilename='config.json',blobId=p.git_blob(CONFIG),size=len(CONFIG)),
        dict(rfilename='action_stats.json',size=30),dict(rfilename='dataset.h5',size=1),
        dict(rfilename='../evil.json',size=1)])
    data.update(updates);return json.dumps(data).encode()


def local(path,kind):
    return dict(path=str(path),status='hashed_readonly',sha256=WEIGHTS if kind=='weights' else hashlib.sha256(CONFIG).hexdigest(),
                git_blob_sha1=None if kind=='weights' else p.git_blob(CONFIG))


def run_fixture(transport=None,**kwargs):
    store=MemoryStore();calls=[]
    def fetch(url,limit,timeout):
        calls.append((url,limit,timeout))
        if transport:return transport(url,limit,timeout)
        return p.FetchResult(200,metadata() if '/api/models/' in url else CONFIG,None,True)
    hasher=kwargs.pop('local_hasher_override',local)
    result=p.refresh('/new',main_approved=True,transport=fetch,local_hasher=hasher,
                     store_factory=lambda path:store,**kwargs)
    return result,store,calls


def test_default_no_approval_does_not_create_read_or_connect():
    def forbidden(*a,**k):pytest.fail('no action before approval')
    with pytest.raises(ValueError):p.refresh('/new',transport=forbidden,local_hasher=forbidden,store_factory=forbidden)


def test_exact_endpoints_and_separate_provenance():
    result,store,calls=run_fixture()
    assert [x[0] for x in calls]==list(p.API_URLS.values())
    assert all(x[1]==2*1024**2 and x[2]<=15 for x in calls)
    assert result['origins']['official']['source_kind']=='official_domain'
    assert result['origins']['mirror']['source_kind']=='mirror_domain_not_official_attestation'
    assert result['cross_origin_independence']=='unknown'
    assert result['training_stats_provenance']=='unknown'
    assert result['P1_admission'] is False
    assert len(result['requests'])==2
    for record in result['requests']:
        assert hashlib.sha256(store.files[record['rawbody_path']]).hexdigest()==record['body_sha256']


def test_pinned_allowlisted_files_only_and_readonly_matches():
    result,store,calls=run_fixture(small_files=('config.json','action_stats.json'))
    assert len(calls)==6
    assert all('/raw/'+COMMIT+'/' in url for url,_,_ in calls if '/api/models/' not in url)
    assert all(limit<=256*1024 for url,limit,_ in calls if '/raw/' in url)
    for origin in result['origins'].values():
        assert origin['weights_lfs_match_local'] is True
        assert origin['config_git_blob_match_local'] is True
        assert origin['config_body_sha256_match_local'] is True
        assert origin['training_stats_provenance']=='unknown'
    assert all('weights.pt' not in url and 'dataset' not in url for url,_,_ in calls)


@pytest.mark.parametrize('error,status,body', [('timeout',None,b'partial'),('transport_error',None,b''),
    ('http_status_not_200',403,b'denied'),('body_cap_reached',200,b'prefix'),('redirect_not_followed',302,b'redirect')])
def test_failures_keep_raw_empty_or_partial_and_hash(error,status,body):
    result,store,calls=run_fixture(transport=lambda *a:p.FetchResult(status,body,error,False),small_files=('config.json',))
    assert len(calls)==2
    for r in result['requests']:
        assert r['error']==error and r['http_status']==status
        assert store.files[r['rawbody_path']]==body
        assert r['body_sha256']==hashlib.sha256(body).hexdigest()
    assert all(x['commit'] is None for x in result['origins'].values())


@pytest.mark.parametrize('data', [b'bad json',metadata(sha='main'),metadata(id='other/repo'),
    metadata(siblings=[dict(rfilename='weights.pt'),dict(rfilename='weights.pt')])])
def test_bad_api_metadata_never_followed(data):
    result,store,calls=run_fixture(transport=lambda *a:p.FetchResult(200,data,None,True),small_files=('config.json',))
    assert len(calls)==2
    assert all(o['metadata_error'] is not None for o in result['origins'].values())
    assert all(o['weights_lfs_match_local'] is None for o in result['origins'].values())


def test_conflicting_origin_commits_not_merged():
    def fetch(url,*args):return p.FetchResult(200,metadata(sha=('a' if 'huggingface.co' in url else 'c')*40),None,True)
    result,_,_=run_fixture(transport=fetch)
    assert result['origins']['official']['commit']!=result['origins']['mirror']['commit']
    assert result['cross_origin_independence']=='unknown'


def test_missing_lfs_and_config_stays_null_not_false_or_zero():
    result,_,_=run_fixture(transport=lambda *a:p.FetchResult(200,metadata(siblings=[]),None,True))
    for o in result['origins'].values():
        assert o['weights_lfs_match_local'] is None and o['config_git_blob_match_local'] is None


def test_local_failure_does_not_hide_public_failure_records():
    def broken(*args):raise OSError('secret must not leak')
    result,store,calls=run_fixture(local_hasher_override=broken)
    assert result['local']['weights']['sha256'] is None
    assert len(calls)==2
    assert b'secret must not leak' not in store.files['report.json']


@pytest.mark.parametrize('url',[
    'https://huggingface.co/api/datasets/quentinll/lewm-pusht?blobs=true',
    'https://huggingface.co/api/models/quentinll/lewm-pusht?blobs=true&token=secret',
    'https://u:p@huggingface.co/api/models/quentinll/lewm-pusht?blobs=true',
    'https://huggingface.co/quentinll/lewm-pusht/raw/main/config.json',
    'https://huggingface.co/quentinll/lewm-pusht/raw/'+COMMIT+'/weights.pt',
    'https://hf-mirror.com/quentinll/lewm-pusht/raw/'+COMMIT+'/../config.json',
    'https://evil.invalid/config.json',
])
def test_url_policy_rejects_arbitrary_redirect_auth_weights_dataset(url):
    assert not p.allowed_url(url)


@pytest.mark.parametrize('timeout',[0,-1,15.01,float('inf'),float('nan')])
def test_invalid_timeout_preflight(timeout):
    with pytest.raises(ValueError):run_fixture(timeout=timeout)


def test_oversize_mock_body_is_capped_logged_not_parsed():
    result,store,_=run_fixture(transport=lambda *a:p.FetchResult(200,b'x'*(p.METADATA_CAP+1),None,True))
    for record in result['requests']:
        assert record['body_bytes']==p.METADATA_CAP and record['complete'] is False
        assert record['error']=='body_cap_reached'


def test_file_not_advertised_or_too_large_not_requested():
    data=metadata(siblings=[dict(rfilename='config.json',size=256*1024+1)])
    result,_,calls=run_fixture(transport=lambda *a:p.FetchResult(200,data,None,True),small_files=('config.json','README.md'))
    assert len(calls)==2
    for o in result['origins'].values():
        assert [x['reason'] for x in o['skipped_files']]==['advertised_size_exceeds_cap','not_advertised']


def test_cli_requires_explicit_main_gate_and_no_token_or_custom_url_option(capsys):
    with pytest.raises(SystemExit):p.main(['--out-dir','/new'])
    with pytest.raises(SystemExit):p.main(['--help'])
    text=capsys.readouterr().out
    assert '--main-approved-network' in text and '--token' not in text and '--url' not in text


def test_diskstore_directory_exclusive(monkeypatch):
    def exists(*a,**k):raise FileExistsError('already exists or dangling symlink')
    monkeypatch.setattr(Path,'mkdir',exists)
    with pytest.raises(FileExistsError):p.DiskStore('/existing').begin()


def test_existing_run_rejected_before_local_or_network():
    store=MemoryStore();store.created=True
    with pytest.raises(FileExistsError):p.refresh('/old',main_approved=True,store_factory=lambda path:store,
        transport=lambda *a:pytest.fail('network'),local_hasher=lambda *a:pytest.fail('read'))


@pytest.mark.parametrize('status,content,length,limit,expected',[
    (200,b'ok',2,10,None), (403,b'denied',6,10,'http_status_not_200'),
    (302,b'redirect',8,10,'redirect_not_followed'),
    (200,b'large',5,4,'body_cap_reached'), (200,b'xx',5,10,'truncated_body'),
    (200,b'four',None,4,'body_cap_reached')])
def test_worker_anonymous_no_redirect_bounded_raw(monkeypatch,status,content,length,limit,expected):
    requests=[];frames=[]
    class Response:
        def __init__(self):self.status=status;self.raw=io.BytesIO(content)
        def getheader(self,key,default=None):
            return str(length) if key=='Content-Length' and length is not None else default
        def read1(self,size):return self.raw.read(size)
    class Connection:
        def __init__(self,host,timeout):assert host=='huggingface.co' and timeout<=15
        def request(self,method,target,headers):requests.append((method,target,headers))
        def getresponse(self):return Response()
        def close(self):pass
    monkeypatch.setenv('HF_TOKEN','RAM_SENTINEL_NEVER_SEND')
    monkeypatch.setattr(p.http.client,'HTTPSConnection',Connection)
    pipe=SimpleNamespace(send=frames.append,close=lambda:None)
    p._network_worker(p.API_URLS['official'],limit,15,pipe)
    assert len(requests)==1 and requests[0][0]=='GET'
    assert set(requests[0][2])=={'User-Agent','Accept','Accept-Encoding'}
    assert 'RAM_SENTINEL_NEVER_SEND' not in repr(requests)+repr(frames)
    assert b''.join(v for k,v in frames if k=='body')==content[:limit]
    assert frames[-1][1][0]==expected


def test_worker_timeout_retains_partial_prefix(monkeypatch):
    frames=[]
    class Response:
        status=200
        def __init__(self):self.first=True
        def getheader(self,key,default=None):return default
        def read1(self,size):
            if self.first:self.first=False;return b'partial'
            raise TimeoutError('DO_NOT_PRINT')
    class Connection:
        def __init__(self,*a,**k):pass
        def request(self,*a,**k):pass
        def getresponse(self):return Response()
        def close(self):pass
    monkeypatch.setattr(p.http.client,'HTTPSConnection',Connection)
    p._network_worker(p.API_URLS['official'],100,15,SimpleNamespace(send=frames.append,close=lambda:None))
    assert ('body',b'partial') in frames and frames[-1]==('done',('timeout',False))
    assert 'DO_NOT_PRINT' not in repr(frames)


def fake_process_transport(monkeypatch,frames):
    pending=list(frames);terminated=[];polls=[]
    class Receiver:
        def poll(self,timeout):polls.append(timeout);return bool(pending)
        def recv(self):return pending.pop(0)
        def close(self):pass
    sender=SimpleNamespace(close=lambda:None)
    class Process:
        def __init__(self,**kwargs):self.alive=True
        def start(self):pass
        def is_alive(self):return self.alive
        def terminate(self):terminated.append(True);self.alive=False
        def kill(self):self.alive=False
        def join(self,timeout):pass
    context=SimpleNamespace(Pipe=lambda **k:(Receiver(),sender),Process=Process)
    monkeypatch.setattr(p.multiprocessing,'get_context',lambda method:context)
    monkeypatch.setattr(p.time,'monotonic',lambda:0)
    return terminated,polls


def test_parent_deadline_retains_raw_and_stops_worker(monkeypatch):
    terminated,polls=fake_process_transport(monkeypatch,[('status',200),('body',b'partial')])
    result=p.fetch_public(p.API_URLS['official'],100,15)
    assert result==p.FetchResult(200,b'partial','timeout',False)
    assert terminated and all(0<t<=15 for t in polls)


def test_parent_caps_bad_worker_frame(monkeypatch):
    fake_process_transport(monkeypatch,[('status',200),('body',b'x'*101)])
    result=p.fetch_public(p.API_URLS['official'],100,15)
    assert len(result.body)==100 and result.error=='body_cap_reached'


def test_transport_policy_rejects_before_spawning(monkeypatch):
    monkeypatch.setattr(p.multiprocessing,'get_context',lambda *a:pytest.fail('spawn disallowed'))
    with pytest.raises(ValueError):p.fetch_public('https://evil.invalid',1,15)
    with pytest.raises(ValueError):p.fetch_public(p.API_URLS['official'],p.METADATA_CAP+1,15)
    url=f'https://huggingface.co/{p.REPO}/raw/{COMMIT}/config.json'
    with pytest.raises(ValueError):p.fetch_public(url,p.SMALL_CAP+1,15)


def test_local_hash_stream_is_readonly_sha256_and_git_blob(monkeypatch):
    opened=[]
    info=SimpleNamespace(st_mode=stat.S_IFREG,st_dev=1,st_ino=2,st_size=len(CONFIG),st_mtime_ns=3,st_ctime_ns=4)
    class Stream(io.BytesIO):
        def fileno(self):return 99
    def read(path,mode):opened.append(mode);return Stream(CONFIG)
    monkeypatch.setattr(Path,'is_symlink',lambda path:False)
    monkeypatch.setattr(Path,'open',read)
    monkeypatch.setattr(Path,'stat',lambda path:info)
    monkeypatch.setattr(p.os,'fstat',lambda fd:info)
    result=p.hash_local('/fixture/config.json','config')
    assert opened==['rb'] and result['sha256']==hashlib.sha256(CONFIG).hexdigest()
    assert result['git_blob_sha1']==p.git_blob(CONFIG)


def test_transport_exception_sanitized_and_raw_record_still_saved():
    def fail(*args):raise RuntimeError('TOKEN_SENTINEL_NEVER_SERIALIZE')
    result,store,_=run_fixture(transport=fail)
    assert all(r['error']=='transport_error' and r['http_status'] is None for r in result['requests'])
    assert b'TOKEN_SENTINEL_NEVER_SERIALIZE' not in store.files['report.json']
    assert all(store.files[r['rawbody_path']]==b'' for r in result['requests'])


def test_cli_approved_argument_plumbing_without_execution(monkeypatch,capsys):
    captured=[]
    def stub(output,**kwargs):
        captured.append((output,kwargs));return {'requests':[]}
    monkeypatch.setattr(p,'refresh',stub)
    p.main(['--out-dir','/new','--main-approved-network','--sources','official',
            '--small-files','README.md','config.json','--timeout','12',
            '--local-weights','/local/weights.pt','--local-config','/local/config.json'])
    output,kwargs=captured[0]
    assert output=='/new' and kwargs['main_approved'] is True
    assert kwargs['sources']==('official',) and kwargs['timeout']==12
    assert kwargs['small_files']==('README.md','config.json')
    assert kwargs['local_weights']==Path('/local/weights.pt')
    assert json.loads(capsys.readouterr().out)['training_stats_provenance']=='unknown'


@pytest.mark.parametrize('kwargs',[{'sources':('official','official')},{'sources':('evil',)},
    {'small_files':('weights.pt',)},{'small_files':('../stats.json',)}])
def test_bad_scope_rejected_before_creating_output(kwargs):
    with pytest.raises(ValueError):p.refresh('/new',main_approved=True,
        store_factory=lambda path:pytest.fail('preflight must precede mkdir'),**kwargs)


def test_blob_conflict_is_recorded_not_training_provenance():
    data=metadata(siblings=[dict(rfilename='weights.pt',lfs=dict(sha256='c'*64)),
                            dict(rfilename='config.json',blobId='d'*40,size=len(CONFIG))])
    def fetch(url,*args):return p.FetchResult(200,data if '/api/models/' in url else CONFIG,None,True)
    result,_,_=run_fixture(transport=fetch,small_files=('config.json',))
    for origin in result['origins'].values():
        assert origin['weights_lfs_match_local'] is False
        assert origin['config_git_blob_match_local'] is False
        assert origin['config_body_sha256_match_local'] is True
        assert origin['files'][0]['body_git_blob_match_metadata'] is False
        assert origin['training_stats_provenance']=='unknown'
