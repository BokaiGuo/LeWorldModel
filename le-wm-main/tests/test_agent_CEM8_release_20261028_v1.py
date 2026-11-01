"""RAM-only durable release protocol; no real results, model, GPU or physics."""
from copy import deepcopy
from pathlib import Path
import sys
import pytest
from experiments.spectral_reachability import agent_CEM8_release_20261028_v1 as r


class RAM:
    def __init__(self):self.files={};self.directories={'/RAM','/RAM/input'};self.order=[];self.free=r.FLOOR+r.CAP
    def read(self,path,cap):
        data=self.files[str(path)];r.require(len(data)<=cap,'bounded RAM read');return data
    def sha(self,path):return r.digest(self.files[str(path)])
    def names(self,path):return {Path(p).name for p in self.files if str(Path(p).parent)==str(path)}
    def claim(self,path):
        r.require(str(path) not in self.directories and str(path) not in self.files,'exclusive NEW directory')
        r.require(self.free>=r.FLOOR+r.CAP,'disk floor');self.directories.add(str(path));self.order.append('claim')
    def write(self,path,raw):
        r.require(str(path) not in self.files,'exclusive file')
        used=sum(len(v) for p,v in self.files.items() if Path(p).parent==Path(path).parent)
        r.require(used+len(raw)<=r.CAP and self.free>=r.FLOOR+len(raw),'cap/floor')
        self.files[str(path)]=raw;self.order.append(Path(path).name);return self.sha(path)


def fixture(monkeypatch):
    fs=RAM();producer=b'RAM frozen producer';parentfile=Path('/RAM/parent.json')
    fs.files[str(r.PRODUCER_FILE)]=producer
    monkeypatch.setattr(r,'PRODUCER_SHA',r.digest(producer));monkeypatch.setattr(r,'PARENT_PREP',parentfile)
    fs.files['/RAM/input_model']=b'RAM saved checkpoint hash only'
    parent=dict(schema=r.PARENT_NAME+'.prepare',bindings={'/RAM/input_model':fs.sha('/RAM/input_model')})
    fs.files[str(parentfile)]=r.canonical(parent);monkeypatch.setattr(r,'PARENT_SHA',fs.sha(parentfile))
    prep=dict(schema=r.PRODUCER_NAME+'.prepare',bindings={str(parentfile):r.PARENT_SHA,str(r.PRODUCER_FILE):r.PRODUCER_SHA})
    fs.files['/RAM/prepared.json']=r.canonical(prep)
    fs.files['/RAM/input/binding.json']=r.canonical(dict(prepared='/RAM/prepared.json',prepare_sha256=fs.sha('/RAM/prepared.json')))
    report=dict(schema=r.PRODUCER_NAME+'.report',prepared='/RAM/prepared.json',prepare_sha256=fs.sha('/RAM/prepared.json'),
        artifact_sha256={'binding.json':fs.sha('/RAM/input/binding.json')},status='teacher8_CEM_commands_sealed_not_admitted')
    fs.files['/RAM/input/report.json']=r.canonical(report)
    for f in r.OWN_FILES:fs.files[str(f)]=('RAM own '+str(f)).encode()
    result=dict(schema=r.PRODUCER_NAME+'.consume',status='teacher8_CEM_commands_verified_not_admitted',errors=[],
        plans=[dict(root_slot=s,source_slot=s,condition=c,prefix_arm=a,anchor=i,
                    whole_run_report_sha256=fs.sha('/RAM/input/report.json'),model_Q=float(i)) for i,(s,c,a) in enumerate(r.ORDER)],
        P1_admission=False,P3_admission=False,equal_total_compute=False,new_neural_calls=0,new_physics_calls=0,
        arbitrary_original_checks={'all_preserved':[False,None,{'native':'no truncation'}]})
    calls=[]
    def consumer(directory,h):
        calls.append((directory,h));fs.order.append('consume');return deepcopy(result)
    args=dict(directory='/RAM/input',report_sha=fs.sha('/RAM/input/report.json'),
              prepared='/RAM/prepared.json',prepare_sha=fs.sha('/RAM/prepared.json'),output='/RAM/release',io=fs,consumer=consumer)
    return fs,result,calls,args


def test_exact_result_complete_input_inventory_and_durable_order(monkeypatch):
    fs,result,calls,args=fixture(monkeypatch);summary=r.run(**args)
    doc=r.decode(fs.files['/RAM/release/receipt.json'])
    assert doc['producer_result']==result
    assert doc['input_sha256']['/RAM/input_model']==fs.sha('/RAM/input_model')
    assert doc['input_sha256']['/RAM/input/report.json']==args['report_sha']
    assert doc['input_sha256']['/RAM/prepared.json']==args['prepare_sha']
    assert doc['input_sha256'][str(r.PRODUCER_FILE)]==r.PRODUCER_SHA
    assert fs.order.index('claim')<fs.order.index('request.json')<fs.order.index('consume')<fs.order.index('receipt.json')
    assert summary['sha256']==fs.sha('/RAM/release/receipt.json') and summary['plans']==8
    assert calls==[('/RAM/input',args['report_sha'])] and doc['P1_admission'] is False


def test_release_verify_reinvokes_saved_consumer_no_new_files(monkeypatch):
    fs,_,calls,args=fixture(monkeypatch);summary=r.run(**args);before=deepcopy(fs.files)
    out=r.verify('/RAM/release',summary['sha256'],io=fs,consumer=args['consumer'])
    assert out['status']=='release_reverified_not_admitted' and len(calls)==2 and fs.files==before


def test_reentry_rejected_before_second_consume(monkeypatch):
    fs,_,calls,args=fixture(monkeypatch);r.run(**args)
    with pytest.raises(ValueError):r.run(**args)
    assert len(calls)==1


def test_recursive_entry_loses_exclusive_claim_before_consume(monkeypatch):
    fs,result,_,args=fixture(monkeypatch);entered=[]
    def consumer(*a):
        entered.append(True)
        with pytest.raises(ValueError):r.run(**args)
        return result
    args['consumer']=consumer;r.run(**args);assert entered==[True]


@pytest.mark.parametrize('target',['model','artifact','source','prepare','report','parent','own_doc','extra_artifact'])
def test_input_tamper_failclosed_before_release(monkeypatch,target):
    fs,_,calls,args=fixture(monkeypatch)
    names=dict(model='/RAM/input_model',artifact='/RAM/input/binding.json',source=str(r.PRODUCER_FILE),
        prepare='/RAM/prepared.json',report='/RAM/input/report.json',parent=str(r.PARENT_PREP),own_doc=str(r.OWN_FILES[2]))
    if target=='extra_artifact':fs.files['/RAM/input/unlisted']=b'extra'
    else:fs.files[names[target]]+=b'changed'
    # Own wrapper source files are newly bound, not a preexisting frozen input.
    if target=='own_doc':
        summary=r.run(**args);assert summary['sha256']==fs.sha('/RAM/release/receipt.json');return
    with pytest.raises(ValueError):r.run(**args)
    assert '/RAM/release/receipt.json' not in fs.files and calls==[]


@pytest.mark.parametrize('target',['model','source','artifact','report','own_doc'])
def test_mutation_during_true_consume_detected_and_reservation_kept(monkeypatch,target):
    fs,result,_,args=fixture(monkeypatch)
    name=dict(model='/RAM/input_model',source=str(r.PRODUCER_FILE),artifact='/RAM/input/binding.json',
              report='/RAM/input/report.json',own_doc=str(r.OWN_FILES[2]))[target]
    def consumer(*a):fs.files[name]+=b'changed';return result
    args['consumer']=consumer
    with pytest.raises(ValueError):r.run(**args)
    assert '/RAM/release/receipt.json' not in fs.files and '/RAM/release/error.json' in fs.files
    with pytest.raises(ValueError):r.run(**args)


@pytest.mark.parametrize('target',['status','errors','denominator','condition','report_sha','P1','neural','oversize'])
def test_invalid_result_never_persisted_as_release(monkeypatch,target):
    fs,result,_,args=fixture(monkeypatch)
    if target=='status':result['status']='prepared_only'
    elif target=='errors':result['errors']=['failure']
    elif target=='denominator':result['plans'].pop()
    elif target=='condition':result['plans'][0]['condition']='on'
    elif target=='report_sha':result['plans'][0]['whole_run_report_sha256']='f'*64
    elif target=='P1':result['P1_admission']=True
    elif target=='neural':result['new_neural_calls']=520
    else:result['huge']='x'*r.CAP
    args['consumer']=lambda *a:deepcopy(result)
    with pytest.raises(ValueError):r.run(**args)
    assert '/RAM/release/receipt.json' not in fs.files and '/RAM/release/error.json' in fs.files


def test_saved_consumer_exception_not_hidden_or_retried(monkeypatch):
    fs,_,_,args=fixture(monkeypatch)
    def consumer(*a):raise RuntimeError('native module mismatch')
    args['consumer']=consumer
    with pytest.raises(RuntimeError,match='native module mismatch'):r.run(**args)
    error=r.decode(fs.files['/RAM/release/error.json'])
    assert error['status']=='failed_no_release' and 'native module mismatch' in error['message']


@pytest.mark.parametrize('target',['receipt','request','extra','model'])
def test_verify_tamper_rejected(monkeypatch,target):
    fs,_,_,args=fixture(monkeypatch);summary=r.run(**args)
    if target=='extra':fs.files['/RAM/release/extra']=b'wrong'
    else:
        name={'receipt':'/RAM/release/receipt.json','request':'/RAM/release/request.json','model':'/RAM/input_model'}[target]
        fs.files[name]+=b'changed'
    with pytest.raises(ValueError):r.verify('/RAM/release',summary['sha256'],io=fs,consumer=args['consumer'])


def test_invalid_external_sha_floor_and_output_within_producer(monkeypatch):
    fs,_,calls,args=fixture(monkeypatch);args['report_sha']='future'
    with pytest.raises(ValueError):r.run(**args)
    assert calls==[] and fs.order==[]
    fs,_,calls,args=fixture(monkeypatch);args['output']='/RAM/input/release'
    with pytest.raises(ValueError):r.run(**args)
    assert fs.order==[]
    fs,_,calls,args=fixture(monkeypatch);fs.free=r.FLOOR-1
    with pytest.raises(ValueError):r.run(**args)
    assert calls==[]


def test_conflicting_direct_parent_hash_rejected(monkeypatch):
    fs,_,calls,args=fixture(monkeypatch)
    p=r.decode(fs.files['/RAM/prepared.json']);p['bindings']['/RAM/input_model']='f'*64
    fs.files['/RAM/prepared.json']=r.canonical(p);args['prepare_sha']=fs.sha('/RAM/prepared.json')
    report=r.decode(fs.files['/RAM/input/report.json']);report['prepare_sha256']=args['prepare_sha']
    fs.files['/RAM/input/report.json']=r.canonical(report);args['report_sha']=fs.sha('/RAM/input/report.json')
    with pytest.raises(ValueError):r.run(**args)
    assert calls==[]


def test_help_no_torch_and_default_provider_is_actual_producer(capsys):
    assert 'torch' not in sys.modules
    with pytest.raises(SystemExit):r.main(['--help'])
    assert 'verify' in capsys.readouterr().out
    producer=r.load_producer()
    assert producer.consume.__module__.endswith(r.PRODUCER_NAME)
    assert 'torch' not in sys.modules


def test_actual_producer_consume_invalid_abi_no_neural_or_files():
    # Invoke the real function, stopping at its first external-SHA gate.
    with pytest.raises(ValueError):r.checked_consume(dict(directory='/RAM/never_open',report_sha='not-a-SHA'),None)
    assert 'torch' not in sys.modules


def test_resealed_envelope_tamper_not_confused_with_real_consumer(monkeypatch):
    fs,_,_,args=fixture(monkeypatch);r.run(**args)
    path='/RAM/release/receipt.json';doc=r.decode(fs.files[path]);doc['new_neural_calls']=520
    fs.files[path]=r.canonical(doc)
    with pytest.raises(ValueError):r.verify('/RAM/release',fs.sha(path),io=fs,consumer=args['consumer'])


def test_local_io_atomic_claim_fsync_and_xb_in_RAM(monkeypatch):
    from io import BytesIO
    from types import SimpleNamespace
    created=[];modes=[];synced=[];raws=[]
    class RAMFile(BytesIO):
        def fileno(self):return 8
        def close(self):raws.append(self.getvalue());super().close()
    monkeypatch.setattr(Path,'exists',lambda _:False)
    monkeypatch.setattr(Path,'is_symlink',lambda _:False)
    monkeypatch.setattr(Path,'is_dir',lambda _:True)
    monkeypatch.setattr(Path,'iterdir',lambda _:iter(()))
    monkeypatch.setattr(Path,'mkdir',lambda p,**kw:created.append((str(p),kw)))
    monkeypatch.setattr(Path,'open',lambda p,mode:modes.append(mode) or RAMFile())
    monkeypatch.setattr(r.shutil,'disk_usage',lambda _:SimpleNamespace(free=r.FLOOR+r.CAP))
    monkeypatch.setattr(r,'syncdir',lambda p:synced.append(str(p)))
    monkeypatch.setattr(r.os,'fsync',lambda fd:synced.append(fd))
    io=r.LocalIO();io.claim('/RAM/new_release');h=io.write('/RAM/new_release/request.json',b'{}')
    assert created==[('/RAM/new_release',{'mode':0o700})] and modes==['xb']
    assert h==r.digest(b'{}') and raws==[b'{}'] and synced==['/RAM',8,'/RAM/new_release']
