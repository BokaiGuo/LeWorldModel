"""Offline bounded network mocks only; no live sockets, models or pickle loading."""
from contextlib import nullcontext
import hashlib
import io
import json
from pathlib import Path
import random
import signal
import struct
from unittest.mock import patch
import zlib

import pytest
from experiments.spectral_reachability import p1_hilewm_bounded_object_fetch_20261027_v2 as m


class Log:
    def __init__(self): self.events=[]
    def add(self,event,**fields): self.events.append((event,fields))


class Response:
    def __init__(self,body,start,length,*,status=206,headers=None,url=None,onread=None):
        self.body=io.BytesIO(body); self.status=status; self.url=url or m.pure.URL
        self.headers={'Content-Range':f'bytes {start}-{start+length-1}/{m.pure.ARCHIVE_BYTES}',
                      'Content-Length':str(length)}
        self.headers.update(headers or {}); self.reads=0; self.onread=onread
    def geturl(self): return self.url
    def close(self): pass
    def read(self,n):
        self.reads+=1
        if self.onread: self.onread()
        return self.body.read(n)


class One:
    def __init__(self,response): self.response=response; self.calls=0
    def open(self,req,timeout):
        assert req.full_url==m.pure.URL and timeout<=m.SOCKET_SECONDS
        self.calls+=1; return self.response


def client(tmp_path,response,mode='fetch',allowed=None,**kwargs):
    return m.Client(tmp_path,Log(),allowed or {(1,2)},mode,opener=One(response),
                    deadline=lambda x:nullcontext(),**kwargs)


def test_real_plan_fixed_binding_budget():
    project=Path(__file__).resolve().parents[1]
    spec,provenance=m.contract(project)
    assert spec.packed==113104429
    assert len(m.ranges(spec,'fetch')[1])==432
    assert m.PROBE_CAP==589971 and m.FULL_CAP==117429973
    assert m.COMBINED_CAP==118019944<120*1024*1024
    assert m.COMBINED_CAP+4251271<120*1024*1024
    assert len(m.ranges(spec,'probe')[1])==2


@pytest.mark.parametrize('status,headers,url',[
    (200,{},None),(302,{},None),(403,{},None),
    (206,{'Content-Range':'bytes 1-2/3'},None),
    (206,{'Content-Length':'9'},None),(206,{'Content-Encoding':'gzip'},None),
    (206,{},'https://elsewhere.invalid'),
])
def test_contract_failure_never_reads_or_retries(tmp_path,status,headers,url):
    response=Response(b'ab',1,2,status=status,headers=headers,url=url)
    c=client(tmp_path,response)
    with pytest.raises(m.ContractError): c.get(1,2,retry=True)
    assert response.reads==0 and c.attempted==1 and c.extra==0 and c.received==0


def test_disallowed_and_reserved_cap_no_network(tmp_path):
    c=client(tmp_path,Response(b'ab',1,2))
    with pytest.raises(m.ContractError): c.get(2,2,retry=True)
    c.reserved=c.cap
    with pytest.raises(m.ContractError): c.get(1,2,retry=True)
    assert c.opener.calls==0


def test_cross_phase_budget_pre_request(tmp_path):
    c=client(tmp_path,Response(b'ab',1,2),inherited=m.COMBINED_CAP-1)
    with pytest.raises(m.ContractError): c.get(1,2,retry=False)
    assert c.opener.calls==0


def test_probe_zero_retries_retains_failed_bytes(tmp_path):
    c=client(tmp_path,Response(b'a',1,2),mode='probe')
    with pytest.raises(m.TransferError,match='Truncated'): c.get(1,2,retry=True)
    assert c.attempted==1 and c.extra==0 and c.received==1 and c.completed==0
    assert (tmp_path/'failed_range_0001.part').read_bytes()==b'a'
    assert c.receipts[0]['body_sha256']==hashlib.sha256(b'a').hexdigest()


def test_full_retry_once_commits_only_successful_bytes(tmp_path):
    class Retry:
        n=0
        def open(self,req,timeout):
            self.n+=1
            return Response(b'a' if self.n==1 else b'ab',1,2)
    c=client(tmp_path,None); c.opener=Retry()
    assert c.get(1,2,retry=True)==b'ab'
    assert (c.attempted,c.completed,c.extra,c.received,c.reserved)==(2,1,1,3,4)
    assert (tmp_path/'failed_range_0001.part').read_bytes()==b'a'


def test_two_failures_stop_same_segment(tmp_path):
    class Fail:
        def open(self,*a,**kw): return Response(b'a',1,2)
    c=client(tmp_path,None);c.opener=Fail()
    with pytest.raises(m.TransferError): c.get(1,2,retry=True)
    assert c.attempted==2 and c.extra==1 and c.completed==0


def test_global_extra_attempt_budget(tmp_path):
    class Alternating:
        n=0
        def open(self,req,timeout):
            self.n+=1
            return Response(b'a' if self.n%2 else b'ab',1,2)
    c=client(tmp_path,None);c.opener=Alternating()
    for _ in range(16): assert c.get(1,2,retry=True)==b'ab'
    with pytest.raises(m.TransferError,match='16 extra'): c.get(1,2,retry=True)
    assert c.extra==16 and c.attempted==33


def test_attempt_wallclock_after_read_rejects_complete_body(tmp_path):
    class Clock:
        value=0
        def __call__(self): return self.value
    clock=Clock()
    def advance(): clock.value+=46
    c=client(tmp_path,Response(b'ab',1,2,onread=advance),clock=clock)
    with pytest.raises(m.TransferError,match='wall-clock'): c.get(1,2,retry=False)
    assert c.received==2 and c.completed==0


def test_phase_deadline_prevents_open(tmp_path):
    class Clock:
        value=0
        def __call__(self): return self.value
    clock=Clock();c=client(tmp_path,Response(b'ab',1,2),clock=clock)
    clock.value=m.FULL_SECONDS+1
    with pytest.raises(m.DeadlineExceeded): c.get(1,2,retry=False)
    assert c.opener.calls==0


def test_real_signal_deadline_no_sleep_and_nested_restore():
    assert signal.getitimer(signal.ITIMER_REAL)[0]==0
    with m.hard_deadline(10):
        with pytest.raises(m.DeadlineExceeded):
            with m.hard_deadline(5): signal.raise_signal(signal.SIGALRM)
        assert 0<signal.getitimer(signal.ITIMER_REAL)[0]<=10
    assert signal.getitimer(signal.ITIMER_REAL)[0]==0 and m._deadline_depth==0


def test_incomplete_read_counts_returned_partial(tmp_path):
    response=Response(b'',1,2)
    def fail(n): raise m.http.client.IncompleteRead(b'a',1)
    response.read=fail;c=client(tmp_path,response)
    with pytest.raises(m.TransferError,match='IncompleteRead'): c.get(1,2,retry=False)
    assert c.received==1 and c.receipts[0]['received']==1


def test_journal_secondary_does_not_mask_transfer(tmp_path):
    class Broken(Log):
        def add(self,event,**kw):
            if event=='attempt_terminal': raise OSError('journal error')
            super().add(event,**kw)
    c=client(tmp_path,Response(b'a',1,2));c.journal=Broken()
    with pytest.raises(m.TransferError,match='Truncated'): c.get(1,2,retry=False)
    assert c.secondary[0]['phase']=='attempt_journal'


def setup_run(tmp_path,monkeypatch):
    raw=random.Random(42).randbytes(600000)
    compressor=zlib.compressobj(wbits=-15);packed=compressor.compress(raw)+compressor.flush()
    spec=m.pure.Spec(m.pure.MEMBER,100,len(packed),len(raw),zlib.crc32(raw)&0xffffffff,8,8)
    name=spec.name.encode();extra=b'\x00'*(96-len(name))
    header=struct.pack('<4s5H3I2H',b'PK\x03\x04',20,8,8,0,0,0,0,0,len(name),len(extra))
    image=b'\x00'*100+header+name+extra+packed
    offset=100+126;tail=b'x'*m.TAIL_SIZE
    monkeypatch.setattr(m,'DATA_OFFSET',offset)
    monkeypatch.setattr(m.pure,'TAIL_SHA',m.sha(tail))
    monkeypatch.setattr(m,'contract',lambda p:(spec,{'mock':'only'}))
    class Opener:
        def __init__(self,fail=None): self.calls={};self.fail=fail
        def open(self,request,timeout):
            start,end=map(int,request.get_header('Range').split('=')[1].split('-'))
            length=end-start+1;self.calls[start]=self.calls.get(start,0)+1
            body=tail if start==m.TAIL_START else image[start:end+1]
            if self.fail=='first' and start==offset and self.calls[start]==1: body=body[:10]
            if self.fail=='always' and start==offset: body=body[:10]
            return Response(body,start,length)
    project=tmp_path/'project';(project/'results').mkdir(parents=True)
    return raw,spec,Opener,project


def test_full_two_stage_mock_download_and_no_overwrite(tmp_path,monkeypatch):
    raw,spec,Opener,project=setup_run(tmp_path,monkeypatch)
    probe=project/'results'/'probe'
    result=m.run('probe',project,probe,opener=Opener(),deadline=lambda s:nullcontext())
    assert result['status']=='probe_completed_not_object_verified'
    assert not (probe/'download.part').exists()
    report=probe/'report.json';out=project/'results'/'full'
    full=m.run('fetch',project,out,probe_report=report,probe_sha=m.pure.file_digest(report),
               opener=Opener('first'),deadline=lambda s:nullcontext())
    assert full['status']=='object_verified_not_loaded'
    assert (out/full['artifact']).read_bytes()==raw
    assert full['verified_object']['crc32']==f'{spec.crc:08x}'
    assert full['network']['extra_attempts']==1
    assert not full['unpickled'] and not full['static_pickle_scanned']
    assert m.verify_chain(out/'events.jsonl')
    with pytest.raises(FileExistsError):
        m.run('fetch',project,out,probe_report=report,probe_sha=m.pure.file_digest(report),opener=Opener())


def test_full_requires_external_probe_sha_before_output(tmp_path,monkeypatch):
    _,_,Opener,project=setup_run(tmp_path,monkeypatch)
    out=project/'results'/'no'
    with pytest.raises(m.ContractError): m.run('fetch',project,out,opener=Opener())
    probe=project/'results'/'probe'
    m.run('probe',project,probe,opener=Opener(),deadline=lambda s:nullcontext())
    with pytest.raises(m.ContractError,match='SHA'):
        m.run('fetch',project,out,probe_report=probe/'report.json',probe_sha='0'*64,opener=Opener())
    assert not out.exists()


def test_probe_failure_retains_range_but_no_object(tmp_path,monkeypatch):
    _,_,Opener,project=setup_run(tmp_path,monkeypatch)
    out=project/'results'/'failed'
    with pytest.raises(m.TransferError): m.run('probe',project,out,opener=Opener('always'),deadline=lambda s:nullcontext())
    r=json.loads((out/'report.json').read_text())
    assert r['status']=='failed_preserved' and r['network']['attempted']==4
    assert (out/'failed_range_0004.part').stat().st_size==10
    assert not (out/'download.part').exists()


def test_source_disappearance_preserves_transfer_first_error(tmp_path,monkeypatch):
    _,_,Opener,project=setup_run(tmp_path,monkeypatch)
    original=m.seal; calls=0
    def gone():
        nonlocal calls
        calls+=1
        if calls>1: raise FileNotFoundError('source gone')
        return original()
    monkeypatch.setattr(m,'seal',gone)
    out=project/'results'/'failed'
    with pytest.raises(m.TransferError,match='Truncated'):
        m.run('probe',project,out,opener=Opener('always'),deadline=lambda s:nullcontext())
    r=json.loads((out/'report.json').read_text())
    assert r['first_error']['type']=='TransferError'
    assert r['secondary_errors'][0]['phase']=='final_seal'


def test_probe_bytes_tampered_rejects_full_before_open(tmp_path,monkeypatch):
    _,_,Opener,project=setup_run(tmp_path,monkeypatch)
    probe=project/'results'/'probe'
    m.run('probe',project,probe,opener=Opener(),deadline=lambda s:nullcontext())
    report=probe/'report.json'; report_sha=m.pure.file_digest(report)
    (probe/'probe_0.bin').write_bytes(b'x'*m.CHUNK)
    out=project/'results'/'full'
    with pytest.raises(m.ContractError,match='Probe bytes'):
        m.run('fetch',project,out,probe_report=report,probe_sha=report_sha,opener=Opener())
    assert not out.exists()


def test_full_failed_ranges_never_enter_inflater(tmp_path,monkeypatch):
    _,_,Opener,project=setup_run(tmp_path,monkeypatch)
    probe=project/'results'/'probe'
    m.run('probe',project,probe,opener=Opener(),deadline=lambda s:nullcontext())
    report=probe/'report.json'; out=project/'results'/'full'
    with pytest.raises(m.TransferError,match='Truncated'):
        m.run('fetch',project,out,probe_report=report,probe_sha=m.pure.file_digest(report),
              opener=Opener('always'),deadline=lambda s:nullcontext())
    r=json.loads((out/'report.json').read_text())
    assert r['network']['extra_attempts']==1 and r['network']['attempted']==5
    assert r['partial']['packed']==r['partial']['expanded']==0
    assert (out/'download.part').stat().st_size==0
    assert (out/'failed_range_0004.part').read_bytes()==(out/'failed_range_0005.part').read_bytes()
    assert not (out/'pusht_hi_lewm_epoch15_object.ckpt').exists()


def test_validator_change_fatal_without_retry(tmp_path):
    r=Response(b'ab',1,2,headers={'ETag':'changed'})
    c=client(tmp_path,r);c.validators['etag']='original'
    with pytest.raises(m.ContractError,match='validator'): c.get(1,2,retry=True)
    assert r.reads==0 and c.extra==0


def test_signal_during_network_captured_as_failed_attempt(tmp_path):
    class Interrupted:
        def open(self,*args,**kwargs): signal.raise_signal(signal.SIGALRM)
    c=m.Client(tmp_path,Log(),{(1,2)},'probe',opener=Interrupted())
    with pytest.raises(m.TransferError,match='DeadlineExceeded'): c.get(1,2,retry=False)
    assert c.received==0 and c.attempted==1 and c.completed==0
    assert c.receipts[0]['error']['type']=='TransferError'
    assert signal.getitimer(signal.ITIMER_REAL)[0]==0
