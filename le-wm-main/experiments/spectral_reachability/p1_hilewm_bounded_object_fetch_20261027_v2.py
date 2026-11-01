"""Two-stage, deadline/cumulative-budget bounded Hi-LeWM acquisition. No unpickling."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import http.client
import json
import os
from pathlib import Path
import signal
import socket
import time
import urllib.error
import urllib.request

V1_SHA = '961c23a8ab0af587a8169ce0163005cc36eb88ad15493fbcd0b72b9b1614cb29'
V1_PATH = Path(__file__).with_name('p1_hilewm_bounded_object_fetch_20261027_v1.py')
if hashlib.sha256(V1_PATH.read_bytes()).hexdigest() != V1_SHA:
    raise RuntimeError('Sealed pure helper dependency changed; refusing import')
from . import p1_hilewm_bounded_object_fetch_20261027_v1 as pure

CHUNK = 256 * 1024
TAIL_START, TAIL_SIZE = 2299798367, 65557
DATA_OFFSET = 634194454
ATTEMPT_SECONDS, SOCKET_SECONDS = 45, 10
PROBE_SECONDS, FULL_SECONDS = 150, 7200
MAX_EXTRA_ATTEMPTS = 16
PROBE_CAP = TAIL_SIZE + 126 + 2*CHUNK  # 589,971
FULL_CAP = 113104429 + 2*TAIL_SIZE + 126 + MAX_EXTRA_ATTEMPTS*CHUNK
COMBINED_CAP = PROBE_CAP + FULL_CAP  # 118,019,944 < 120MiB
SCHEMA = 'p1_hilewm_bounded_object_fetch_20261027_v2'
_deadline_depth = 0


class ContractError(ValueError): pass
class TransferError(OSError): pass
class DeadlineExceeded(TimeoutError): pass


def sha(data): return hashlib.sha256(data).hexdigest()


def seal():
    return {'source_sha256': pure.file_digest(Path(__file__)), 'pure_v1_sha256': pure.file_digest(V1_PATH)}


@contextmanager
def hard_deadline(seconds):
    """Linux main-thread timer bounds DNS/connect/headers/body, not just idle reads."""
    global _deadline_depth
    if seconds <= 0: raise DeadlineExceeded('Phase wall-clock exhausted')
    old_timer = signal.getitimer(signal.ITIMER_REAL)
    if old_timer[0] != 0 and _deadline_depth == 0:
        raise ContractError('Existing process timer: do not override')
    previous = signal.getsignal(signal.SIGALRM)
    started = time.monotonic()
    def alarm(signum, frame): raise DeadlineExceeded('Attempt wall-clock exhausted')
    signal.signal(signal.SIGALRM, alarm)
    _deadline_depth += 1
    signal.setitimer(signal.ITIMER_REAL, min(seconds,old_timer[0]) if old_timer[0] else seconds)
    try: yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        _deadline_depth -= 1
        if old_timer[0]:
            remaining = old_timer[0]-(time.monotonic()-started)
            if remaining > 0: signal.setitimer(signal.ITIMER_REAL, remaining,old_timer[1])


class Journal:
    def __init__(self, output):
        self.output = output
        self.file = (output/'events.jsonl').open('x')
        self.previous, self.sequence = '0'*64, 0
    def add(self, event, **fields):
        record = {'sequence': self.sequence, 'previous_sha256': self.previous,
                  'time_ns': time.time_ns(), 'event': event, **fields}
        h = sha(json.dumps(record,sort_keys=True,separators=(',',':')).encode())
        line = json.dumps({'record':record,'sha256':h})+'\n'
        pure.space(self.output,len(line.encode()))
        self.file.write(line); self.file.flush(); os.fsync(self.file.fileno())
        self.previous, self.sequence = h, self.sequence+1
    def close(self): self.file.close()


def verify_chain(path):
    previous = '0'*64
    with path.open() as file:
        for i,line in enumerate(file):
            row = json.loads(line); record = row['record']
            if record['sequence'] != i or record['previous_sha256'] != previous:
                raise ContractError('Journal sequence/link mismatch')
            previous = sha(json.dumps(record,sort_keys=True,separators=(',',':')).encode())
            if previous != row['sha256']: raise ContractError('Journal hash mismatch')
    return previous


def ranges(spec, mode):
    control = {(TAIL_START,TAIL_SIZE),(spec.offset,30),(spec.offset+30,96)}
    payload = [(DATA_OFFSET+i,min(CHUNK,spec.packed-i)) for i in range(0,spec.packed,CHUNK)]
    if mode=='probe': payload=payload[:2]
    return control | set(payload), payload


class Client:
    def __init__(self, output, journal, allowed, mode, inherited=0, opener=None,
                 clock=time.monotonic, deadline=hard_deadline):
        self.output,self.journal,self.allowed,self.mode = output,journal,allowed,mode
        self.opener = opener or urllib.request.build_opener(pure.NoRedirect())
        self.clock,self.deadline = clock,deadline
        self.started=clock(); self.ends=self.started+(PROBE_SECONDS if mode=='probe' else FULL_SECONDS)
        self.cap=PROBE_CAP if mode=='probe' else FULL_CAP
        self.inherited=inherited
        self.reserved=self.received=self.attempted=self.completed=self.extra=0
        self.validators={}; self.receipts=[]; self.secondary=[]

    def accounting(self):
        return {'attempted':self.attempted,'completed':self.completed,'extra_attempts':self.extra,
                'reserved_body_bytes':self.reserved,'received_body_bytes':self.received,
                'phase_cap':self.cap,'probe_reserved_inherited':self.inherited,
                'combined_reserved_body_bytes':self.reserved+self.inherited,
                'combined_cap':COMBINED_CAP,'elapsed_seconds':self.clock()-self.started}

    def small(self,start,length): return self.get(start,length,retry=False)

    def get(self,start,length,*,retry):
        if (start,length) not in self.allowed: raise ContractError('Range outside frozen schedule')
        for ordinal in range(2 if retry and self.mode=='fetch' else 1):
            if ordinal:
                if self.extra>=MAX_EXTRA_ATTEMPTS: raise TransferError('Global 16 extra-attempt cap')
                self.extra+=1
            try: return self.attempt(start,length,ordinal)
            except TransferError:
                if not (retry and self.mode=='fetch' and ordinal==0): raise
        raise AssertionError('Unreachable')

    def attempt(self,start,length,ordinal):
        remaining=self.ends-self.clock()
        if remaining<=0: raise DeadlineExceeded('Phase wall-clock exhausted')
        if self.reserved+length>self.cap or self.inherited+self.reserved+length>COMBINED_CAP:
            raise ContractError('Cumulative reserved-body cap')
        self.reserved+=length; self.attempted+=1
        row={'attempt':self.attempted,'ordinal':ordinal,'start':start,'length':length,
             'url':pure.URL,'received':0,'status':None,'headers':{}}
        self.receipts.append(row)
        self.journal.add('attempt',**row)
        data=bytearray(); first=None; response=None; started=self.clock()
        try:
            with self.deadline(min(ATTEMPT_SECONDS,remaining)):
                request=urllib.request.Request(pure.URL,headers={
                    'Range':f'bytes={start}-{start+length-1}','Accept-Encoding':'identity',
                    'User-Agent':'LeWM-bounded-object-v2'})
                try: response=self.opener.open(request,timeout=min(SOCKET_SECONDS,remaining))
                except urllib.error.HTTPError as exc: response=exc
                row['status']=response.status
                row['headers']={k.lower():v for k,v in response.headers.items() if k.lower() in (
                    'content-range','content-length','content-encoding','content-type',
                    'etag','last-modified','date','accept-ranges')}
                self.journal.add('headers',attempt=self.attempted,status=row['status'],headers=row['headers'])
                if response.geturl()!=pure.URL or response.status!=206:
                    raise ContractError('Non-206/redirect: body not read')
                h=row['headers']
                if (h.get('content-range')!=f'bytes {start}-{start+length-1}/{pure.ARCHIVE_BYTES}'
                        or h.get('content-length')!=str(length)
                        or h.get('content-encoding','identity').lower()!='identity'):
                    raise ContractError('Range/length/encoding mismatch: body not read')
                for key in ('etag','last-modified'):
                    if key in h:
                        if key in self.validators and h[key]!=self.validators[key]:
                            raise ContractError('Archive validator changed')
                        self.validators[key]=h[key]
                while len(data)<length:
                    if self.clock()>=self.ends or self.clock()-started>=ATTEMPT_SECONDS:
                        raise DeadlineExceeded('Attempt/phase wall-clock exhausted')
                    try: block=response.read(min(65536,length-len(data)))
                    except http.client.IncompleteRead as exc:
                        block=exc.partial
                        if len(block)>length-len(data): raise ContractError('Overlong partial read')
                        data.extend(block); self.received+=len(block)
                        raise TransferError('IncompleteRead') from exc
                    if not block: raise TransferError('Truncated range')
                    if len(block)>length-len(data): raise ContractError('Overlong body')
                    data.extend(block); self.received+=len(block)
                if self.clock()>self.ends or self.clock()-started>ATTEMPT_SECONDS:
                    raise DeadlineExceeded('Completed after wall-clock limit')
        except (DeadlineExceeded,TimeoutError,socket.timeout,urllib.error.URLError,
                ConnectionError,http.client.HTTPException) as exc:
            first=TransferError(f'{type(exc).__name__}: {exc}')
        except BaseException as exc: first=exc
        finally:
            if response is not None:
                try: response.close()
                except BaseException as exc:
                    self.secondary.append({'phase':'response_close','message':str(exc)})
                    if first is None: first=ContractError('Response close failed')
            row.update(received=len(data),body_sha256=sha(bytes(data)),elapsed_seconds=self.clock()-started,
                       error=None if first is None else {'type':type(first).__name__,'message':str(first)})
            if first is not None and data:
                try:
                    pure.space(self.output,len(data))
                    path=self.output/f'failed_range_{self.attempted:04d}.part'
                    with path.open('xb') as f:
                        f.write(data); f.flush(); os.fsync(f.fileno())
                    row['failed_body_file']=path.name
                except BaseException as exc:
                    self.secondary.append({'phase':'failed_body_preservation','message':str(exc)})
            try: self.journal.add('attempt_terminal',**row)
            except BaseException as exc:
                self.secondary.append({'phase':'attempt_journal','message':str(exc)})
                if first is None: first=ContractError('Attempt receipt not durable')
        if first is not None: raise first
        self.completed+=1
        return bytes(data)


def probe_binding(path, expected_sha, spec, provenance):
    if not expected_sha or pure.file_digest(path)!=expected_sha:
        raise ContractError('External probe report SHA mismatch')
    report=json.loads(path.read_text())
    if (report.get('schema')!=SCHEMA or report.get('mode')!='probe'
            or report.get('status')!='probe_completed_not_object_verified'
            or report.get('first_error') or report.get('secondary_errors')
            or report.get('source_unchanged') is not True
            or report.get('seal')!=seal() or report.get('provenance')!=provenance
            or report.get('spec')!=pure.asdict(spec)):
        raise ContractError('Probe source/provenance/status mismatch')
    network=report['network']
    if (network['attempted']!=5 or network['completed']!=5 or network['extra_attempts']!=0
            or network['reserved_body_bytes']!=PROBE_CAP
            or network['received_body_bytes']!=PROBE_CAP):
        raise ContractError('Probe budget incomplete')
    if pure.file_digest(path.parent/'events.jsonl')!=report['journal_sha256']:
        raise ContractError('Probe journal file changed')
    verify_chain(path.parent/'events.jsonl')
    expected=[(DATA_OFFSET,CHUNK),(DATA_OFFSET+CHUNK,CHUNK)]
    if len(report['probe_ranges'])!=2: raise ContractError('Probe range count')
    for i,(row,(start,length)) in enumerate(zip(report['probe_ranges'],expected)):
        data=(path.parent/f'probe_{i}.bin').read_bytes()
        if (row['start'],row['length'])!=(start,length) or len(data)!=length or sha(data)!=row['sha256']:
            raise ContractError('Probe bytes/range changed')
    return network['reserved_body_bytes']


def contract(project):
    spec,provenance=pure.binding(project)
    if spec.offset+30+96!=DATA_OFFSET: raise ContractError('Fixed local-header layout')
    return spec,provenance


def run(mode,project,output,*,probe_report=None,probe_sha=None,opener=None,clock=time.monotonic,deadline=hard_deadline):
    spec,provenance=contract(project)
    inherited=0
    if mode=='fetch':
        if probe_report is None: raise ContractError('Full download requires successful external-bound probe')
        inherited=probe_binding(probe_report,probe_sha,spec,provenance)
    if mode not in ('probe','fetch'): raise ContractError('Unknown phase')
    results=(project/'results').resolve(strict=True)
    if output.parent.resolve(strict=True)!=results: raise ContractError('NEW direct results child required')
    if mode=='fetch' and output.absolute()==probe_report.parent.absolute(): raise ContractError('Do not append to probe')
    pure.space(results,(spec.expanded if mode=='fetch' else 2*CHUNK)+8*pure.MIB)
    output.mkdir(exist_ok=False)
    journal=Journal(output)
    allowed,payload=ranges(spec,mode)
    client=Client(output,journal,allowed,mode,inherited,opener,clock,deadline)
    report={'schema':SCHEMA,'mode':mode,'status':'started','seal':seal(),
            'spec':pure.asdict(spec),'provenance':provenance,'first_error':None,'secondary_errors':[],
            'probe_report_sha256':probe_sha,'neural_calls':0,'physics_calls':0,'unpickled':False,
            'static_pickle_scanned':False,'probe_ranges':[]}
    sink=None; original=None
    phase_context=deadline(max(0,client.ends-clock())); phase_entered=False
    try:
        phase_context.__enter__(); phase_entered=True
        journal.add('start',seal=report['seal'],mode=mode)
        if sha(client.small(TAIL_START,TAIL_SIZE))!=pure.TAIL_SHA:
            raise ContractError('Remote tail binding changed')
        if pure.local_header(spec,client)!=DATA_OFFSET: raise ContractError('Data offset mismatch')
        if mode=='probe':
            for i,(start,length) in enumerate(payload):
                data=client.get(start,length,retry=False)
                pure.space(output,len(data))
                with (output/f'probe_{i}.bin').open('xb') as f:
                    f.write(data); f.flush(); os.fsync(f.fileno())
                report['probe_ranges'].append({'start':start,'length':length,'sha256':sha(data)})
            report['status']='probe_completed_not_object_verified'
        else:
            with (output/'download.part').open('xb') as f:
                sink=pure.InflateSink(f,spec,output)
                for start,length in payload:
                    # No failed range or probe bytes enter this fresh decompression stream.
                    data=client.get(start,length,retry=True)
                    sink.feed(data)
                    f.flush(); os.fsync(f.fileno())
                    journal.add('committed_range',start=start,length=length,sha256=sha(data),
                                packed=sink.packed,expanded=sink.expanded,prefix_sha256=sink.raw_sha.hexdigest())
                verified=sink.finish()
            if sha(client.small(TAIL_START,TAIL_SIZE))!=pure.TAIL_SHA:
                raise ContractError('Final tail changed')
            if pure.file_digest(output/'download.part')!=verified['object_sha256']:
                raise ContractError('Persisted object SHA mismatch')
            report['verified_object']=verified
            journal.add('object_verified',**verified)
            final=output/'pusht_hi_lewm_epoch15_object.ckpt'
            os.link(output/'download.part',final); pure.fsync_directory(output)
            report['artifact']=final.name
            report['status']='object_verified_not_loaded'
    except BaseException as exc:
        original=exc
        report['first_error']={'type':type(exc).__name__,'message':str(exc)}
        report['status']='failed_preserved'
    finally:
        if phase_entered:
            try: phase_context.__exit__(None,None,None)
            except BaseException as exc:
                report['secondary_errors'].append({'phase':'deadline_close','message':str(exc)})
        report['network']=client.accounting(); report['attempt_receipts']=client.receipts
        report['secondary_errors'].extend(client.secondary)
        if sink:
            report['partial']={'packed':sink.packed,'expanded':sink.expanded,'prefix_sha256':sink.raw_sha.hexdigest()}
        for phase,action in (
            ('final_seal',lambda: report.update(source_unchanged=seal()==report['seal'])),
            ('final_journal',lambda:journal.add('terminal',status=report['status'],first_error=report['first_error'])),
            ('journal_close',journal.close),
            ('journal_digest',lambda:report.update(journal_sha256=pure.file_digest(output/'events.jsonl'))),
        ):
            try: action()
            except BaseException as exc: report['secondary_errors'].append({'phase':phase,'message':str(exc)})
        if report['secondary_errors'] or report.get('source_unchanged') is not True: report['status']='failed_preserved'
        try:
            pure.space(output,2*pure.MIB)
            with (output/'report.json').open('x') as f:
                json.dump(report,f,indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
            pure.fsync_directory(output)
        except BaseException as exc:
            if original is not None: raise original from exc
            raise
    if original is not None: raise original
    if report['status']=='failed_preserved': raise ContractError('Finalization failed; inspect receipt')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('plan','probe','fetch'))
    p.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--output',type=Path)
    p.add_argument('--probe-report',type=Path)
    p.add_argument('--probe-report-sha256')
    p.add_argument('--acknowledge-probe-network',action='store_true')
    p.add_argument('--acknowledge-full-download',action='store_true')
    args=p.parse_args(); project=args.project.resolve()
    if args.mode=='plan':
        spec,provenance=contract(project)
        print(json.dumps({'schema':SCHEMA,'spec':pure.asdict(spec),'provenance':provenance,'seal':seal(),
            'probe_cap':PROBE_CAP,'full_cap':FULL_CAP,'combined_cap':COMBINED_CAP,
            'chunk':CHUNK,'attempt_seconds':ATTEMPT_SECONDS,'socket_seconds':SOCKET_SECONDS,
            'probe_seconds':PROBE_SECONDS,'full_seconds':FULL_SECONDS,'max_extra_attempts':MAX_EXTRA_ATTEMPTS,
            'full_payload_ranges':len(ranges(spec,'fetch')[1]),'network_executed':False},indent=2))
        return
    if args.output is None: p.error('NEW --output required')
    if args.mode=='probe' and not args.acknowledge_probe_network: p.error('Separate MAIN probe approval required')
    if args.mode=='fetch' and (not args.acknowledge_full_download or args.probe_report is None or not args.probe_report_sha256):
        p.error('Separate full approval and external successful probe SHA required')
    print(json.dumps(run(args.mode,project,args.output.absolute(),probe_report=args.probe_report,
                         probe_sha=args.probe_report_sha256),indent=2))


if __name__=='__main__': main()
