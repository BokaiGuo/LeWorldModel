from contextlib import contextmanager
from copy import deepcopy
import fcntl
import hashlib
import json
import os

import numpy as np
import pytest

from experiments.spectral_reachability import codex_p2_prefix_recompile_host_20261030_v1 as h
from experiments.spectral_reachability import codex_p2_prefix_recompile_saved_20261030_v1 as saved
from experiments.spectral_reachability import codex_p2_prefix_recompile_kernel_20261030_v1 as k


@contextmanager
def scope(_):
    yield


class FixtureAdapter:
    """Synthetic accounting only; never a model or CUDA execution."""
    def __init__(self,meter):
        self.meter=meter

    def packet(self,fn):
        def nested():
            for axis in ('action_encoder','predict','predictor','pred_proj'):
                self.meter.call(axis,lambda:None)
            return fn()
        return self.meter.call('packet',nested)

    def root_jacobian(self,context):
        def work():
            self.packet(lambda:None);self.packet(lambda:None)
            return np.vstack((np.diag(np.arange(20,0,-1)),np.zeros((172,20))))
        return self.meter.call('jacobian',work)

    def packet_jacobian(self,context):
        def work():
            self.packet(lambda:None)
            return np.vstack((np.diag(np.arange(10,0,-1)),np.zeros((182,10))))
        return self.meter.call('jacobian',work)

    def step(self,context,packet):
        def work():
            c=deepcopy(context)
            c['latent_history'][-1][0]+=float(np.sum(packet))/100
            return c
        return self.packet(work)


@pytest.fixture(scope='module')
def cases():
    context={'latent_history':[[0.0]*192],'action_history':[]}
    return [dict(base=i,context=deepcopy(context),goal=[0.0]*192) for i in range(4)]


def test_real_prepare_source_and_contract():
    v=h.material()
    assert len(v['parent']['cases'])==4 and v['protocol']==k.contract()
    assert v['physical'] is False and v['scientific_pass'] is False


def test_lock_inode_must_be_inherited(tmp_path):
    path=tmp_path/'lock';path.touch()
    with pytest.raises(ValueError,match='inherited'):
        h.inherited_lock(path)
    fd=os.open(path,os.O_RDWR)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert h.inherited_lock(path)['inode']==os.fstat(fd).st_ino
    finally:
        os.close(fd)


@pytest.mark.parametrize('arm',k.ARMS)
def test_branch_consumer_and_tamper(cases,arm):
    events=[];branch=k.run_branch(cases[0],arm,FixtureAdapter,events.append,scope)
    result=saved.validate_branch(cases[0],branch,events)
    assert result['P2_admission'] is False
    bad=deepcopy(branch)
    if arm=='joint':
        bad['candidates'][0]['cost']+=1
    else:
        bad['prefix_contexts'][0]['jacobian'][0][0]+=1
    # Superficially reseal the branch hash, leaving the algorithmic witnesses intact.
    altered=deepcopy(events);altered[-1]['event']['result_sha256']=k.old.digest(bad)
    with pytest.raises(ValueError):
        saved.validate_branch(cases[0],bad,altered)


def test_complete_saved_batch_and_artifact_tamper(tmp_path,monkeypatch,cases):
    panel=tmp_path/'panel.json';panel.write_text(json.dumps(dict(cases=cases)))
    fake={'parent':dict(panel_path=str(panel),gpu_uuid='fixture',expected_model_hash='hash')}
    monkeypatch.setattr(h,'verify',lambda *_:fake)
    monkeypatch.setattr(h.identity,'verify',lambda *_:None)
    proof=dict(device=1,inode=2,fd=9)
    out=tmp_path/'run';out.mkdir()
    j=h.journal.Journal(out/'child_events.jsonl',h.CAP,1024)
    j.event(dict(kind='child_start',lock_proof=proof,order=k.contract()['order']))
    j.event(dict(kind='model_loaded',device=dict(uuid='fixture'),model_sha256='hash'))
    rows=[]
    for i,(base,arm) in enumerate(k.contract()['order']):
        branch=k.run_branch(cases[base],arm,FixtureAdapter,j.event,scope)
        name=f'branch_{i}.json';sha=h.prep.original.previous.write_new(out/name,branch)
        rows.append(dict(base=base,arm=arm,status='complete',file=name,sha256=sha,
                         resource_counts=branch['resource_counts']))
        j.event(dict(kind='branch_file',index=i,file=name,sha256=sha))
    j.append(dict(kind='child_terminal',status='exploratory_complete',error=None),terminal=True)
    tail=j.tail;j.close()
    report=dict(schema=h.NAME+'.report',status='exploratory_complete',prepare_sha256='fixture',
        error=None,lock_proof=proof,device=dict(uuid='fixture'),model_before='hash',model_after='hash',
        physical=False,P2_admission=False,loaded_sources=[],journal_tail=tail,rows=rows)
    reportsha=h.prep.original.previous.write_new(out/'report.json',report)
    j=h.journal.Journal(out/'supervisor_events.jsonl',1024*1024,1024)
    for event in [dict(kind='lock_acquired',device=1,inode=2),dict(kind='child_spawn_intent'),
                  dict(kind='child_spawned'),dict(kind='child_exited',returncode=0),
                  dict(kind='supervisor_terminal',returncode=0)]:
        j.event(event)
    j.close()
    supsha=hashlib.sha256((out/'supervisor_events.jsonl').read_bytes()).hexdigest()
    receipt=saved.consume(out,reportsha,supsha,'fixture','fixture')
    assert len(receipt['rows'])==8 and receipt['P2_admission'] is False
    (out/'branch_0.json').write_bytes(b'tampered')
    with pytest.raises(ValueError):
        saved.consume(out,reportsha,supsha,'fixture','fixture')
