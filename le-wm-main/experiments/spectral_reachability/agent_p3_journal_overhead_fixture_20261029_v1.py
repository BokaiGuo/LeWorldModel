"""Nonneural lifecycle/IO fixture; not a production timing attribution."""
from pathlib import Path
import time
from experiments.spectral_reachability import agent_p3_effective_schedule_20261029_v1 as p

NAME='agent_p3_journal_overhead_fixture_20261029_v1'
NAMES=('action_encoder','predictor','pred_proj','encoder','projector')


class FakeModule:
    def forward(self,*args):return args[0]


class FakeModel:
    def __init__(self):
        for name in NAMES:setattr(self,name,FakeModule())


class FakeInput:
    shape=(1,3,192)


def depth(fn):
    n=0
    while '_fn' in (getattr(fn,'__kwdefaults__',None) or {}):
        n+=1;fn=fn.__kwdefaults__['_fn']
    return n


def observe_fixture(nodes=4,raise_at=None,preexisting=False):
    """Real observe/finally on a fake model; NO NativeSignal init/inference."""
    model=FakeModel();initial={}
    if preexisting:
        for name in NAMES:getattr(model,name).forward=lambda *args:args[0]
    for name in NAMES:
        mod=getattr(model,name);initial[name]=dict(mod.__dict__)
    rows=[]
    for i in range(nodes):
        backend=p.signal.NativeSignal.__new__(p.signal.NativeSignal)
        backend.model=model;calls=[]
        backend.call=lambda resource,fn,inputs,**kw:(calls.append(resource),fn())[1]
        row=dict(node=i,before={name:depth(getattr(model,name).forward) for name in NAMES})
        try:
            with backend.observe():
                row['inside']={name:depth(getattr(model,name).forward) for name in NAMES}
                model.action_encoder.forward(FakeInput())
                model.predictor.forward(FakeInput(),FakeInput())
                model.pred_proj.forward(FakeInput())
                if i==raise_at:raise RuntimeError('fixture exception')
        except RuntimeError:row['fixture_exception']=True
        row.update(after={name:depth(getattr(model,name).forward) for name in NAMES},calls=calls,
            exact_attribute_restore=all(getattr(model,name).__dict__==initial[name] for name in NAMES))
        rows.append(row)
    return rows


class FileProbe:
    def __init__(self,path,counts):self.path=path;self.counts=counts
    def is_file(self):self.counts['is_file']+=1;return self.path.is_file()
    def stat(self):self.counts['explicit_stat']+=1;return self.path.stat()


class PathProbe:
    """Counts existing Store.check traversal without an extra profiling scan."""
    def __init__(self,path,counts):self.path=path;self.counts=counts
    def __fspath__(self):return str(self.path)
    def __truediv__(self,name):return self.path/name
    def rglob(self,pattern):
        self.counts['scans']+=1
        for child in self.path.rglob(pattern):
            self.counts['entries']+=1;yield FileProbe(child,self.counts)


class MeasuredStore(p.Store):
    """Only pytest tmpdir; inherit original check/write/namespace/emit/put."""
    def __init__(self,path):
        path=Path(path).resolve();path.mkdir()
        self.counts=dict(scans=0,entries=0,is_file=0,explicit_stat=0,check_wall_ns=0)
        self.path=PathProbe(path,self.counts);self.finalizing=False;self.index=0;self.previous=None
    def check(self,size=4096):
        begin=time.monotonic_ns()
        try:return super().check(size)
        finally:self.counts['check_wall_ns']+=time.monotonic_ns()-begin


def io_fixture(directory,files_per_node=(0,8,24)):
    """Real Store+NativeJournal+Ledger; tiny files, zero neural/physics.

    Timings are measured but never asserted monotonic. Structural traversal
    counts, return exclusion and original restore behavior are the evidence.
    """
    directory=Path(directory).resolve();store=MeasuredStore(directory/'output');out=[]
    caps={k:0 for k in p.core.AXES};caps['schedule_decisions']=1
    for node,nfiles in enumerate(files_per_node):
        for i in range(nfiles):
            # Own temporary metadata through the actual durable write path.
            store.put(f'n{node}_f{i}.json',dict(fixture=True,node=node,index=i))
        journal=p.live.NativeJournal(store,f'n{node}',lambda:store.check())
        before=dict(store.counts)
        ledger=p.core.Ledger(caps,sink=journal.event);ledger.scope=dict(node=node,fixture=True)
        body=[]
        def work():
            start=time.monotonic_ns()
            journal.event(dict(kind='fixture_body',node=node))
            body.append(time.monotonic_ns()-start)
            return 'no neural work'
        assert ledger.call('schedule',work)=='no neural work'
        receipt=ledger.receipt();after=dict(store.counts)
        journal.close()
        event=receipt['events'][0]
        out.append(dict(node=node,added_files=nfiles,receipt=receipt,
            measured_body_wall_ns=body[0],observed_call_wall_ns=event['ended_ns']-event['started_ns'],
            checks_and_IO_counts={k:after[k]-before[k] for k in after},
            note='call interval includes attempt sink + body journal; return sink occurs after ended_ns'))
    return dict(schema=NAME+'.fixture',rows=out,production_attribution=False,
        neural_calls=0,physics_calls=0,timing_monotonicity_asserted=False,
        interpretation='traversal counts demonstrate growing IO work, not the fraction of actual CUDA job growth')
