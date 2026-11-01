"""RAM-testable native-ledger orchestration; not a production CUDA runner."""
from copy import deepcopy
import numpy as np
from experiments.spectral_reachability import agent_p1_fixed_prefix_fd_20261029_v1 as fd

F, core = fd.F, fd.core
require, raw = fd.require, fd.raw
CONTROL = {'fd_intent','fd_return','pair_null'}


def roster(): return [dict(index=i,status='not_started',result=None,resources=None) for i in range(64)]


def dispatch(groups, source_rows, backend_factory, journal_factory):
    """Callback entry; the caller must authenticate the source before invoking."""
    require(len(groups)==len(source_rows)==64,'fixed full 64 groups')
    rows=roster()
    for i,(q,source,row) in enumerate(zip(groups,source_rows,rows,strict=True)):
        require(q['index']==source['index']==i and source['status']=='sealed','source group order')
        row['status']='attempted'; journal=None; ledger=None
        try:
            journal=journal_factory(i)
            valid=sum(p['valid'] for p in fd.pairs(q,source['result']))
            ledger=core.Ledger(F.caps(2*valid,0),sink=journal.event)
            backend=backend_factory(q['task'],journal,ledger)
            with backend.observe(): row['result']=fd.evaluate(q,source['result'],backend,journal.event)
            require(ledger.attempted==ledger.completed==F.caps(2*valid,0),'all valid pair resources')
            row['status']='sealed'
        except Exception as exc:
            row.update(status='failed_partial',error=type(exc).__name__+': '+str(exc))
        finally:
            if ledger is not None:row['resources']=ledger.receipt()
        if row['status']!='sealed':break
    return dict(rows=rows,group_denominator=64,pair_denominator=2560,
                status='complete_model_only' if all(x['status']=='sealed' for x in rows) else 'partial_failclosed',
                physical_release=False,P1_admission=False)


def audit_group(q, source_result, result, receipt, events):
    """Rebuild all native calls, then rerun the saved-only callback math."""
    pairs=fd.pairs(q,source_result); schedule=[]
    for i,p in enumerate(pairs):
        if p['valid']:
            for sign in ('plus','minus'):
                schedule.append((f'fd_{i:02}_{sign}',q['expected_mid'],p[sign+'_commands']))
    native=[e for e in events if e['kind'] not in CONTROL]
    starts,ends,_=F.audit_native(q['task'],receipt,native,schedule)
    require(len(starts)==len(ends)==len(schedule) and receipt['caps']==F.caps(len(schedule),0),'exact native pair budget')
    class Replay:
        def __init__(self):self.i=0
        def step(self,c,a,label):
            i=self.i; self.i+=1
            require(i<len(schedule) and schedule[i][0]==label and raw(c)==raw(schedule[i][1]) and
                    F.bits(a)==F.bits(schedule[i][2]),'saved call identity')
            return deepcopy(ends[i]['context'])
    replay=Replay(); controls=[]; rebuilt=fd.evaluate(q,source_result,replay,controls.append)
    require(replay.i==len(schedule) and raw(rebuilt)==raw(result) and raw(controls)==raw([e for e in events if e['kind'] in CONTROL]),
            'full saved pair replay')
    positions={id(e):i for i,e in enumerate(events)}
    for e in events:
        if e['kind']!='fd_intent':continue
        returns=[x for x in events if x['kind']=='fd_return' and positions[id(x)]>positions[id(e)]]
        require(returns and sum(positions[id(e)]<positions[id(s)]<positions[id(returns[0])] for s in starts)==1,
                'each pair call bracketed')
    return dict(index=q['index'],pairs=40,valid_pairs=result['valid_pairs'],native_forwards=len(schedule),
                gradient_correctness_established=False,P1_admission=False)
