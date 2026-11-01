"""H15 saved-auditable single-slot execution seam; HOST dispatch disabled.

The real H15 NativeTree interface is wired but no GPU runner is authorized:
source/model/precision closure and a two-root host lifecycle remain outstanding.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from experiments.spectral_reachability import p2_h15_sra_prepare_20261029 as prep
from experiments.spectral_reachability import agent_p3_parent_native_cuda_20261029_v1 as native
from experiments.spectral_reachability import agent_p3_fullcommit_live_20261028_v1 as accounting
from experiments.spectral_reachability import agent_p3_runtime_contract_20261028_v1 as core
from experiments.spectral_reachability import agent_p3_c3_k17_physical_transport_20261029_v1 as transport

NAME='p2_h15_sra_host_candidate_20261029_v1'
need=core.require
MAX_JOURNAL=256*1024**2


def caps():
    value=accounting.predict_axes(132)
    value.update(jacobian_API=1,VJP_directions=192,grad_API=192,raw_backward_callbacks=192)
    return value


def host_backend(task, model, device, sink, guard, *, authorized=False):
    """Actual native constructor signature; deliberately unavailable for dispatch."""
    raise ValueError('HOST CUDA not authorized: source/model/precision closure incomplete')


def validate_case(case, task):
    """Fail closed on any disagreement with a reconstructed prepare case.

    Hash namespaces may differ between producers; equality is required rather
    than assuming identifiers with the same label have the same semantics.
    """
    task=core.validate_task(task); b=task['binding']; n=case['action_normalizer']
    need(case['task_id']==b['task_id'] and case['task_sha256']==b['task_sha256'] and
        case['native_root']['root_sha256']==b['root_sha256'] and case['model_sha256']==b['model_sha256'] and
        _canonical(case['root_context'])==_canonical(task['root_context']) and
        _canonical(case['goal'])==_canonical(task['goal']) and _canonical(case['q'])==_canonical(task['projection']) and
        _canonical(n['mean'])==_canonical(task['action_mean']) and _canonical(n['std'])==_canonical(task['action_std']),
        'source task/root/model/context/normalizer mismatch')
    return True


def _canonical(value): return core.canonical(value)
def _sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _put(path,value):
    data=_canonical(value)+b'\n'
    with Path(path).open('xb') as stream:
        need(stream.write(data)==len(data),'full report write');stream.flush();os.fsync(stream.fileno())
    fd=os.open(Path(path).parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def run_fixture(directory,task,factory):
    """Fixture only; writes native events before return, no physical action."""
    task=core.validate_task(task); path=Path(directory)
    need(not path.exists() and not path.is_symlink() and path.parent.is_dir(),'exclusive directory')
    path.mkdir()
    journal=transport.Journal(path/'events.jsonl',MAX_JOURNAL)
    class Sink:
        def event(self,event):journal.emit('event',event)
    sink=Sink(); ledger=core.Ledger(caps(),sink=sink.event)
    report=dict(schema=NAME+'.report',status='failed_partial',task_sha256=core.digest(task),
        backend='RAM_FIXTURE',result=None,resources=None,error=None,ready_to_execute=False,physical_release=False)
    try:
        backend=factory(deepcopy(task),sink)
        need(getattr(backend,'kind',None)=='RAM_FIXTURE','fixture backend only')
        request=core.request(task,dict(context=task['root_context']),operation='compile_finite_horizon_J',horizon=3)
        def invoke(kind,request,fn):
            sink.event(dict(kind='native_call_attempt',method=kind,task=task,request=request))
            try: value=fn()
            except BaseException as exc:
                sink.event(dict(kind='native_call_failure',error=dict(type=type(exc).__name__,message=str(exc))))
                raise
            sink.event(dict(kind='native_call_return',result=value))
            return value
        ledger.scope=dict(stage='signal',index=0)
        signal=invoke('signal',request,lambda:backend.signal(request,ledger))
        bank=prep.spectral_bank(signal['jacobian'],signal['nominal'],task['goal'])
        sink.event(dict(kind='bank',value=bank))
        scored=[]
        for i,actions in enumerate(bank['raw15']):
            ledger.scope=dict(stage='candidate',index=i)
            req=dict(parent_context=task['root_context'],actions=actions,segmented=False)
            path_out=invoke('replay',req,lambda:backend.replay(req['parent_context'],actions,False,ledger))
            core.checked_path(task,task['root_context'],actions,path_out)
            delta=np.asarray(path_out[-1]['latent_history'][-1],np.float64)-np.asarray(task['goal'],np.float64)
            score=float(delta@delta/192)
            need(np.isfinite(score),'finite candidate score')
            scored.append(dict(index=i,score=score,path=path_out))
            sink.event(dict(kind='candidate',index=i,score=score))
        winner=min(range(40),key=lambda i:(scored[i]['score'],i))
        paths=[];actions=bank['raw15'][winner]
        for i,segmented in enumerate((False,True)):
            ledger.scope=dict(stage='verification',index=i)
            req=dict(parent_context=task['root_context'],actions=actions,segmented=segmented)
            paths.append(invoke('replay',req,lambda:backend.replay(req['parent_context'],actions,segmented,ledger)))
        need(paths[0]==paths[1]==scored[winner]['path'],'whole/segmented exact replay')
        need(ledger.attempted==ledger.completed==caps(),'exact native budget')
        report.update(status='fixture_complete',result=dict(bank=bank,candidates=scored,winner=winner,replays=paths))
    except BaseException as exc:
        report['error']=dict(type=type(exc).__name__,message=str(exc))
    finally:
        report['resources']=ledger.receipt()
        journal.close()
    report['journal_sha256']=_sha(path/'events.jsonl')
    _put(path/'report.json',report)
    return dict(status=report['status'],report_sha256=_sha(path/'report.json'))


def consume(directory,report_sha256,task,*,permit_fixture=False):
    """Fail closed on failures; verify all saved native IO and rebuild selection."""
    path=Path(directory); task=core.validate_task(task)
    need(not path.is_symlink() and _sha(path/'report.json')==report_sha256,'external report SHA')
    report=json.loads((path/'report.json').read_bytes())
    need(report['status']=='fixture_complete','partial or failed slot')
    need(permit_fixture is True and report['backend']=='RAM_FIXTURE','fixtures cannot certify HOST')
    need(report['schema']==NAME+'.report' and report['task_sha256']==core.digest(task) and
         report['ready_to_execute'] is False and report['physical_release'] is False and report['error'] is None,'identity/flags')
    need(_sha(path/'events.jsonl')==report['journal_sha256'],'journal SHA')
    events=[e['payload'] for e in transport.read_journal(path/'events.jsonl')]
    receipt=report['resources']; accounting.audit_resources(receipt,[e for e in events if e['kind'].startswith('resource_')])
    need(receipt['caps']==caps() and receipt['attempted']==receipt['completed']==caps(),'exact paid resource caps')
    calls=[];active=None;bank_event=None;candidate_events=[]
    for e in events:
        if e['kind']=='native_call_attempt':
            need(active is None and e['task']==task,'native call task/sequence');active=[e]
        elif active is not None:
            active.append(e)
            if e['kind'] in ('native_call_return','native_call_failure'):
                need(native.audit_native(active) is True,'full native IO audit');calls.append(active);active=None
        elif e['kind']=='bank':
            need(bank_event is None,'unique bank');bank_event=e['value']
        elif e['kind']=='candidate':candidate_events.append(e)
        else:raise ValueError('unexpected event outside native call')
    need(active is None and len(calls)==43 and len(candidate_events)==40,'full 1 signal + 40 candidate + 2 replay')
    signal=calls[0][-1]['result']; need(calls[0][0]['method']=='signal' and calls[0][0]['request']['horizon']==3,'H15 signal')
    bank=prep.spectral_bank(signal['jacobian'],signal['nominal'],task['goal'])
    result=report['result'];need(_canonical(bank)==_canonical(bank_event)==_canonical(result['bank']),'bank recomputation')
    scored=[]
    for i,call in enumerate(calls[1:41]):
        req=call[0]['request'];actions=bank['raw15'][i];out=call[-1]['result']
        need(call[0]['method']=='replay' and req==dict(parent_context=task['root_context'],actions=actions,segmented=False),'candidate native request')
        delta=np.asarray(out[-1]['latent_history'][-1],np.float64)-np.asarray(task['goal'],np.float64)
        score=float(delta@delta/192);scored.append(dict(index=i,score=score,path=out))
        need(candidate_events[i]==dict(kind='candidate',index=i,score=score),'candidate score')
    winner=min(range(40),key=lambda i:(scored[i]['score'],i))
    need(result['candidates']==scored and result['winner']==winner,'winner recomputation and stable tie')
    for i,call in enumerate(calls[41:]):
        need(call[0]['request']==dict(parent_context=task['root_context'],actions=bank['raw15'][winner],segmented=bool(i)) and
            call[-1]['result']==scored[winner]['path']==result['replays'][i],'exact paid replay')
    return dict(schema=NAME+'.consume',candidate_count=40,winner=winner,resources=receipt,
        fixture_only=True,ready_to_execute=False,physical_release=False,scientific_admission=False,
        limitations='no HOST source/model/precision closure, no real hardware, no physical result')
