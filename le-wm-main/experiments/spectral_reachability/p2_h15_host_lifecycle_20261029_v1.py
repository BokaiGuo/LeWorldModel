"""Saved two-root H15 native lifecycle candidate. HOST dispatch stays disabled."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import numpy as np

from experiments.spectral_reachability import p2_h15_host_attestation_20261029_v1 as att
from experiments.spectral_reachability import p2_h15_bridge_lifecycle_20261029_v1 as bridge
from experiments.spectral_reachability import p2_h15_sra_host_candidate_20261029_v1 as single

core=att.core
native=att.native
transport=single.transport
NAME='p2_h15_host_lifecycle_20261029_v1'
need=core.require


class GuardedNative:
    """Production callback binding; called only by the still-disabled HOST entry."""
    kind='HOST_CUDA'
    def __init__(self,session,root,sink):
        self.session=session; self.root=root
        self.backend=native.NativeTree(session.model,session.device,session.mapped['roots'][root]['task'],sink)
    def check(self,ledger):
        self.session.check()
        need(self.backend.task==self.session.mapped['roots'][self.root]['task'] and
             type(ledger) is core.Ledger and ledger.caps==single.caps(),'root and ledger binding')
    def signal(self,req,ledger):
        self.check(ledger); value=self.backend.signal(req,ledger); self.check(ledger); return value
    def replay(self,parent,actions,segmented,ledger):
        self.check(ledger); value=self.backend.replay(parent,actions,segmented,ledger); self.check(ledger); return value


def _slot(path,task,factory,guard,kind):
    path.mkdir(); journal=transport.Journal(path/'events.jsonl',single.MAX_JOURNAL)
    class Sink:
        def event(self,e): journal.emit('event',e)
    sink=Sink(); ledger=core.Ledger(single.caps(),sink=sink.event)
    report=dict(schema=NAME+'.slot',backend=kind,task_sha256=core.digest(task),status='failed',
                error=None,result=None,resources=None)
    def call(method,req,fn):
        guard(); sink.event(dict(kind='native_call_attempt',method=method,task=task,request=req))
        try:
            value=fn(); guard()
            sink.event(dict(kind='native_call_return',result=value)); return value
        except Exception as exc:
            sink.event(dict(kind='native_call_failure',error=dict(type=type(exc).__name__,message=str(exc))))
            raise
    try:
        backend=factory(task,sink); need(backend.kind==kind,'backend identity')
        ledger.scope=dict(stage='signal',index=0)
        req=core.request(task,dict(context=task['root_context']),operation='compile_finite_horizon_J',horizon=3)
        signal=call('signal',req,lambda:backend.signal(req,ledger))
        bank=single.prep.spectral_bank(signal['jacobian'],signal['nominal'],task['goal'])
        sink.event(dict(kind='bank',value=bank)); scored=[]
        for i,actions in enumerate(bank['raw15']):
            ledger.scope=dict(stage='candidate',index=i)
            req=dict(parent_context=task['root_context'],actions=actions,segmented=False)
            out=call('replay',req,lambda:backend.replay(task['root_context'],actions,False,ledger))
            core.checked_path(task,task['root_context'],actions,out)
            delta=np.asarray(out[-1]['latent_history'][-1],np.float64)-np.asarray(task['goal'],np.float64)
            score=float(delta@delta/192);need(np.isfinite(score),'finite score')
            scored.append(dict(index=i,score=score,path=out));sink.event(dict(kind='candidate',index=i,score=score))
        winner=min(range(40),key=lambda i:(scored[i]['score'],i)); paths=[];actions=bank['raw15'][winner]
        for i in (0,1):
            ledger.scope=dict(stage='verification',index=i)
            req=dict(parent_context=task['root_context'],actions=actions,segmented=bool(i))
            paths.append(call('replay',req,lambda:backend.replay(task['root_context'],actions,bool(i),ledger)))
        need(paths[0]==paths[1]==scored[winner]['path'],'two exact replays')
        need(ledger.attempted==ledger.completed==single.caps(),'measured budget')
        report.update(status='complete',result=dict(bank=bank,candidates=scored,winner=winner,replays=paths))
    except Exception as exc:
        report['error']=dict(type=type(exc).__name__,message=str(exc))
    finally:
        report['resources']=ledger.receipt();journal.close()
    report['journal_sha256']=bridge.sha(path/'events.jsonl');single._put(path/'report.json',report)
    return report,bridge.sha(path/'report.json')


def _audit_slot(path,expected,task,kind):
    report=bridge.source.prep.read(path/'report.json',expected)
    need(report['schema']==NAME+'.slot' and report['backend']==kind and report['task_sha256']==core.digest(task), 'slot identity')
    need(bridge.sha(path/'events.jsonl')==report['journal_sha256'],'slot journal')
    events=[e['payload'] for e in transport.read_journal(path/'events.jsonl')]
    single.accounting.audit_resources(report['resources'],[e for e in events if e['kind'].startswith('resource_')])
    need(report['resources']['caps']==single.caps(),'budget caps')
    if report['status']=='failed':
        need(report['error'] is not None and report['result'] is None,'failure no candidate release')
        return report
    need(report['status']=='complete' and report['error'] is None and report['resources']['attempted']==report['resources']['completed']==single.caps(),'complete measured budget')
    calls=[];active=None;bank_event=None;candidate_events=[]
    for e in events:
        if e['kind']=='native_call_attempt':
            need(active is None and e['task']==task,'native call ordering');active=[e]
        elif active is not None:
            active.append(e)
            if e['kind'] in ('native_call_return','native_call_failure'):
                need(native.audit_native(active) is True,'native IO audit');calls.append(active);active=None
        elif e['kind']=='bank':need(bank_event is None,'one bank');bank_event=e['value']
        elif e['kind']=='candidate':candidate_events.append(e)
        else:raise ValueError('unexpected event')
    need(active is None and len(calls)==43 and len(candidate_events)==40,'all paid calls')
    req=core.request(task,dict(context=task['root_context']),operation='compile_finite_horizon_J',horizon=3)
    need(calls[0][0]['method']=='signal' and calls[0][0]['request']==req,'signal binding')
    signal=calls[0][-1]['result'];bank=single.prep.spectral_bank(signal['jacobian'],signal['nominal'],task['goal'])
    need(core.canonical(bank)==core.canonical(bank_event)==core.canonical(report['result']['bank']),'bank recompute')
    scored=[]
    for i,call in enumerate(calls[1:41]):
        req=dict(parent_context=task['root_context'],actions=bank['raw15'][i],segmented=False)
        need(call[0]['method']=='replay' and call[0]['request']==req,'candidate request')
        out=call[-1]['result'];core.checked_path(task,task['root_context'],req['actions'],out)
        d=np.asarray(out[-1]['latent_history'][-1],np.float64)-np.asarray(task['goal'],np.float64)
        score=float(d@d/192);need(np.isfinite(score),'finite score')
        scored.append(dict(index=i,score=score,path=out))
        need(candidate_events[i]==dict(kind='candidate',index=i,score=score),'candidate event')
    winner=min(range(40),key=lambda i:(scored[i]['score'],i));result=report['result']
    need(result['candidates']==scored and result['winner']==winner,'winner rebuild')
    for i,call in enumerate(calls[41:]):
        need(call[0]['method']=='replay' and call[0]['request']==dict(parent_context=task['root_context'],actions=bank['raw15'][winner],segmented=bool(i)) and
             call[-1]['result']==result['replays'][i]==scored[winner]['path'],'replay binding')
    return report


def run_host(directory,session):
    return _run(directory,session,lambda root,task,sink:GuardedNative(session,root,sink),'HOST_CUDA')


def run_fixture(directory,session,factory):
    need(session.fixture is True and not session.closed,'fixture session only')
    return _run(directory,session,lambda root,task,sink:factory(task,sink),'RAM_FIXTURE')


def _run(directory,session,factory,kind):
    need(kind=='RAM_FIXTURE' and session.fixture is True, 'HOST engine dispatch not authorized')
    mapped=bridge.build_bridge();need(session.mapped==mapped,'exact bridge')
    path=Path(directory);need(not path.is_symlink(),'exclusive path');path.mkdir()
    journal=transport.Journal(path/'lifecycle.jsonl',1024*1024)
    transitions=[];slots=[];attestation=None;attestation_error=None
    try:
        journal.emit('bridge',dict(sha256=core.digest(mapped),backend=kind))
        before=dict(model=session.model_before,precision=session.precision_before,
                    sources=session.source_before,fixture_only=session.fixture)
        journal.emit('attestation_before',before)
        for root in (0,1):
            task=mapped['roots'][root]['task'];transitions.append(('start',root));bridge.source.lifecycle(transitions)
            journal.emit('start',dict(root=root,task_sha256=core.digest(task)))
            try:
                r,h=_slot(path/('root'+str(root)),task,lambda t,s:factory(root,t,s),session.check,kind)
                _audit_slot(path/('root'+str(root)),h,task,kind)
                action='seal' if r['status']=='complete' else 'fail'
                slot=dict(root=root,status=r['status'],report_sha256=h)
            except Exception as exc:
                action='fail';slot=dict(root=root,status='outer_failure',report_sha256=None,
                                       error=dict(type=type(exc).__name__,message=str(exc)[:2048]))
            transitions.append((action,root));slots.append(slot);journal.emit(action,slot)
            if action=='fail':break
        try:attestation=session.close()
        except Exception as exc:attestation_error=dict(type=type(exc).__name__,message=str(exc)[:2048])
        journal.emit('attestation',dict(value=attestation,error=attestation_error))
    finally:journal.close()
    report=dict(schema=NAME+'.report',bridge_sha256=core.digest(mapped),backend=kind,slots=slots,
                root_status=bridge.source.lifecycle(transitions),fixed_denominator=2,
                attestation_before=before,attestation=attestation,attestation_error=attestation_error,
                lifecycle_sha256=bridge.sha(path/'lifecycle.jsonl'),host_authorized=False,physical_release=False)
    single._put(path/'report.json',report)
    return dict(report_sha256=bridge.sha(path/'report.json'))


def consume(directory,expected,*,permit_fixture=False):
    path=Path(directory);need(path.is_dir() and not path.is_symlink(),'directory')
    report=bridge.source.prep.read(path/'report.json',expected);mapped=bridge.build_bridge()
    kind=report['backend']
    need(report['schema']==NAME+'.report' and report['bridge_sha256']==core.digest(mapped) and
         kind in ('RAM_FIXTURE','HOST_CUDA') and (kind!='RAM_FIXTURE' or permit_fixture is True) and report['host_authorized'] is False and
         report['physical_release'] is False and report['fixed_denominator']==2,'fixture-only consumer')
    need(bridge.sha(path/'lifecycle.jsonl')==report['lifecycle_sha256'],'lifecycle SHA')
    events=transport.read_journal(path/'lifecycle.jsonl')
    before=report['attestation_before']
    need(before['fixture_only'] is (kind=='RAM_FIXTURE') and before['precision']==native.precision_contract(),'before snapshot')
    expected_events=[('bridge',dict(sha256=core.digest(mapped),backend=report['backend'])),('attestation_before',before)]
    transitions=[];resources=[]
    for i,slot in enumerate(report['slots']):
        need(i<2 and slot['root']==i,'slot order');task=mapped['roots'][i]['task']
        transitions.append(('start',i));expected_events.append(('start',dict(root=i,task_sha256=core.digest(task))))
        if slot['status'] in ('complete','failed'):
            r=_audit_slot(path/('root'+str(i)),slot['report_sha256'],task,report['backend'])
            need(r['status']==slot['status'],'slot status');resources.append(r['resources'])
            action='seal' if r['status']=='complete' else 'fail'
        else:
            need(slot['status']=='outer_failure' and slot['report_sha256'] is None and isinstance(slot.get('error'),dict),'outer failure')
            action='fail'
        need(action!='fail' or i==len(report['slots'])-1,'stop after failure')
        transitions.append((action,i));expected_events.append((action,slot))
    need(report['slots'] and (len(report['slots'])==2 or transitions[-1][0]=='fail'),'fixed denominator')
    need(bridge.source.lifecycle(transitions)==report['root_status'],'lifecycle state')
    a=report['attestation'];err=report['attestation_error']
    if a is not None:
        need(err is None and a['fixture_only'] is (kind=='RAM_FIXTURE') and a['host_authorized'] is False and a['physical_release'] is False and
             before['model']==a['model_before'] and before['precision']==a['precision_before'] and before['sources']==a['sources_before'] and
             a['model_before']==a['model_after'] and a['precision_before']==a['precision_after']==native.precision_contract() and
             a['sources_before']==a['sources_after'],'attestation snapshots')
        if kind=='HOST_CUDA':
            need(a['model_before']['original_state_dict']==mapped['roots'][0]['task']['binding']['model_sha256'] and
                 a['sources_before']==att.verify_sources(att.historical_inventory()),'host model and source evidence')
    else:need(isinstance(err,dict),'missing final attestation')
    expected_events.append(('attestation',dict(value=a,error=err)))
    need([(e['kind'],e['payload']) for e in events]==expected_events,'lifecycle journal')
    return dict(root_status=report['root_status'],attestation=a,attestation_error=err,resources=resources,
                fixture_only=(kind=='RAM_FIXTURE'),host_authorized=False,physical_release=False,
                host_certified=False)
