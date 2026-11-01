"""Fixed H15 projected gradient slot and saved native transcript audit; no model loader."""
from pathlib import Path
import numpy as np
from experiments.spectral_reachability import p2_h15_cem_matched_host_20261029_v1 as cem
from experiments.spectral_reachability import p2_h15_gradient_solver_20261029_v1 as gn
from experiments.spectral_reachability import p2_h15_gn_direction_guard_20261029_v1 as fd

life,bridge,core,need=cem.life,cem.bridge,cem.core,cem.need
NAME='p2_h15_gradient_matched_slot_20261029_v1'


def caps():
    value=life.single.accounting.predict_axes(117)
    value.update(jacobian_API=4,VJP_directions=768,grad_API=768,raw_backward_callbacks=768)
    return value


def audit_ad(call):
    start=call[0]; body=call[1:-1]; result=call[-1]['result']; task=start['task']
    need(start['method']=='current_ad' and start['request']=={'logits':result['logits']},'AD request')
    z=np.asarray(result['logits'],np.float64);raw=gn.raw(z)
    need(result['raw']==raw.tolist(),'AD raw mapping')
    j=core.finite(result['raw_jacobian'],(192,30),True)
    n=core.finite(result['nominal'],(192,),True)
    need(np.array_equal(np.asarray(result['logit_jacobian'],np.float64),j.astype(np.float64)*(1-np.tanh(z)**2)[None,:]),'logit chain')
    origins=[x for x in body if x['kind']=='origin']
    need(len(origins)==1,'one current AD origin')
    blob=life.native.signal.unblob
    origin=origins[0]['arrays']
    need(set(origin)=={'latent','history','raw_flat','mean','std'} and
         np.array_equal(blob(origin['raw_flat']),raw.reshape(30)) and
         np.array_equal(blob(origin['latent']),np.asarray(task['root_context']['latent_history'],np.float32)[None]) and
         np.array_equal(blob(origin['history']),np.asarray(task['root_context']['action_history'],np.float32)[None]) and
         np.array_equal(blob(origin['mean']),np.asarray(task['action_mean'],np.float32)) and
         np.array_equal(blob(origin['std']),np.asarray(task['action_std'],np.float32)),'current AD origin')
    # Reuse the original program auditor after translating packet fields; its
    # signal branch is deliberately hard-coded to zero-action origins.
    translated=[]
    for event in body:
        if event['kind']=='packet':
            step=event['step'];a=event['arrays']
            need(set(a)=={'raw','token','outgoing'},'current AD packet keys')
            pin=[x for x in body if x['kind']=='io_input' and x.get('step')==step and 'predict:latent' in x['arrays']]
            need(len(pin)==1,'current AD predict input')
            history=blob(a['outgoing'])[:,:2]
            event=dict(event,kind='program_packet',arrays=dict(a,latent=pin[0]['arrays']['predict:latent'],
                history=life.native.signal.blob(history),mean=origin['mean'],std=origin['std']))
        elif event['kind']=='boundary':event=dict(event,kind='program_boundary')
        translated.append(event)
    terminal=life.native.live.audit_trajectory(task,translated,'current_AD',3,raw,None,program=True)
    need(np.array_equal(n,terminal),'AD trajectory')
    def vals(k):return [blob(x['arrays']['value']) for x in body if x['kind']==k]
    for kind,v in (('AD_terminal',n),('jacobian',j)):
        x=vals(kind);need(len(x)==1 and np.array_equal(x[0],v),'AD result buffer')
    need([x['direction'] for x in body if x['kind']=='direction']==list(range(192)),'AD directions')
    ins=[x for x in body if x['kind']=='io_input' and 'grad:basis' in x['arrays']]
    outs=[x for x in body if x['kind']=='io_return' and 'grad:output' in x['arrays']]
    backs=vals('raw_backward');need(len(ins)==len(outs)==len(backs)==192,'AD grad witnesses')
    eye=np.eye(192,dtype=np.float32)
    for i,(a,b,c) in enumerate(zip(ins,outs,backs,strict=True)):
        need(np.array_equal(blob(a['arrays']['grad:basis']),eye[i]) and
             np.array_equal(blob(a['arrays']['grad:output']),n) and
             np.array_equal(blob(a['arrays']['grad:raw']),raw.reshape(30)) and
             np.array_equal(blob(b['arrays']['grad:output']),j[i]) and np.array_equal(c,j[i]),'AD row binding')
    allowed=life.native.RESOURCE|{'origin','packet','boundary','AD_terminal','jacobian','raw_backward','io_input','io_return','direction'}
    need(all(x['kind'] in allowed for x in body),'AD event whitelist')
    expected=life.single.accounting.predict_axes(3)
    expected.update(jacobian_API=1,VJP_directions=192,grad_API=192,raw_backward_callbacks=192)
    actual=dict.fromkeys(core.AXES,0)
    for x in body:
        if x['kind']=='resource_attempt':
            for k in actual:actual[k]+=x['entry']['delta'][k]
    need(actual==expected,'AD resource axes')


def execute(task, invoke):
    """Identical algorithm for actual callbacks and saved-only replay."""
    context=task['root_context'];goal=task['goal']; counter=[0,0]
    def replay(raw,segmented=False):
        actions=np.asarray(raw,np.float32).tolist()
        req=dict(parent_context=context,actions=actions,segmented=segmented)
        out=invoke('replay',req)
        core.checked_path(task,context,actions,out)
        return out
    def evaluate(raw):
        counter[0]+=1
        return replay(raw)[-1]['latent_history'][-1]
    def linearize(z):
        counter[1]+=1
        result=invoke('current_ad',dict(logits=z.tolist()))
        need(result['logits']==z.tolist(),'same incumbent AD')
        fd.verify(result,evaluate)
        return result['nominal'],result['logit_jacobian']
    result=gn.solve(goal,evaluate,linearize)
    need(counter==[33,4],'fixed candidate and FD calls')
    paths=[replay(result['actions'],False),replay(result['actions'],True)]
    need(paths[0]==paths[1],'final whole segmented replay')
    d=np.asarray(paths[0][-1]['latent_history'][-1],np.float64)-np.asarray(goal,np.float64)
    need(float(np.mean(d*d))==result['score'],'final replay incumbent score')
    return dict(solution=result,replays=paths)


def run_slot(path,task,factory,guard,kind,root):
    task=core.validate_task(task);path=Path(path);need(root in (0,1) and kind in ('HOST_CUDA','RAM_FIXTURE'),'slot identity')
    path.mkdir();journal=life.transport.Journal(path/'events.jsonl',life.single.MAX_JOURNAL)
    class Sink:
        def event(self,e):journal.emit('event',e)
    sink=Sink();ledger=core.Ledger(caps(),sink=sink.event)
    report=dict(schema=NAME+'.slot',backend=kind,root=root,task_sha256=core.digest(task),status='failed',error=None,result=None,resources=None)
    try:
        backend=factory(sink);need(backend.kind==kind,'backend kind')
        index=0
        def invoke(method,req):
            nonlocal index
            ledger.scope=dict(call=index,method=method);index+=1;guard()
            sink.event(dict(kind='native_call_attempt',method=method,task=task,request=req))
            try:
                if method=='replay':out=backend.replay(req['parent_context'],req['actions'],req['segmented'],ledger)
                else:out=backend.linearize(req['logits'],ledger)
                guard();sink.event(dict(kind='native_call_return',result=out));return out
            except Exception as exc:
                sink.event(dict(kind='native_call_failure',error=dict(type=type(exc).__name__,message=str(exc))));raise
        result=execute(task,invoke)
        need(ledger.attempted==ledger.completed==caps(),'exact paid budget')
        report.update(status='complete',result=result)
    except Exception as exc:report['error']=dict(type=type(exc).__name__,message=str(exc)[:2048])
    finally:report['resources']=ledger.receipt();journal.close()
    report['journal_sha256']=bridge.sha(path/'events.jsonl');life.single._put(path/'report.json',report)
    return report,bridge.sha(path/'report.json')


def audit_slot(path,sha,task,kind,root):
    path=Path(path);task=core.validate_task(task);r=bridge.source.prep.read(path/'report.json',sha)
    need(r['schema']==NAME+'.slot' and r['status']=='complete' and r['error'] is None and r['backend']==kind and r['root']==root and r['task_sha256']==core.digest(task),'slot identity')
    need(bridge.sha(path/'events.jsonl')==r['journal_sha256'],'slot journal SHA')
    events=[x['payload'] for x in life.transport.read_journal(path/'events.jsonl')]
    life.native.resource_census(r['resources'],events)
    need(r['resources']['caps']==caps() and r['resources']['attempted']==r['resources']['completed']==caps(),'exact paid budget')
    calls=[];active=None
    for event in events:
        if event['kind']=='native_call_attempt':need(active is None and event['task']==task,'call order');active=[event]
        elif active is not None:
            active.append(event)
            if event['kind']=='native_call_return':
                if active[0]['method']=='replay':need(life.native.audit_native(active),'replay native audit')
                else:audit_ad(active)
                calls.append(active);active=None
        else:raise ValueError('event outside native call')
    need(active is None and len(calls)==39,'complete calls')
    index=0
    def invoke(method,req):
        nonlocal index
        call=calls[index];index+=1
        need(call[0]['method']==method and call[0]['request']==req,'deterministic request binding')
        return call[-1]['result']
    result=execute(task,invoke)
    need(index==len(calls) and result==r['result'],'full algorithm reconstruction')
    return r
