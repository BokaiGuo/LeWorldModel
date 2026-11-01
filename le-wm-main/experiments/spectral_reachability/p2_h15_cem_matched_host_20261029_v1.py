"""Matched H15 ball-domain CEM slot; native callback and saved audit."""
from pathlib import Path
import argparse
import json
import os
import resource
import shutil
import time
import numpy as np
from experiments.spectral_reachability import p2_h15_host_lifecycle_20261029_v1 as life
from experiments.spectral_reachability import p2_h15_bridge_lifecycle_20261029_v1 as bridge
from experiments.spectral_reachability import p2_h15_controlled_host_20261029_v1 as host

NAME='p2_h15_cem_matched_host_20261029_v1'
RADIUS=0.75*np.sqrt(15)
STD_FLOOR=0.05
core=life.core
need=life.need


def draw(mean,std,seed):
    mean=np.asarray(mean,np.float64);std=np.asarray(std,np.float64)
    need(mean.shape==std.shape==(30,) and np.isfinite(mean).all() and np.isfinite(std).all() and np.all(std>=STD_FLOOR),'distribution')
    rng=np.random.Generator(np.random.PCG64(seed))
    z=mean+std*rng.standard_normal((7,30))
    norm=np.linalg.norm(z,axis=1,keepdims=True)
    return z*np.minimum(1,RADIUS/np.maximum(norm,1e-300))


def raw(z):
    z=np.asarray(z,np.float64)
    need(z.ndim==2 and z.shape[1]==30 and np.isfinite(z).all() and np.all(np.linalg.norm(z,axis=1)<=RADIUS+1e-12),'logit ball')
    return np.tanh(z).astype(np.float32).reshape(-1,15,2)


def update(samples,scores):
    samples=np.asarray(samples,np.float64); scores=np.asarray(scores,np.float64)
    need(samples.shape==(7,30) and scores.shape==(7,) and np.isfinite(samples).all() and np.isfinite(scores).all(),'finite generation')
    elite=np.argsort(scores,kind='stable')[:2]; selected=samples[elite]
    return selected.mean(axis=0),np.maximum(selected.std(axis=0,ddof=0),STD_FLOOR)


def caps(): return life.single.accounting.predict_axes(132)


def run_slot(path,task,factory,guard,kind,root):
    task=core.validate_task(task); path=Path(path);need(root in (0,1) and kind in ('RAM_FIXTURE','HOST_CUDA'),'slot identity')
    path.mkdir();journal=life.transport.Journal(path/'events.jsonl',life.single.MAX_JOURNAL)
    class Sink:
        def event(self,e): journal.emit('event',e)
    sink=Sink();ledger=core.Ledger(caps(),sink=sink.event)
    report=dict(schema=NAME+'.slot',backend=kind,root=root,task_sha256=core.digest(task),status='failed',error=None,result=None,resources=None)
    def call(actions,segmented,index):
        guard(); req=dict(parent_context=task['root_context'],actions=actions,segmented=segmented)
        sink.event(dict(kind='native_call_attempt',method='replay',task=task,request=req))
        try:
            out=backend.replay(task['root_context'],actions,segmented,ledger);guard()
            core.checked_path(task,task['root_context'],actions,out)
            sink.event(dict(kind='native_call_return',result=out));return out
        except Exception as exc:
            sink.event(dict(kind='native_call_failure',error=dict(type=type(exc).__name__,message=str(exc))));raise
    try:
        backend=factory(sink);mean=np.zeros(30);std=np.ones(30); generations=[];best=None
        for gen in range(6):
            z=draw(mean,std,123+root*6+gen); actions=raw(z);scores=[]
            for i in range(7):
                ix=gen*7+i;ledger.scope=dict(stage='candidate',index=ix)
                path_out=call(actions[i].tolist(),False,ix)
                d=np.asarray(path_out[-1]['latent_history'][-1],np.float64)-np.asarray(task['goal'],np.float64)
                score=float(d@d/192);need(np.isfinite(score),'finite score');scores.append(score)
                if best is None or score<best['score']:best=dict(index=ix,score=score,actions=actions[i].tolist(),path=path_out)
            generations.append(dict(logits=z.tolist(),scores=scores));mean,std=update(z,scores)
            sink.event(dict(kind='generation',index=gen,logits=z.tolist(),scores=scores))
        paths=[]
        for i in (0,1):
            ledger.scope=dict(stage='verification',index=i)
            paths.append(call(best['actions'],bool(i),i))
        need(paths[0]==paths[1]==best['path'],'exact paid replays')
        need(ledger.attempted==ledger.completed==caps(),'matched exact budget')
        report.update(status='complete',result=dict(generations=generations,winner=best,replays=paths))
    except Exception as exc:report['error']=dict(type=type(exc).__name__,message=str(exc)[:2048])
    finally:report['resources']=ledger.receipt();journal.close()
    report['journal_sha256']=bridge.sha(path/'events.jsonl');life.single._put(path/'report.json',report)
    return report,bridge.sha(path/'report.json')


def audit_slot(path,sha,task,kind,root):
    path=Path(path);task=core.validate_task(task);r=bridge.source.prep.read(path/'report.json',sha)
    need(r['schema']==NAME+'.slot' and r['status']=='complete' and r['error'] is None and r['backend']==kind and r['root']==root and r['task_sha256']==core.digest(task),'complete slot identity')
    need(bridge.sha(path/'events.jsonl')==r['journal_sha256'],'journal SHA')
    events=[x['payload'] for x in life.transport.read_journal(path/'events.jsonl')]
    life.single.accounting.audit_resources(r['resources'],[e for e in events if e['kind'].startswith('resource_')])
    need(r['resources']['caps']==caps() and r['resources']['attempted']==r['resources']['completed']==caps(),'paid budget')
    calls=[];active=None;gens=[]
    for e in events:
        if e['kind']=='native_call_attempt':need(active is None and e['task']==task,'call ordering');active=[e]
        elif active is not None:
            active.append(e)
            if e['kind'] in ('native_call_return','native_call_failure'):
                need(life.native.audit_native(active) is True,'native IO');calls.append(active);active=None
        elif e['kind']=='generation':gens.append(e)
        else:raise ValueError('unexpected event')
    need(active is None and len(calls)==44 and len(gens)==6,'call roster')
    mean=np.zeros(30);std=np.ones(30);best=None
    for gen in range(6):
        z=draw(mean,std,123+root*6+gen);actions=raw(z);scores=[]
        for i in range(7):
            ix=gen*7+i;c=calls[ix];req=dict(parent_context=task['root_context'],actions=actions[i].tolist(),segmented=False)
            need(c[0]['method']=='replay' and c[0]['request']==req,'candidate binding');out=c[-1]['result'];core.checked_path(task,task['root_context'],req['actions'],out)
            d=np.asarray(out[-1]['latent_history'][-1],np.float64)-np.asarray(task['goal'],np.float64);score=float(d@d/192);scores.append(score)
            if best is None or score<best['score']:best=dict(index=ix,score=score,actions=req['actions'],path=out)
        need(gens[gen]==dict(kind='generation',index=gen,logits=z.tolist(),scores=scores) and r['result']['generations'][gen]==dict(logits=z.tolist(),scores=scores),'generation rebuild')
        mean,std=update(z,scores)
    need(r['result']['winner']==best,'winner rebuild')
    for i in (0,1):
        call=calls[42+i];need(call[0]['request']==dict(parent_context=task['root_context'],actions=best['actions'],segmented=bool(i)) and call[-1]['result']==best['path']==r['result']['replays'][i],'paid replay')
    return r


AUTH='MAIN_APPROVED_P2_H15_MATCHED_CEM_HOST_CUDA'


def prepare_value():
    h=host.build(); mapped=bridge.build_bridge()
    return dict(schema=NAME+'.prepare',host_contract=h,bridge_sha256=core.digest(mapped),
        source_sha256=bridge.sha(__file__),test_sha256=bridge.sha(bridge.source.prep.ROOT/'tests'/('test_'+NAME+'.py')),
        domain=dict(logit_dimension=30,radius=float(RADIUS),raw='float64 numpy tanh then float32; original H15 mapping',
                    task_goal='identity192 mean squared terminal',horizon=15),
        algorithm=dict(iterations=6,population=7,elite=2,std_floor=STD_FLOOR,seed_rule='123 + root*6 + generation',
                       delivery='best evaluated sample, stable first tie; paid whole/segmented replay'),
        budget_per_root=caps(),h15_budget_per_root=life.single.caps(),fixed_denominator=2,
        caveat='matched predictor samples only; H15 additionally uses VJP; known observed roots, not preregistered')


def verify_prepare(path,sha):
    value=bridge.source.prep.read(Path(path),sha)
    need(value==prepare_value(),'prepared source/protocol mismatch');return value


class Native:
    kind='HOST_CUDA'
    def __init__(self,session,root,sink):
        self.session=session;self.root=root
        self.backend=life.native.NativeTree(session.model,session.device,session.mapped['roots'][root]['task'],sink)
    def replay(self,parent,actions,segmented,ledger):
        self.session.check();need(self.backend.task==self.session.mapped['roots'][self.root]['task'] and
            type(ledger) is core.Ledger and ledger.caps==caps(),'native task/ledger')
        result=self.backend.replay(parent,actions,segmented,ledger);self.session.check();return result


def run(prepared,sha,output,authorization):
    started=time.monotonic();p=verify_prepare(prepared,sha);need(authorization==AUTH,'explicit CUDA authorization')
    output=Path(output);need(output.is_absolute() and output.parent==bridge.source.prep.ROOT/'results' and
        output.name.startswith(NAME+'_') and not output.exists() and not output.is_symlink(),'exclusive output')
    need(shutil.disk_usage(output.parent).free>=host.CAP+host.FLOOR,'disk allowance')
    _,hard=resource.getrlimit(resource.RLIMIT_FSIZE);need(hard==resource.RLIM_INFINITY or hard>=host.FILE_CAP,'file cap available')
    resource.setrlimit(resource.RLIMIT_FSIZE,(host.FILE_CAP,hard));output.mkdir()
    state=dict(schema=NAME+'.report',status='failed_partial',prepared_sha256=sha,slots=[],root_status=['not_started','not_started'],
               errors=[],attestation=None,physical_release=False,scientific_admission=False)
    journal=None;store=None
    try:
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import SDPBackend,sdpa_kernel
        from experiments.gpu_preflight import require_cuda_allocation
        state['host']=require_cuda_allocation('cuda:0');device=torch.device('cuda:0');torch.cuda.reset_peak_memory_stats(device)
        guard=host.ResourceGuard(output,time.monotonic,lambda:(torch.cuda.max_memory_allocated(device),torch.cuda.max_memory_reserved(device)),
                                  lambda:shutil.disk_usage(output).free,started)
        guard.check();store=host.LoaderStore(output,guard)
        historical=bridge.source.prep.read(bridge.source.HOST,bridge.source.HOST_SHA)
        model,_=host.historical.old.loader.loaded_model(historical,torch,device,store,bridge.source.HOST_SHA)
        store.close();store=None
        state['loader_sha256']=bridge.sha(output/'loader.json');state['loader_events_sha256']=bridge.sha(output/'loader_events.jsonl')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        with sdpa_kernel(SDPBackend.MATH),torch.autocast(device_type='cuda',enabled=False),torch.autocast(device_type='cpu',enabled=False):
            session=host.GuardedSession(bridge.build_bridge(),model,device,torch,host.att.historical_inventory(),host.att.model_hash(model),guard=guard)
            need(bridge.sha(__file__)==p['source_sha256'],'own source changed');host.check_extra_sources(p['host_contract']['extra_sources'])
            state['attestation_before']=dict(model=session.model_before,precision=session.precision_before,sources=session.source_before,fixture_only=False)
            journal=life.transport.Journal(output/'lifecycle.jsonl',1024*1024);transitions=[]
            journal.emit('attestation_before',state['attestation_before'])
            def check():
                session.check();need(bridge.sha(__file__)==p['source_sha256'],'own source changed')
                host.check_extra_sources(p['host_contract']['extra_sources'])
            for root in (0,1):
                task=session.mapped['roots'][root]['task'];transitions.append(('start',root));journal.emit('start',dict(root=root,task_sha256=core.digest(task)))
                try:
                    slot,digest=run_slot(output/('root'+str(root)),task,lambda sink:Native(session,root,sink),check,'HOST_CUDA',root)
                    if slot['status']=='complete':audit_slot(output/('root'+str(root)),digest,task,'HOST_CUDA',root)
                    action='seal' if slot['status']=='complete' else 'fail'
                    item=dict(root=root,status=slot['status'],report_sha256=digest)
                except Exception as exc:
                    action='fail';item=dict(root=root,status='outer_failure',report_sha256=None,error=dict(type=type(exc).__name__,message=str(exc)[:2048]))
                transitions.append((action,root));state['slots'].append(item);state['root_status']=bridge.source.lifecycle(transitions);journal.emit(action,item)
                if action=='fail':break
            state['attestation']=session.close();host.check_extra_sources(p['host_contract']['extra_sources'])
            need(bridge.sha(__file__)==p['source_sha256'],'own source changed');guard.check()
            journal.emit('attestation',state['attestation']);state['peak']=dict(allocated=torch.cuda.max_memory_allocated(device),reserved=torch.cuda.max_memory_reserved(device))
            if state['root_status']==['sealed','sealed']:state['status']='recorded_not_admitted'
    except BaseException as exc:state['errors'].append(dict(type=type(exc).__name__,message=str(exc)[:2048]))
    finally:
        if store is not None:store.close()
        if journal is not None:journal.close();state['lifecycle_sha256']=bridge.sha(output/'lifecycle.jsonl')
        state['elapsed_seconds']=time.monotonic()-started;life.single._put(output/'report.json',state)
    return dict(status=state['status'],report_sha256=bridge.sha(output/'report.json'))


def consume(directory,report_sha,prepared,prepare_sha):
    p=verify_prepare(prepared,prepare_sha);path=Path(directory);r=bridge.source.prep.read(path/'report.json',report_sha)
    need(r['schema']==NAME+'.report' and r['prepared_sha256']==prepare_sha and r['status']=='recorded_not_admitted' and
         r['root_status']==['sealed','sealed'] and not r['errors'] and r['physical_release'] is False and r['scientific_admission'] is False,'complete host report')
    old=bridge.source.prep.read(bridge.source.HOST,bridge.source.HOST_SHA)
    loader=bridge.source.prep.read(path/'loader.json',r['loader_sha256']);host.historical.old.loader.validate_loader(loader,old,bridge.source.HOST_SHA)
    need(bridge.sha(path/'loader_events.jsonl')==r['loader_events_sha256'] and bridge.sha(path/'lifecycle.jsonl')==r['lifecycle_sha256'],'journal hash')
    life.transport.read_journal(path/'loader_events.jsonl');events=life.transport.read_journal(path/'lifecycle.jsonl')
    mapped=bridge.build_bridge();expect=[('attestation_before',r['attestation_before'])];resources=[]
    need(len(r['slots'])==2,'slot roster')
    for root,slot in enumerate(r['slots']):
        task=mapped['roots'][root]['task'];need(slot['root']==root and slot['status']=='complete','slot order')
        result=audit_slot(path/('root'+str(root)),slot['report_sha256'],task,'HOST_CUDA',root);resources.append(result['resources'])
        expect.extend([('start',dict(root=root,task_sha256=core.digest(task))),('seal',slot)])
    a=r['attestation'];b=r['attestation_before']
    need(a is not None and a['fixture_only'] is False and b['model']==a['model_before']==a['model_after'] and
         a['model_before']['original_state_dict']==old['expected_model_hash'] and
         b['precision']==a['precision_before']==a['precision_after']==host.att.native.precision_contract() and
         b['sources']==a['sources_before']==a['sources_after']==host.att.verify_sources(host.att.historical_inventory()),'attestation')
    expect.append(('attestation',a));need([(e['kind'],e['payload']) for e in events]==expect,'lifecycle')
    need(r['peak']['allocated']<=host.ALLOCATED and r['peak']['reserved']<=host.RESERVED and r['elapsed_seconds']<host.WALL,'runtime limits')
    return dict(schema=NAME+'.consume',status='saved_native_checked_not_admitted',resources=resources,
                limitation='saved audit cannot prove absence of uninstrumented calls or physical outcome',physical_release=False)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=('prepare','verify','run','consume'))
    ap.add_argument('--prepared',type=Path,required=True);ap.add_argument('--sha256');ap.add_argument('--output',type=Path)
    ap.add_argument('--authorization');ap.add_argument('--report-sha256');args=ap.parse_args()
    if args.command=='prepare':
        need(args.prepared.is_absolute() and args.prepared.name.startswith(NAME+'_') and args.sha256 is None,'exclusive prepare')
        life.single._put(args.prepared,prepare_value()); result=dict(sha256=bridge.sha(args.prepared))
    elif args.command=='verify':need(args.sha256 and args.output is None,'verify arguments');verify_prepare(args.prepared,args.sha256);result=dict(verified=True)
    elif args.command=='run':need(args.sha256 and args.output and not args.report_sha256,'run arguments');result=run(args.prepared,args.sha256,args.output,args.authorization)
    else:need(args.sha256 and args.output and args.report_sha256,'consume arguments');result=consume(args.output,args.report_sha256,args.prepared,args.sha256)
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
