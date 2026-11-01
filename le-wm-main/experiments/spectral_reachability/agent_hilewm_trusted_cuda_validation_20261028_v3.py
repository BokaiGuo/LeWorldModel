"""Four original fixed cases, independent source-layout dictionary reference.

MAIN HOST CUDA only; v1/v2/loader/export remain immutable and unreleased.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import math
import os
from pathlib import Path
import re
import shutil
import time
import numpy as np
from experiments.spectral_reachability import agent_hilewm_trusted_cuda_validation_20261028_v2 as prior

old,loader,ROOT,need=prior.old,prior.loader,prior.ROOT,prior.need
TAG='agent_hilewm_trusted_cuda_validation_20261028_v3'
MANIFEST=ROOT/'results/agent_hilewm_trusted_cuda_validation_20261028_v2_prepare_MAIN01/manifest.json'
MANIFEST_SHA='eadbb1d11383a46b02866a1966588b482dccbb45d6684a55c812d81925782c72'
REPORT=ROOT/'results/agent_hilewm_trusted_cuda_validation_20261028_v2_cuda_MAIN01/report.json'
REPORT_SHA='8f19346c99aba5c3cc5dd6efec878356938c24bcb53023444c37791f94ff6771'
RECEIPT=ROOT/'results/agent_hilewm_trusted_cuda_validation_20261028_v2_consume_MAIN01/receipt.json'
RECEIPT_SHA='0c2b3055c91cd01101774511e8dff24693ec46caa3a91a7ef43b495eeba0a4d9'
PINS={
 'experiments/spectral_reachability/agent_hilewm_trusted_cuda_validation_20261028_v2.py':'35836e271ac09e3b393e33f01adf3f39d5c2b828aa0f2734acb11d30d15102f0',
 'tests/test_agent_hilewm_trusted_cuda_validation_20261028_v2.py':'155aa858acc46540c7063e9794b01ed81ea2ad134e76aa5b08a50200109970f5',
 'innerdocs/agent_hilewm_trusted_cuda_validation_20261028_v2.md':'5b92d489e593b8930eea3f30e6c347e75aa918c58410327cd13e2a939055147c'}
CAP,FLOOR,WALL=old.CAP,old.FLOOR,old.WALL


def closure():
    prior.closure()
    for p,h in PINS.items():need(loader.sha(loader.bytes_read(ROOT/p,loader.JSON_CAP))==h,'frozen v2 dependency SHA')
    return {str(p):loader.sha(loader.bytes_read(p,loader.JSON_CAP)) for p in
        (Path(__file__),ROOT/'tests'/('test_'+TAG+'.py'),ROOT/'innerdocs'/(TAG+'.md'))}


def policy():
    p=deepcopy(old.policy());p['schema']=TAG+'.policy'
    p.update(reference_rollout='per-step reshape -> append -> stack -> context slice; independently rolled reference history',
        quantizer='identity authenticated by v2.source_evidence; no imported original classes',
        high_predict_calls_total=22,trace='all22 native/high dictionary inputs and outputs, exact source-layout',
        legacy_v1_released=False,official_parity=False)
    return p


def upstream():
    # The complete actual v2 strict is rebuilt, never a status-only projection.
    accepted=prior.consume(MANIFEST,MANIFEST_SHA,REPORT,REPORT_SHA)
    need(accepted==loader.json_read(RECEIPT,RECEIPT_SHA),'entire persisted v2 actual receipt exact')
    evidence=prior.source_evidence();m=loader.json_read(MANIFEST,MANIFEST_SHA)
    need(m['source_evidence']==evidence and accepted['report_sha256']==REPORT_SHA
        and accepted['status']=='all_saved_diagnostic_rows_verified_not_release','actual v2 source evidence/strict')
    return dict(manifest_path=str(MANIFEST),manifest_sha256=MANIFEST_SHA,report_path=str(REPORT),report_sha256=REPORT_SHA,
        receipt_path=str(RECEIPT),receipt_sha256=RECEIPT_SHA,
        full_strict_rebuilt_here=True,strict_receipt=accepted,source_evidence=evidence),old.verify(prior.PREPARE,prior.PREPARE_SHA)


def build():
    own=closure();authority,parent=upstream()
    return dict(schema=TAG+'.prepare',policy=policy(),own_sha256=own,dependency_sha256=PINS,upstream=authority,
        parent_prepare_path=str(prior.PREPARE),parent_prepare_sha256=prior.PREPARE_SHA,
        export_report=parent['export_report'],export_report_sha256=parent['export_report_sha256'],
        export_receipt=parent['export_receipt'],export_receipt_sha256=parent['export_receipt_sha256'],ir_sha256=loader.IR_SHA,
        legacy_v1_released=False,official_parity=False,baseline_runnable=False,scientific_admission=False)


def scoped(path):
    p=Path(path).absolute();need(p.parent==ROOT/'results' and p.resolve()==p
        and re.fullmatch(re.escape(TAG)+r'_[A-Za-z0-9_-]+',p.name),'exclusive v3 output directory');return p


def prepare(output,*,main_authorized=False):
    need(main_authorized is True,'MAIN authorization before saved prepare I/O')
    output=scoped(output);need(not output.exists(),'new exclusive prepare')
    m=build();output.mkdir(mode=0o700);h=old.write_new(output/'manifest.json',old.encoded(m))
    return dict(path=str(output/'manifest.json'),sha256=h)


def verify(path,h):
    m=loader.json_read(path,h);need(old.encoded(m)==old.encoded(build()),'entire v3 manifest/v2 strict/source evidence rebuilt');return m


def reference_rollout(t,state,graph,z,actions,meter=None,*,predict=None):
    """Own independent source transcription; NEVER calls owned rollout/high.

    Source-verified quantize_latents returns its argument identically. Preserve
    torch.stack's actual dense storage before taking a possibly strided tail.
    """
    b,s,steps,width=actions.shape
    if z.ndim==2:z=z[:,None].expand(-1,s,-1)
    need(z.shape[:2]==(b,s),'fixed source initial B,S')
    history=z.reshape(b*s,1,-1);act_hist=[];result=[]
    high=predict if predict is not None else lambda zz,aa:old.reference_high(t,state,graph,zz,aa,meter)
    for k in range(steps):
        act_k=actions[:,:,k,:].reshape(b*s,width)
        act_hist.append(act_k)
        act_seq=t.stack(act_hist,dim=1)
        context=min(state['high_predictor.pos_embedding'].shape[1],history.shape[1],act_seq.shape[1])
        nxt=high(history[:,-context:],act_seq[:,-context:])[:,-1:]
        history=t.cat((history,nxt),dim=1);result.append(nxt[:,0])
    return t.stack(result,dim=1).reshape(b,s,steps,-1)


def reference_cost(t,state,graph,z,goal,candidates,meter=None,*,predict=None,record=None):
    b,s,h,w=candidates.shape
    pred=reference_rollout(t,state,graph,z,candidates.reshape(b,s,h*(w//32),32),meter,predict=predict)
    terminal=pred[:,:,-1];expanded=goal[:,None].expand(-1,s,-1)
    residual=terminal-expanded;square=residual.square();value=square.sum(dim=-1)
    if record is not None:record(dict(terminal=terminal,goal=expanded,residual=residual,square=square,value=value))
    return value


def contiguous_layout(shape,device):
    stride=[];n=1
    for s in reversed(shape):stride.insert(0,n);n*=s
    return dict(shape=list(shape),stride=stride,storage_offset=0,contiguous=True,dtype='torch.float32',device=device)


def tail_layout(batch,length,width,context,device):
    shape=[batch,context,width];stride=[length*width,width,1];off=(length-context)*width
    return dict(shape=shape,stride=stride,storage_offset=off,contiguous=batch==1 or length==context,dtype='torch.float32',device=device)


class Journal:
    """Durable unfiltered dispatch/high lifecycle. No producer codec patch."""
    def __init__(self,directory):
        self.directory=Path(directory);self.path=self.directory/'events.jsonl';self.stream=self.path.open('xb');self.events=[]
    def emit(self,value):
        data=old.encoded(value).replace(b'\n',b'')+b'\n'
        need(sum(p.stat().st_size for p in self.directory.iterdir() if p.is_file())+len(data)<=CAP-1024**2
            and shutil.disk_usage(self.directory).free>=FLOOR+len(data),'journal cap/final1MiB reserve/floor')
        need(self.stream.write(data)==len(data),'full journal write');self.stream.flush();os.fsync(self.stream.fileno());self.events.append(deepcopy(value))
    def close(self):self.stream.flush();os.fsync(self.stream.fileno());self.stream.close()


class OwnedTrace:
    """Instance hooks observe unchanged loader, no replacement of model methods."""
    def __init__(self,root,save,emit,meter):
        self.root,self.save,self.emit,self.meter=root,save,emit,meter;self.rows=[];self.pending=None;self.handles=[]
    def install(self):
        def macro(module,args):
            need(self.pending is None and len(args)==1,'sequential macro entry')
            i=len(self.rows);self.pending=dict(index=i,actions=self.save('high'+str(i)+'_actions',args[0]))
        def pre(module,args):
            need(self.pending is not None and len(args)==2 and 'z' not in self.pending,'high entry after same macro')
            self.pending.update(z=self.save('high'+str(len(self.rows))+'_z',args[0]),
                condition=self.save('high'+str(len(self.rows))+'_condition',args[1]))
            event=dict(phase='attempt',batch=args[0].shape[0],context=args[0].shape[1]);self.meter.append(event)
            self.emit(dict(kind='high_attempt',route='owned',index=len(self.rows)))
        def post(module,args,value):
            need(self.pending is not None and 'predictor_output' not in self.pending,'high single return')
            self.pending['predictor_output']=self.save('high'+str(len(self.rows))+'_predictor',value)
            self.meter.append(dict(phase='return',batch=args[0].shape[0],context=args[0].shape[1]))
        def projected(module,args,value):
            need(self.pending is not None and 'predictor_output' in self.pending and len(args)==1,'projector after actual high return')
            i=len(self.rows);shape=self.pending['z']['layout']['shape']
            self.pending.update(projector_input=self.save('high'+str(i)+'_projector_input',args[0]),
                projector_output=self.save('high'+str(i)+'_projector_output',value),output=self.save('high'+str(i)+'_output',value.reshape(*shape)))
            self.rows.append(self.pending);self.pending=None;self.emit(dict(kind='high_return',route='owned',index=i))
        self.handles.append(self.root.macro_to_condition.register_forward_pre_hook(macro))
        self.handles.append(self.root.high_predictor.register_forward_pre_hook(pre))
        self.handles.append(self.root.high_predictor.register_forward_hook(post))
        self.handles.append(self.root.high_pred_proj.register_forward_hook(projected))
    def remove(self):
        for h in self.handles:h.remove()


def dispatch(t,root,state,graph,directory,tick,journal):
    """Actual four-case driver; injection seam permits only nonneural tests."""
    rows=[dict(id=c['id'],status='not_started',inputs={},traces={},owned=None,reference=None) for c in policy()['cases']]
    meters=dict(owned=[],reference=[]);ledger=[];active={};handles_removed=0
    def save(label,value):
        tick();return dict(array=old.save_array(directory,label+'.npy',value.detach().cpu().numpy()),layout=prior.layout(value))
    def owned_save(label,value):return save(active['id']+'_owned_'+label,value)
    observer=OwnedTrace(root,owned_save,journal.emit,meters['owned'])
    def call(label,fn):
        tick();need(len([x for x in ledger if x['phase']=='attempt'])<8,'original eight dispatch cap')
        e=dict(label=label,phase='attempt');ledger.append(e);journal.emit(dict(kind='dispatch',**e))
        v=fn();t.cuda.synchronize();tick()
        e=dict(label=label,phase='return');ledger.append(e);journal.emit(dict(kind='dispatch',**e));return v
    first_error=None
    try:
        observer.install()
        for case,row in zip(policy()['cases'],rows):
            row['status']='attempted';active['id']=case['id'];begin=len(observer.rows);reftrace=[]
            arrays={role:old.array_input(case[key],case['seed']+offset) for role,key,offset in
                (('z','z',0),('actions','a',7),('goal','goal',13)) if key in case}
            v={k:t.from_numpy(a).to(next(root.parameters()).device) for k,a in arrays.items()}
            row['inputs']={k:save(case['id']+'_input_'+k,x) for k,x in v.items()}
            def refhigh(z,a):
                i=len(reftrace);label=case['id']+'_reference_high'+str(i)
                q=dict(index=i,z=save(label+'_z',z),actions=save(label+'_actions',a));reftrace.append(q)
                journal.emit(dict(kind='high_attempt',route='reference',index=i))
                value=old.reference_high(t,state,graph,z,a,meters['reference'])
                q['output']=save(label+'_output',value);journal.emit(dict(kind='high_return',route='reference',index=i));return value
            def rec_cost(values):row['cost_buffers']={k:save(case['id']+'_cost_'+k,x) for k,x in values.items()}
            with t.inference_mode():
                if case['kind']=='direct':
                    actual=call(case['id']+':owned',lambda:root.predict_high(v['z'],v['actions']))
                    ref=call(case['id']+':reference',lambda:refhigh(v['z'],v['actions']))
                elif case['kind']=='rollout':
                    actual=call(case['id']+':owned',lambda:root.rollout_high(v['z'],v['actions']))
                    ref=call(case['id']+':reference',lambda:reference_rollout(t,state,graph,v['z'],v['actions'],predict=refhigh))
                else:
                    actual=call(case['id']+':owned',lambda:root.get_cost(dict(planner_level='high',z_init=v['z'],z_goal=v['goal']),v['actions']))
                    ref=call(case['id']+':reference',lambda:reference_cost(t,state,graph,v['z'],v['goal'],v['actions'],predict=refhigh,record=rec_cost))
            row['traces']=dict(owned=observer.rows[begin:],reference=reftrace)
            row['owned']=save(case['id']+'_owned',actual);row['reference']=save(case['id']+'_reference',ref)
            a,b=(x.detach().cpu().numpy() for x in (actual,ref));row['comparison']=prior.comparison(a,b)
            need(list(a.shape)==case['output'] and row['comparison']['exact_equal'],'full fixed case mismatch, no tolerance')
            row['status']='recorded'
    except Exception as exc:
        first_error=dict(type=type(exc).__name__,message=str(exc)[:2000])
        # Preserve every reached tensor/entry and all four denominator rows.
        for row in rows:
            if row['status']=='attempted':
                row['status']='failed_partial';row['traces']=dict(owned=observer.rows[begin:],reference=reftrace)
    finally:
        observer.remove();handles_removed=len(observer.handles)
    return dict(rows=rows,ledger=ledger,high_predictor_meters=meters,hooks_removed=handles_removed,
        pending_owned=observer.pending,errors=[] if first_error is None else [first_error])


def run_cuda(path,h,output,*,main_authorized=False):
    need(main_authorized is True,'MAIN HOST CUDA authorization required')
    need(os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','CUBLAS_WORKSPACE_CONFIG=:4096:8 required before torch import')
    started=time.monotonic();m=verify(path,h);output=scoped(output)
    need(shutil.disk_usage(output.parent).free>=CAP+FLOOR,'32MiB output plus200MiB disk floor before CUDA')
    output.mkdir(mode=0o700)
    journal=Journal(output);before=after=construction=runtime=None;result=None;errors=[];root=None
    def tick():need(time.monotonic()-started<=WALL,'whole600s including verification boundary watchdog')
    try:
        import torch as t
        need(t.cuda.is_available(),'HOST CUDA only, never CPU neural fallback')
        t.use_deterministic_algorithms(True);t.backends.cuda.matmul.allow_tf32=False;t.backends.cudnn.allow_tf32=False
        tick();root,construction=loader.construct_cuda(m['export_report'],m['export_report_sha256'],m['export_receipt'],m['export_receipt_sha256'],main_authorized=True)
        device=next(root.parameters()).device;state=root.state_dict();_,graph,_=loader.audit();before=old.weights(root)
        need(before==construction['tensor_value_hashes'],'actual initial tensors export exact')
        runtime=dict(device=str(device),torch=str(t.__version__),cuda=t.version.cuda,name=t.cuda.get_device_name(device),
            deterministic=t.are_deterministic_algorithms_enabled(),tf32_matmul=t.backends.cuda.matmul.allow_tf32,
            tf32_cudnn=t.backends.cudnn.allow_tf32,autocast=t.is_autocast_enabled('cuda'))
        need(runtime['autocast'] is False,'no mixed precision')
        result=dispatch(t,root,state,graph,output,tick,journal);errors.extend(result['errors'])
        after=old.weights(root);need(after==before,'all parameters/buffers byte-identical');verify(path,h);tick()
    except Exception as exc:
        errors.append(dict(type=type(exc).__name__,message=str(exc)[:2000]))
        if root is not None:
            try:after=old.weights(root)
            except Exception as final:errors.append(dict(type=type(final).__name__,message='final weights: '+str(final)[:1000]))
    finally:journal.close()
    if result is None:result=dict(rows=[dict(id=c['id'],status='not_started',inputs={},traces={},owned=None,reference=None) for c in policy()['cases']],
        ledger=[],high_predictor_meters=dict(owned=[],reference=[]),hooks_removed=0,pending_owned=None)
    report=dict(schema=TAG+'.report',status='all4_source_layout_reference_verified_not_release' if not errors else 'failed_partial_preserved',
        manifest_path=str(Path(path).absolute()),manifest_sha256=h,own_sha256=closure(),policy=policy(),
        **{k:v for k,v in result.items() if k!='errors'},errors=errors,model_before=before,model_after=after,construction=construction,runtime=runtime,
        journal_sha256=loader.sha(loader.bytes_read(journal.path,CAP)),journal_events=journal.events,wall_seconds=time.monotonic()-started,
        official_original_model_parity=False,official_parity=False,baseline_runnable=False,legacy_v1_released=False,scientific_admission=False,
        encoder_calls=0,macro_calls=0,AD=0,physics=0)
    rh=old.write_new(output/'report.json',old.encoded(report))
    need(not errors,'failed partial preserved at '+str(output/'report.json'));return dict(path=str(output/'report.json'),sha256=rh,status=report['status'])


def validate_cost_reduction(square,value):
    """Independent conservative arithmetic bound, never route-parity tolerance."""
    need(square.dtype==value.dtype==np.dtype('<f4') and square.shape==(2,2,192)
        and value.shape==(2,2) and np.isfinite(square).all() and np.isfinite(value).all()
        and (square>=0).all() and (value>=0).all(),'finite nonnegative original cost reduction ABI')
    # Any f32 reduction tree has at most n-1 additions on a path. Use n=192
    # conservatively, plus one minimum f32 subnormal per operation. Account
    # independently for rounding the f64 audit sum, not NumPy/CUDA bit parity.
    n=192;u32=2.**-24;u64=2.**-53
    g32=n*u32/(1-n*u32);g64=n*u64/(1-n*u64)
    total=square.astype(np.float64).sum(axis=-1)
    bound=np.nextafter((g32+g64)/(1-g64)*total+n*2.**-149,np.inf)
    error=np.abs(value.astype(np.float64)-total)
    need((error<=bound).all(),'independent source cost float32 reduction bound')
    return dict(float64_sum=total.tolist(),absolute_error=error.tolist(),absolute_bound=bound.tolist())


def validate_arrays(directory,r):
    files={'report.json','events.jsonl'};seen=set();device=r['runtime']['device']
    owned_offset=0
    def read(ref):
        need(set(ref)=={'array','layout'},'exact tensor descriptor/layout ABI');d,l=ref['array'],ref['layout'];a=old.read_array(directory,d)
        need(d['file'] not in seen,'unique saved tensor role filename');seen.add(d['file']);files.add(d['file'])
        need(set(l)=={'shape','stride','storage_offset','contiguous','dtype','device'} and l['shape']==list(a.shape)
            and l['dtype']=='torch.float32' and l['device']==device and len(l['stride'])==a.ndim
            and all(type(s) is int and s>=0 for s in l['stride']) and type(l['storage_offset']) is int and l['storage_offset']>=0,'real bounded layout')
        expected=1;contiguous=True
        for size,stride in zip(reversed(a.shape),reversed(l['stride'])):
            if size>1:contiguous &= stride==expected;expected*=size
        need(type(l['contiguous']) is bool and l['contiguous']==contiguous,'layout contiguity arithmetic');return a
    def eq(a,b):return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
    for case,row in zip(policy()['cases'],r['rows'],strict=True):
        inputs={k:read(v) for k,v in row['inputs'].items()}
        roles={k:(case[shape],case['seed']+off) for k,shape,off in (('z','z',0),('actions','a',7),('goal','goal',13)) if shape in case}
        need(set(inputs)==set(roles),'complete original input roles')
        for k,(shape,seed) in roles.items():need(eq(inputs[k],old.array_input(shape,seed)) and row['inputs'][k]['layout']==contiguous_layout(shape,device),'literal input recipe/layout')
        route_outputs={}
        for route in ('owned','reference'):
            trace=row['traces'][route];steps=1 if case['kind']=='direct' else (5 if case['kind']=='rollout' else 4)
            need(len(trace)==steps,'all native/high reference steps')
            if case['kind']=='direct':history=inputs['z'];actions=inputs['actions']
            else:
                b,s=case['a'][:2];actions=inputs['actions'].reshape(b,s,steps,32)
                history=np.broadcast_to(inputs['z'][:,None],(b,s,192)).reshape(b*s,1,192);packets=[]
            outputs=[]
            for k,q in enumerate(trace):
                need(type(q['index']) is int and q['index']==(owned_offset+k if route=='owned' else k),'literal native-global/reference-local trace index')
                need(set(q)==({'index','actions','z','condition','predictor_output','projector_input','projector_output','output'} if route=='owned' else {'index','z','actions','output'}),'complete trace tensor roles')
                data={name:read(ref) for name,ref in q.items() if name!='index'}
                if case['kind']=='direct':z,a=history,actions;zl=contiguous_layout(list(z.shape),device);al=contiguous_layout(list(a.shape),device)
                else:
                    packets.append(actions[:,:,k,:].reshape(b*s,32));sequence=np.stack(packets,axis=1);c=min(4,k+1)
                    z,a=history[:,-c:],sequence[:,-c:];zl=tail_layout(b*s,k+1,192,c,device);al=tail_layout(b*s,k+1,32,c,device)
                need(eq(data['z'],z) and eq(data['actions'],a) and q['z']['layout']==zl and q['actions']['layout']==al,'exact own-history/source append-stack input bytes AND strides/offset')
                need(data['output'].shape==z.shape,'full native high output')
                if route=='owned':
                    need(data['condition'].shape==data['predictor_output'].shape==z.shape
                        and eq(data['projector_input'],data['predictor_output'].reshape(-1,192))
                        and eq(data['output'],data['projector_output'].reshape(z.shape)),'actual full module arrays and projection joins')
                if case['kind']=='direct':route_outputs[route]=data['output']
                else:
                    nxt=data['output'][:,-1:];history=np.concatenate((history,nxt),axis=1);outputs.append(nxt[:,0])
            if case['kind']!='direct':route_outputs[route]=np.stack(outputs,axis=1).reshape(b,s,steps,192)
        a,b=read(row['owned']),read(row['reference'])
        need(list(a.shape)==list(b.shape)==case['output'] and old.encoded(prior.comparison(a,b))==old.encoded(row['comparison'])
            and row['comparison']['exact_equal'] is True,'complete four outputs byte exact, no tolerance')
        if case['kind']=='cost':
            data={k:read(v) for k,v in row['cost_buffers'].items()};need(set(data)=={'terminal','goal','residual','square','value'},'complete unchanged source cost formula buffers')
            terminal=route_outputs['reference'][:,:,-1];goal=np.broadcast_to(inputs['goal'][:,None],terminal.shape)
            need(eq(data['terminal'],terminal) and eq(data['goal'],goal) and eq(data['residual'],terminal-goal)
                and eq(data['square'],data['residual']**2) and eq(data['value'],b)
                and eq(route_outputs['owned'],route_outputs['reference']),'source terminal/goal/residual/square/source CUDA sum output exact')
            # GPU sum(dim=-1) is source-bound, not falsely relabelled NumPy's
            # reduction bits. Exact route sum equality above remains mandatory.
            validate_cost_reduction(data['square'],data['value'])
        else:need(eq(a,route_outputs['owned']) and eq(b,route_outputs['reference']),'full trace to original output join')
        owned_offset+=1 if case['kind']=='direct' else 5 if case['kind']=='rollout' else 4
    need({p.name for p in Path(directory).iterdir()}==files and all(p.is_file() and not p.is_symlink() for p in Path(directory).iterdir()),'complete arrays/journal/report inventory')
    need(sum(p.stat().st_size for p in Path(directory).iterdir())<=CAP,'original32MiB output cap')


def consume(path,h,report_path,rh):
    m=verify(path,h);r=loader.json_read(report_path,rh);directory=Path(report_path).parent
    need(r['schema']==TAG+'.report' and r['status']=='all4_source_layout_reference_verified_not_release' and r['errors']==[]
        and r['manifest_path']==str(Path(path).absolute()) and r['manifest_sha256']==h and r['own_sha256']==closure()
        and old.encoded(r['policy'])==old.encoded(policy()) and len(r['rows'])==4 and r['pending_owned'] is None,'complete own v3 authority, never old consume')
    need([v['id'] for v in r['rows']]==[v['id'] for v in policy()['cases']] and all(v['status']=='recorded' for v in r['rows']),'original four fixed cases')
    need(all(r[k] is False for k in ('official_original_model_parity','official_parity','baseline_runnable','legacy_v1_released','scientific_admission'))
        and all(type(r[k]) is int and r[k]==0 for k in ('encoder_calls','macro_calls','AD','physics')),'no expanded resource/science claims')
    _,spec,_,export=loader.authorities(m['export_report'],m['export_report_sha256'],m['export_receipt'],m['export_receipt_sha256'])
    expected={e['key']:row['array']['value_sha256'] for row,e in zip(export['rows'],spec['entries'],strict=True) if row['array'] is not None}
    c=r['construction'];runtime=r['runtime']
    need(r['model_before']==r['model_after']==c['tensor_value_hashes']==expected and c['tensor_copy_attempts']==c['tensor_copy_returns']==425
        and c['state_key_count']==425 and c['module_count']==531 and c['entry_count']==491 and c['all_parameters_frozen'] is True
        and c['forward_calls']==0 and c['baseline_runnable'] is False,'complete immutable exported weights/construction')
    need(c['schema']==loader.TAG+'.construction' and c['status']=='owned_tensor_graph_constructed_not_native_forward_verified'
        and c['ir_sha256']==loader.IR_SHA and c['own_sha256']=={str(ROOT/p):v for p,v in old.PINS.items() if 'trusted_loader' in p}
        and c['export_report_sha256']==loader.EXPORT_SHA and c['export_receipt_sha256']==m['export_receipt_sha256']
        and c['device']==runtime['device'] and c['torch']==runtime['torch'],'exact source/export/device identity')
    need(runtime['device'].startswith('cuda:') and runtime['torch'] and runtime['cuda'] and runtime['name']
        and runtime['deterministic'] is True and runtime['tf32_matmul'] is runtime['tf32_cudnn'] is runtime['autocast'] is False
        and type(r['wall_seconds']) in (int,float) and math.isfinite(r['wall_seconds']) and 0<=r['wall_seconds']<=WALL,'actual CUDA/precision/whole-stage wall')
    shapes=[(1,1),(2,4)]+[(4,k) for k in (1,2,3,4,4,1,2,3,4)]
    meter=[dict(phase=p,batch=b,context=c) for b,c in shapes for p in ('attempt','return')]
    need(old.encoded(r['high_predictor_meters'])==old.encoded(dict(owned=meter,reference=meter))
        and type(r['hooks_removed']) is int and r['hooks_removed']==4,'all11 high predictor calls per route and hooks removed')
    ledger=[];events=[]
    for case in policy()['cases']:
        n=1 if case['kind']=='direct' else 5 if case['kind']=='rollout' else 4
        for route in ('owned','reference'):
            label=case['id']+':'+route;ledger.append(dict(label=label,phase='attempt'));events.append(dict(kind='dispatch',label=label,phase='attempt'))
            for i in range(n):events.extend([dict(kind='high_attempt',route=route,index=i),dict(kind='high_return',route=route,index=i)])
            ledger.append(dict(label=label,phase='return'));events.append(dict(kind='dispatch',label=label,phase='return'))
    # Owned observer serial is global; reference indices restart per original case.
    offset=0
    for case in policy()['cases']:
        n=1 if case['kind']=='direct' else 5 if case['kind']=='rollout' else 4
        start=next(i for i,e in enumerate(events) if e==dict(kind='dispatch',label=case['id']+':owned',phase='attempt'))
        for e in events[start+1:start+1+2*n]:e['index']+=offset
        offset+=n
    journal=loader.bytes_read(directory/'events.jsonl',CAP)
    need(loader.sha(journal)==r['journal_sha256'] and old.encoded(r['ledger'])==old.encoded(ledger) and old.encoded(r['journal_events'])==old.encoded(events)
        and journal==b''.join(old.encoded(e).replace(b'\n',b'')+b'\n' for e in events),'all8 dispatches/all22 high entries/returns exact durable chronology, no unknown events')
    validate_arrays(directory,r);need(closure()==m['own_sha256'],'final own source seal')
    return dict(schema=TAG+'.consume',status='all4_saved_source_layout_reference_verified_not_release',manifest_sha256=h,report_sha256=rh,
        rows_verified=4,top_dispatches=8,high_predict_calls_per_route=11,high_predict_calls_total=22,
        upstream_v2_report_sha256=REPORT_SHA,official_original_model_parity=False,official_parity=False,baseline_runnable=False,
        legacy_v1_released=False,scientific_admission=False)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    c=sub.add_parser('prepare');c.add_argument('--output',required=True,type=Path);c.add_argument('--main-authorized',action='store_true')
    for cmd in ('verify','run','consume'):
        c=sub.add_parser(cmd);c.add_argument('--manifest',required=True,type=Path);c.add_argument('--manifest-sha256',required=True)
        if cmd=='run':c.add_argument('--output',required=True,type=Path);c.add_argument('--main-authorized',action='store_true')
        if cmd=='consume':
            c.add_argument('--report',required=True,type=Path);c.add_argument('--report-sha256',required=True)
            c.add_argument('--receipt-output',type=Path)
    a=p.parse_args(argv)
    if a.command=='prepare':r=prepare(a.output,main_authorized=a.main_authorized)
    elif a.command=='verify':verify(a.manifest,a.manifest_sha256);r=dict(status='v3_saved_prepare_verified_no_forward')
    elif a.command=='run':r=run_cuda(a.manifest,a.manifest_sha256,a.output,main_authorized=a.main_authorized)
    else:
        r=consume(a.manifest,a.manifest_sha256,a.report,a.report_sha256)
        if a.receipt_output is not None:
            directory=scoped(a.receipt_output);directory.mkdir(mode=0o700)
            rh=old.write_new(directory/'receipt.json',old.encoded(r));r=dict(path=str(directory/'receipt.json'),sha256=rh,status=r['status'])
    print(old.encoded(r).decode(),end='')


if __name__=='__main__':main()
