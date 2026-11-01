"""MAIN-only actual-mid suffix intervention, reusing frozen full10 native runner.

Compile from strict saved RGB, then replay-gate the identical physical mid before
dispatching the new suffix. No new roots, selection by outcomes, or efficacy gate.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from io import BytesIO
import hashlib
import os
from pathlib import Path
import shutil
import time
import numpy as np
from experiments.spectral_reachability import agent_p2_contextual_physical_rgb_encode_20261028_v1 as bridge
from experiments.spectral_reachability import p1_moving_live_suffix_cuda_20261027_v1 as suffix

NAME='agent_p2_contextual_midfeedback_recompile_20261028_v1'
ROOT=bridge.ROOT
physical=bridge.physical
kernel=physical.kernel
F=suffix.frozen
require,sha,raw=physical.require,physical.sha,physical.raw
AUTH='MAIN_CONTEXTUAL_MIDFEEDBACK_RECOMPILE_HOST_ONLY'
RGB_SOURCE_SHA='fa1d44fb2b753b2deb2c1743cfa0d0496b4fdea42f92d4190cce231260d5375e'
RGB_REPORT_SHA='b57fa03560696c3c0b75fe842ec8263d6ec3961341d6736a4f6e782d1f0e92ea'
RGB_RECEIPT_SHA='3d3d020c113959aec25f4e310b8512b814f7a78976f2bd7d5e58e0e435027f4b'


def contract():
    return dict(plans=8,known_roots=2,scientific_slots=4,warmup=10,prefix=5,suffix=5,horizon=10,
        intervention='new bounded-ridge suffix from actual same-condition mid RGB; old suffix diagnostic only',
        execution='saved observation-conditioned compile, then native full10 replay gated at exact SAME mid; not online latency benchmark',
        choice='always new solve; never choose between old/new by predicted or physical outcome',
        Q='unchanged native192 identity terminal MSE',goal='original sealed goal192 and native task goal',
        domain='remaining original joint20 logit ball .75sqrt(10), original selected prefix logits',damping=.002,
        per_compile=F.budget(),predict_API_per_compile=5,AD_API_per_compile=dict(jacobian=1,grad=1),
        total_compile={k:8*v for k,v in F.budget().items()},mathematical_VJP_directions=1536,
        compiler_candidates=8,old_suffix_diagnostic=8,new_suffix_evaluation=8,new_suffix_replay=8,
        compiler_encoder=0,compiler_projector=0,physical_caps=kernel.contract()['caps'],
        actual_call_axes='API includes predictor/proj; do not add nested axes',total_FLOPs=None,
        inherited_compute=None,equal_FLOPs=False,equal_execution_horizon=True,
        wall_seconds=900,output_cap_bytes=kernel.CAP,final_reserve_bytes=kernel.FINAL,disk_floor_bytes=kernel.FLOOR,
        P1_admission=False,P2_admission=False,fresh=False,physical_release=False,training=False,Qfit=False,retries=0)


def own_files():
    return [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]


def write(path,value):
    path=Path(path).absolute();payload=raw(value)
    require(path.name.startswith(NAME+'_') and not path.exists() and not path.is_symlink()
        and len(payload)<=8*1024**2 and shutil.disk_usage(path.parent).free>=kernel.FLOOR+len(payload),'new own bounded metadata')
    kernel.paired.fsync_dir(path.parent)
    with path.open('xb') as f:require(f.write(payload)==len(payload),'full metadata');f.flush();os.fsync(f.fileno())
    kernel.paired.fsync_dir(path.parent)
    return dict(path=str(path),sha256=sha(path),status=value.get('status','prepared'))


def branch(spec,plan,z0,z5,selected_logits):
    """Compiler whitelist excludes native state, rewards and terminal outcomes."""
    require(physical.bits(plan['commands'])==spec['raw_float32_bits'],'literal parent full10')
    u=np.asarray(selected_logits,np.float64)
    require(u.shape==(20,) and np.isfinite(u).all() and
        np.tanh(u).astype(np.float32).reshape(10,2).tobytes().hex()==spec['raw_float32_bits'],'original selected logits, not inverse rounded raw')
    cap=float(.75*np.sqrt(10));remaining=cap**2-float(u[:10]@u[:10])
    require(remaining>0,'zero remaining radius blocks this branch; never enlarge domain')
    literal=bridge.mid_inputs(spec,z0,z5)
    context=deepcopy(spec['root_context']);context.update(latent_history=literal['context'][0].tolist(),
        action_history=literal['outgoing'][0,:2].tolist())
    b=dict(observed_context=context,observed_raw_history=deepcopy(spec['frames'][1]['source']['raw_action_history10']),
        prefix_commands=deepcopy(plan['commands'][:5]),old_suffix_commands=deepcopy(plan['commands'][5:]),
        goal=deepcopy(plan['goal']),q=np.eye(192).tolist(),action_mean=deepcopy(plan['encoder_identity']['action_mean']),
        action_std=deepcopy(plan['encoder_identity']['action_std']),packet_logit_cap=float(np.sqrt(remaining)))
    b['branch_id']=F.core.digest(b)
    return b


def build(rgb_output,rgb_receipt):
    require(sha(bridge.__file__)==RGB_SOURCE_SHA,'frozen RGB bridge source')
    output=Path(rgb_output).resolve();receipt=Path(rgb_receipt).resolve()
    require(sha(receipt)==RGB_RECEIPT_SHA,'MAIN actual full RGB strict receipt')
    strict=bridge.consume(output,RGB_REPORT_SHA)
    require(raw(strict)==raw(physical.read(receipt)),'whole actual RGB strict reconstruction')
    report=physical.read(output/'report.json');p=bridge.verify(report['prepared'],report['prepare_sha256'])
    physreport=physical.read(Path(p['arguments']['physical_output'])/'report.json')
    pp,data=physical.verify(physreport['execution_binding']['prepared'],physreport['execution_binding']['prepare_sha256'])
    planner=[physical.saved.overlay.read(Path(pp['planner_output'])/f'slot{i}_result.json.gz',physical.saved.EXPANDED) for i in range(4)]
    branches=[]
    for i,(spec,row,plan) in enumerate(zip(p['rows'],report['rows'],pp['plans'],strict=True)):
        require(all(f['source'] is not None for f in spec['frames'][:2]),'missing mid input: keep full8 blocked, no subset')
        winner=planner[i//2]['results'][i%2]
        selected=next(a for a in winner['attempts'] if a['candidate_id']==winner['winner']['candidate_id'])
        load=lambda x:bridge.rgb.load_array(x,lambda n:(output/n).read_bytes())
        branches.append(branch(spec,plan,load(row['frames'][0]['latent']),load(row['frames'][1]['latent']),selected['logits']))
    bindings=dict(p['bindings']);physical.merge(bindings,{str(output/'report.json'):RGB_REPORT_SHA,str(receipt):RGB_RECEIPT_SHA,
        report['prepared']:report['prepare_sha256']})
    physical.merge(bindings,{str(output/n):h for n,h in report['artifact_sha256'].items()})
    physical.merge(bindings,suffix.source_bindings())
    physical.merge(bindings,{str(f):sha(f) for f in own_files()})
    bridge.verify_bindings(bindings)
    return dict(schema=NAME+'.prepare',arguments=dict(rgb_output=str(output),rgb_receipt=str(receipt)),contract=contract(),
        branches=branches,plans=pp['plans'],midpoints=[deepcopy(s['frames'][1]['source']) for s in p['rows']],
        bindings=bindings,physical_prepare=physreport['execution_binding']['prepared'],
        physical_prepare_sha256=physreport['execution_binding']['prepare_sha256'],policy=p['policy'],
        expected_model_hash=p['expected_model_hash'],inherited_RGB_counts=report['counts'],
        inherited_compiler_resources=p['inherited_compiler_resources'],inherited_physical_counts=p['inherited_physical_counts'],
        control_rows=physical.read(p['arguments']['physical_receipt'])['rows'])


def verify(prepared,expected):
    require(sha(prepared)==expected,'own prepare SHA');p=physical.read(prepared)
    require(raw(p)==raw(build(**p['arguments'])),'complete saved preparation reconstruction');return p


class Store(kernel.Store):
    def __init__(self,path):
        require(Path(path).name.startswith(NAME+'_'),'own output only');super().__init__(path)
        self.branch=None;self.started=time.monotonic()
    @property
    def output(self):return self.path
    def check(self,size=4096,final=False):
        require(final or time.monotonic()-self.started<900,'900s whole stage wall');super().check(size,final)
    def _write(self,name,value):return self.write(name,value)
    def event(self,e):self.emit('compiler_event',branch=self.branch,event=e)


def load_npy(output,entry):
    b=bridge.rgb.flat(output,entry['file']).read_bytes();require(hashlib.sha256(b).hexdigest()==entry['sha256'],'full native file SHA')
    s=BytesIO(b);a=np.load(s,allow_pickle=False)
    require(s.tell()==len(b) and str(a.dtype)==entry['dtype'] and list(a.shape)==entry['shape'] and np.isfinite(a).all(),'single finite original NPY')
    return a


def check_compile(b,row,load):
    F.check_result(b,row);F.saved_arrays(b,row,load)
    F.blind.exact_counts(row['resources'],F.budget());F.blind.exact_counts(row['AD_API_resources'],dict(jacobian=1,grad=1))
    F.blind.exact_counts(row['predict_API_resources'],dict(predict=5))
    require(row['predict_API_ledger']==[dict(index=i,stage=s,completed=True) for i,s in enumerate(F.STAGES)],'all five actual API calls')
    nominal=load(row['arrays']['nominal']);jac=load(row['arrays']['jacobian'])
    solve=F.compiler.bounded_ridge(np.asarray(b['q'])@jac,np.asarray(b['q'])@(np.asarray(b['goal'])-nominal),
        damping=F.compiler.DAMPING,cap=b['packet_logit_cap'])
    require(solve==row['solve'],'independent exact saved bounded solve, not Q fitting')


def check_events(b,row,events):
    """Exact original ABI and nested nonrefundable attempts/inputs/returns."""
    ledger_keys=dict(module='module_ledger',math='math_ledger',AD_API='AD_API_ledger',predict_API='predict_API_ledger')
    terminal=dict(module='module_return',math='math_return',AD_API='AD_API_completed',predict_API='predict_API_return')
    allowed={'stage_started','nominal_saved','jacobian_saved','new_suffix_frozen_before_evaluation',
        'suffix_evaluation_attempt','suffix_evaluation_return','module_inputs_saved'}
    allowed|={k+'_attempt' for k in ledger_keys}|set(terminal.values())
    require(all(e['kind'] in allowed for e in events),'no skipped unknown compiler event')
    starts=[e for e in events if e['kind']=='stage_started']
    require([e['stage'] for e in starts]==list(F.STAGES),'all five fixed forward stages')
    raws=[np.zeros((5,2),np.float32).tolist()]*2+[b['old_suffix_commands'],row['new_suffix_commands'],row['new_suffix_commands']]
    for e,s,commands in zip(starts,F.STAGES,raws,strict=True):
        require(e['branch_id']==b['branch_id'] and e['commands']==commands and
            e['expected_input']==F.planner.incoming(b,b['observed_context'],commands),'each literal model context/action')
    for kind,key in ledger_keys.items():
        entries=row[key];attempts=[e['entry'] for e in events if e['kind']==kind+'_attempt']
        want=[dict(e,completed=False) if kind!='module' else
            dict(index=e['index'],resource=e['resource'],stage=e['stage'],completed=False,inputs=[]) for e in entries]
        returns=[entries[1],entries[0]] if kind=='AD_API' else entries
        require(attempts==want and [e['entry'] for e in events if e['kind']==terminal[kind]]==returns,'all actual attempted/returned '+kind)
    require([e['entry'] for e in events if e['kind']=='module_inputs_saved']==
        [{k:v for k,v in e.items() if k!='output'}|{'completed':False} for e in row['module_ledger']],'all native inputs durable before call')
    stack=[];active_stage=None;seen_modules=set()
    for e in events:
        kind=e['kind']
        if kind=='stage_started':require(not stack,'previous expensive calls returned before next stage');active_stage=e['stage']
        elif kind.endswith('_attempt') and kind[:-8] in ledger_keys:
            tag=kind[:-8];entry=e['entry']
            require(entry.get('stage',active_stage)==active_stage,'actual operation stage')
            stack.append((tag,entry['index']))
        elif kind=='module_inputs_saved':
            require(stack and stack[-1]==('module',e['entry']['index']),'inputs between native attempt and return')
            seen_modules.add(e['entry']['index'])
        elif kind in terminal.values():
            tag=next(k for k,v in terminal.items() if v==kind)
            require(stack and stack.pop()==(tag,e['entry']['index']),'exact nested native/math/AD/API return')
            if tag=='module':require(e['entry']['index'] in seen_modules,'no return without durable input')
    require(not stack,'no pending expensive work')
    kinds=[e['kind'] for e in events]
    require(kinds.count('nominal_saved')==kinds.count('jacobian_saved')==kinds.count('new_suffix_frozen_before_evaluation')==1
        and kinds.index('nominal_saved')<kinds.index('jacobian_saved')<kinds.index('new_suffix_frozen_before_evaluation')
        <kinds.index('suffix_evaluation_attempt'),'solve and action freeze BEFORE diagnostic outcomes')
    require([e['receipt'] for e in events if e['kind']=='suffix_evaluation_return']==row['evaluations'] and
        [e['receipt'] for e in events if e['kind']=='suffix_evaluation_attempt']==[
            dict(stage=e['stage'],commands=e['commands'],status='attempted') for e in row['evaluations']],
        'all three evaluation intent/return pairs')


def compile_run(prepared,expected,output,authorization):
    require(authorization==AUTH,'MAIN explicit CUDA before I/O')
    started=time.monotonic();p=verify(prepared,expected);store=Store(output);store.started=started
    rows=[dict(index=i,branch_id=b['branch_id'],status='not_started') for i,b in enumerate(p['branches'])]
    errors=[];model=None;before=after=None
    store.put('binding.json',dict(prepared=str(Path(prepared).resolve()),prepare_sha256=expected))
    try:
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import sdpa_kernel,SDPBackend
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        bridge.native.frozen(model);before=bridge.native.tensor_hash(model)
        require(before==p['expected_model_hash'],'original frozen tensor hash')
        mm,cn=torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32
        forbidden=[]
        def forbid(*args):raise ValueError('no new visual calls during saved-mid compilation')
        try:
            forbidden=[getattr(model,k).register_forward_pre_hook(forbid) for k in ('encoder','projector')]
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            with torch.enable_grad(),torch.autocast(device_type='cuda',enabled=False),sdpa_kernel(SDPBackend.MATH):
                for i,(b,row) in enumerate(zip(p['branches'],rows,strict=True)):
                    store.check();store.branch=i;row['status']='attempted'
                    store.emit('compile_intent',branch=i,branch_id=b['branch_id'],budget=F.budget())
                    sink=suffix.Sink(store,store.check);sync=lambda:torch.cuda.synchronize(device)
                    meter=F.Meter(sink.event,sync);meter.require_native=True
                    modules=F.Modules(model,sink,f'branch_{i:02d}',meter,sync)
                    ad=F.SuffixAD(torch,F.blind.ARMS[1],sink.event,sync);ad.stage=lambda:meter.stage
                    api=suffix.PredictAPI(model,meter,sink,sync)
                    try:
                        with ad,modules,api:F.evaluate(b,F.d2p.TorchBackend(model,device),meter,sink,row)
                    finally:
                        row.update(resources={**modules.counts(),**meter.counts()},AD_API_resources=ad.counts(),
                            predict_API_resources=api.counts(),module_ledger=deepcopy(modules.entries),math_ledger=deepcopy(meter.entries),
                            AD_API_ledger=deepcopy(ad.events),predict_API_ledger=deepcopy(api.entries))
                    check_compile(b,row,sink.load);store.emit('compile_sealed',branch=i,row=row)
        finally:
            for hook in reversed(forbidden):hook.remove()
            torch.backends.cuda.matmul.allow_tf32=mm;torch.backends.cudnn.allow_tf32=cn
    except BaseException as e:errors.append(kernel.error('compile',e))
    finally:
        try:
            if model is not None:bridge.native.frozen(model);after=bridge.native.tensor_hash(model)
            bridge.verify_bindings(p['bindings']);store.check()
        except BaseException as e:errors.append(kernel.error('final_model_source_wall',e))
        complete=not errors and before==after==p['expected_model_hash'] and all(r['status']=='suffix_compiled' for r in rows)
        store.emit('run_finished',status='all8_compiled_science_blocked' if complete else 'partial_or_failed')
        report=dict(schema=NAME+'.compile',status='all8_compiled_science_blocked' if complete else 'partial_or_failed',
            prepared=str(Path(prepared).resolve()),prepare_sha256=expected,contract=contract(),rows=rows,errors=errors,
            model_hash_before=before,model_hash_after=after,elapsed_seconds=time.monotonic()-store.started,
            artifact_sha256={f.name:sha(f) for f in store.path.iterdir()},journal_chain_tail=store.previous)
        store.put('report.json',report,final=True)
    require(complete,'partial retained/no retries: '+str(errors))
    return dict(report=str(store.path/'report.json'),report_sha256=sha(store.path/'report.json'))


def consume_compile(output,expected):
    output=Path(output).resolve();require(sha(output/'report.json')==expected,'external actual compile report SHA')
    r=physical.read(output/'report.json');p=verify(r['prepared'],r['prepare_sha256'])
    require(r['schema']==NAME+'.compile' and r['status']=='all8_compiled_science_blocked' and r['errors']==[] and
        r['contract']==contract() and len(r['rows'])==8 and r['model_hash_before']==r['model_hash_after']==p['expected_model_hash']
        and r['elapsed_seconds']<900,'whole completed frozen compilation only')
    require({f.name for f in output.iterdir()}==set(r['artifact_sha256'])|{'report.json'},'entire compile census')
    for n,h in r['artifact_sha256'].items():require(sha(bridge.rgb.flat(output,n))==h,'compile artifact SHA')
    events=physical.journal(output/'journal.jsonl',r['journal_chain_tail']);plans=[]
    require(all(e['kind'] in ('compile_intent','compiler_event','compile_sealed','run_finished') for e in events) and
        events[-1]['status']==r['status'] and [e['branch'] for e in events[:-1]]==sorted(e['branch'] for e in events[:-1]),
        'whole ordered compiler journal no skipped work')
    require(physical.read(output/'binding.json')==dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256']),'actual compile binding')
    require([e['branch'] for e in events if e['kind']=='compile_intent']==list(range(8)) and
        [e['row'] for e in events if e['kind']=='compile_sealed']==r['rows'],'all8 durable intent and sealed results')
    for i,(b,row,original) in enumerate(zip(p['branches'],r['rows'],p['plans'],strict=True)):
        segment=[e for e in events[:-1] if e['branch']==i]
        require(segment[0]['kind']=='compile_intent' and segment[-1]['kind']=='compile_sealed' and
            all(e['kind']=='compiler_event' for e in segment[1:-1]) and segment[0]['branch_id']==b['branch_id']
            and segment[0]['budget']==F.budget(),'actual original finite per-branch reserve, work, seal')
        require(row['index']==i and row['branch_id']==b['branch_id'],'fixed8 row identity')
        check_compile(b,row,lambda e:load_npy(output,e))
        own=[e['event'] for e in events if e['kind']=='compiler_event' and e['branch']==i]
        check_events(b,row,own)
        plans.append(seal_plan(b,row,original,expected))
    require(all(type(e['branch']) is int and e['branch'] in range(8) for e in events[:-1]),'no unaccounted compiler branch')
    return p,plans


def seal_plan(b,row,original,expected):
    plan=deepcopy(original);plan.update(commands=b['prefix_commands']+row['new_suffix_commands'],
            parent_selection_digest=original['selection_digest'],parent_compiler_selection_digest=original['compiler_selection_digest'],
            parent_witness=original['witness'],parent_predicted_Q=original['predicted_Q'],
            feedback_branch_id=b['branch_id'],feedback_compile_report_sha256=expected,
            new_suffix_Q=row['evaluations'][1]['cost'],new_suffix_context=row['evaluations'][1]['context'],
            new_suffix_witness=dict(scope='actual-mid suffix5 only, NOT model root-to-terminal graph',
                input_context=b['observed_context'],raw5=row['new_suffix_commands'],output_context=row['evaluations'][1]['context'],
                goal=b['goal'],Q='identity192',independent_suffix_replay_exact=row['equal_new_replay'],
                compile_row_sha256=physical.saved.digest(row)),
            compiler_selection_digest=physical.saved.digest(dict(branch=b,result=row)))
    del plan['witness'];del plan['predicted_Q']  # Original imagined graph/Q describe ONLY the parent plan.
    plan['raw_float32_bits']=physical.bits(plan['commands']);plan['raw_sha256']=hashlib.sha256(bytes.fromhex(plan['raw_float32_bits'])).hexdigest()
    plan['selection_digest']=physical.saved.digest({k:v for k,v in plan.items() if k!='selection_digest'})
    return plan


class ReplayGateStore(Store):
    """Native runner unchanged. Fresh step5 triplet must match the encoded input."""
    def __init__(self,path,midpoints):super().__init__(path);self.midpoints=midpoints;self.matched=set()
    def emit(self,kind,**fields):
        out=super().emit(kind,**fields)
        if kind=='fresh_observation' and fields['receipt']['phase']=='plan' and fields['receipt']['step']==5:
            f=fields['receipt'];i=fields['branch'];mid=self.midpoints[i]
            if f['condition']==mid['condition'] and f['overlay_index']==(1 if mid['condition']=='on' else 0):
                require(all(f[k]==mid[k] for k in ('file_sha256','pixel_sha256','state','snapshot')),'exact actual mid replay gate; no approximate restore')
                self.matched.add(i)
        if kind=='operation_attempted' and fields['resource']=='public_step' and fields.get('phase')=='plan' and fields['step']>5:
            require(fields['branch'] in self.matched,'new suffix blocked before native entry until exact mid RGB/state gate')
        return out


def physical_material(compile_output,compile_sha):
    p,plans=consume_compile(compile_output,compile_sha);pp,data=physical.verify(p['physical_prepare'],p['physical_prepare_sha256'])
    pp=deepcopy(pp);pp['plans']=plans;physical.merge(pp['bindings'],p['bindings']);physical.validate_plans(plans,data)
    return p,pp,data


def run_physical(compile_output,compile_sha,output,factory,authorization):
    require(authorization==AUTH,'MAIN explicit physics before any I/O')
    started=time.monotonic()
    p,pp,data=physical_material(compile_output,compile_sha);r=physical.read(Path(compile_output)/'report.json')
    binding=dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256'],planner_report_sha256=pp['planner_report_sha256'],
        strict_receipt_sha256=pp['strict_receipt_sha256'],feedback_compile_output=str(Path(compile_output).resolve()),feedback_compile_sha256=compile_sha)
    store=ReplayGateStore(output,p['midpoints']);store.started=started
    return physical.execute(pp,data,store,factory,authorization=physical.AUTH,binding=binding,started=started)


def consume_physical(output,expected):
    output=Path(output).resolve();require(sha(output/'report.json')==expected,'external actual physical report SHA')
    r=physical.read(output/'report.json');b=r['execution_binding']
    p,pp,data=physical_material(b['feedback_compile_output'],b['feedback_compile_sha256'])
    events=physical.journal(output/'journal.jsonl',r['journal_chain_tail'])
    require(sha(output/'journal.jsonl')==r['journal_sha256'] and
        sha(output/'execution_binding.json')==r['execution_binding_sha256'] and physical.read(output/'execution_binding.json')==b,
        'entire actual execution binding and native journal')
    def image(f):
        from PIL import Image
        path=bridge.rgb.flat(output,f['png']);require(sha(path)==f['file_sha256'],'actual PNG bytes SHA')
        with Image.open(path) as im:return np.asarray(im).copy()
    strict=physical.assemble(r,events,pp,data,image)
    require(strict['status']=='engineering_verified_science_blocked','full8 clean actual native consume')
    require({f.name for f in output.iterdir()}==set(strict['PNG_files'])|{'report.json','journal.jsonl','execution_binding.json'} and
        sum(f.stat().st_size for f in output.iterdir())<=kernel.CAP,'complete native output census and cap')
    for i,mid in enumerate(p['midpoints']):
        frame=[e['receipt'] for e in events if e['kind']=='fresh_observation' and e['branch']==i and e['receipt']['phase']=='plan'
            and e['receipt']['step']==5 and e['receipt']['condition']==mid['condition'] and e['receipt']['overlay_index']==(1 if mid['condition']=='on' else 0)]
        if strict['rows'][i]['own_returned']>5:
            require(len(frame)==1 and all(frame[0][k]==mid[k] for k in ('file_sha256','pixel_sha256','state','snapshot')),'same actual encoded mid before new suffix')
    # Native authority validates every PNG/returned raw step/metric/operation;
    # additionally bind every artifact, own closure, and retain original control.
    bridge.verify_bindings(p['bindings'])
    paired=[]
    for i,(new,old,plan) in enumerate(zip(strict['rows'],p['control_rows'],p['plans'],strict=True)):
        require((old['root_slot'],old['condition'],old['arm'],old['selection_digest'])==
            (plan['root_slot'],plan['condition'],plan['arm'],plan['selection_digest']),'all8 original control identity')
        a,bm=new['final_metrics'],old['final_metrics']
        delta={k:None if a is None or bm is None or a[k] is None or bm[k] is None else float(a[k])-float(bm[k])
            for k in ('native_reward','joint_xy_error','agent_xy_error','object_xy_error','agent_velocity_error','wrapped_angle_error','native_coverage')}
        paired.append(dict(index=i,control=old,intervention=new,new_minus_control=delta,
            native_success_control=None if bm is None else bm['native_success'],native_success_new=None if a is None else a['native_success']))
    require(sha(output/'report.json')==expected,'final report unchanged')
    return dict(schema=NAME+'.physical_consume',status='paired_known_intervention_recorded_NOT_science',rows=paired,
        control_physical_report_sha256='c9e8c37984926a68bf2e4d3a92fe47737c03e719ed51917da58b72c888eb038d',
        feedback_compile_sha256=b['feedback_compile_sha256'],physical_report_sha256=expected,counts=strict['counts'],
        fixed_denominator=8,known_roots=2,total_FLOPs=None,P1_admission=False,P2_admission=False,physical_release=False)


def main():
    q=argparse.ArgumentParser(description=__doc__)
    q.add_argument('command',choices=('prepare','verify','compile','consume_compile','run_physical','consume_physical'))
    for k in ('output','prepared','prepare-sha','rgb-output','rgb-receipt','compile-output','compile-sha','report-sha','receipt'):q.add_argument('--'+k)
    q.add_argument('--authorize-host-cuda',action='store_true');q.add_argument('--authorize-physics',action='store_true');a=q.parse_args()
    if a.command=='prepare':v=write(a.output,build(a.rgb_output,a.rgb_receipt))
    elif a.command=='verify':verify(a.prepared,a.prepare_sha);v=dict(status='verified_no_execution')
    elif a.command=='compile':
        require(a.authorize_host_cuda,'MAIN host CUDA explicit');v=compile_run(a.prepared,a.prepare_sha,a.output,AUTH)
    elif a.command=='consume_compile':
        p,plans=consume_compile(a.compile_output,a.compile_sha)
        v=write(a.receipt,dict(schema=NAME+'.compile_consume',status='compiled_engineering_only',plans=plans,P2_admission=False,physical_release=False))
    elif a.command=='run_physical':
        require(a.authorize_physics,'MAIN explicit physics')
        def factory():
            import gymnasium as gym
            import stable_worldmodel  # noqa: F401
            return gym.make('swm/PushT-v1',render_mode='rgb_array',resolution=224,with_target=False,render_action=False)
        r=run_physical(a.compile_output,a.compile_sha,a.output,factory,AUTH)
        v=dict(status=r['status'],report=str(Path(a.output)/'report.json'),report_sha256=sha(Path(a.output)/'report.json'))
    else:v=write(a.receipt,consume_physical(a.output,a.report_sha))
    print(raw(v).decode(),end='',flush=True)


if __name__=='__main__':main()
