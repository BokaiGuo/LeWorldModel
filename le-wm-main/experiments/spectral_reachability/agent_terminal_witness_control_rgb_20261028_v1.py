"""Old mixed surrogate winner four cp10 RGBs, whole-row inheritance aware.

Own authority/schema; reuses only frozen policy-free image/array/guard low-level.
MAIN HOST CUDA only, never new physics, prediction, selection or fitting.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import shutil
import time
import numpy as np
from experiments.spectral_reachability import agent_mixedmode_physical_20261028_v1 as producer
from experiments.spectral_reachability import agent_mixedmode_prefix_20261028_v1 as model
from experiments.spectral_reachability import agent_terminal_witness_terminal_rgb_20261028_v1 as frozen

NAME='agent_terminal_witness_control_rgb_20261028_v1'
AUTH='MAIN_OLD_MIXED4_CONTROL_TERMINAL_RGB_HOST_CUDA_ONLY'
ROOT=producer.ROOT
bridge=frozen.bridge
require,sha,raw,read=producer.require,producer.sha,producer.raw,producer.read
CAP,FINAL,FLOOR,WALL=frozen.CAP,frozen.FINAL,frozen.FLOOR,frozen.WALL
PHYSICAL_OUTPUT=ROOT/'results/agent_mixedmode_physical_20261028_v1_MAIN01'
PHYSICAL_REPORT_SHA='58bd3b4fcb2042e79d796f3eb26e0004f88b098c70ffdb044d1b03a3eedcc965'
SUMMARY=ROOT/'results/consume_mixedmode_physical_MAIN_20261028_v1_MAIN01.json'
SUMMARY_SHA='eb3c2752dc6ae93d99319439895c99dd136b9847a81cc83dfb8ec71b6ace9e75'
PREPARE=ROOT/'results/agent_mixedmode_physical_20261028_v1_prepare_MAIN01.json'
PREPARE_SHA='667a212745650520c7c852c1192d34e98960c260fafbbd021e1fabed47877049'
PRODUCER_SHA='e4fac2479254db07c34de5d9f88fbe0d19606b8ce2b1f8d7d5e42cbad8a042ae'
FROZEN_SHA='310eda7b2fcc91e069a917f5963a962b610e9ef239a5796d47cc2b466dbec396'


def own_files():
    return [Path(__file__).resolve(),ROOT/'tests'/f'test_{NAME}.py',ROOT/'innerdocs'/f'{NAME}.md']


def contract():
    return dict(schema=NAME+'.contract',physical_schema=producer.NAME+'.consume',fixed_denominator=4,
        known_roots=2,fresh_roots=0,checkpoint=10,original_selection='stage+ridge beta1 over fixed mixed222; never reselect',
        routes=['execute','execute','inherit','inherit'],inheritance='whole authenticated original allmode row and its PNG directory; no splice',
        physical_report_sha256=PHYSICAL_REPORT_SHA,summary_sha256=SUMMARY_SHA,physical_prepare_sha256=PREPARE_SHA,
        Q='original identity192 float64 mean squared distance, original goal192',prediction='original mixed winner nonlinear terminal, NOT objective',
        max_calls=dict(encoder=4,projector=4,predict=0,action_encoder=0,pred_proj=0,AD=0),
        CUDA_only=True,TF32=False,autocast=False,SDP='MATH',batch=1,hidden=[1,257,192],latent=192,
        no_padding=True,no_dedup=True,retries=0,training=False,Qfit=False,new_physics=0,new_prediction=0,
        global_patch=False,guard='frozen instance guard + unwrapped thread-local public AD probe',
        output_cap_bytes=CAP,final_reserve_bytes=FINAL,disk_floor_bytes=FLOOR,wall_seconds=WALL,
        total_FLOPs=None,equal_FLOPs=False,P1_admission=False,scientific_admission=False,freshness=False)


def validate_summary(summary,strict):
    require(summary['schema']=='MAIN.mixedmode_physical_receipt.v1' and
        raw(summary['receipt'])==raw(strict),'entire real internal oldmixed.consume, not wrapper status/caveat')


def raw_bits(a):return producer.raw10(a).tobytes()


def fixed_rows(prep,strict,compiled,model_report,oldprep,oldcompiled,output):
    require(prep['schema']==producer.NAME+'.prepare' and strict['schema']==producer.NAME+'.consume' and
        strict['status']==producer.RECORDED and strict['errors']==[] and strict['fixed_denominator']==4 and
        strict['known_roots']==2 and strict['fresh_roots']==0 and strict['inherited']==2 and
        strict['P1_admission'] is strict['scientific_admission'] is strict['freshness'] is False and
        strict['physical_report_sha256']==PHYSICAL_REPORT_SHA and strict['producer_report_sha256']==producer.REPORT_SHA,
        'actual old mixed full4 native strict authority, no terminal-witness schema projection')
    old=prep['old_physical_consume']
    require(old['schema']==producer.OLD+'.consume' and
        old['physical_report_sha256']==strict['old_report_sha256']==prep['arguments']['old_report_sha256']==producer.OLD_REPORT_SHA and
        Path(prep['arguments']['old_output']).is_absolute(),'actual old allmode report and directory authority')
    require(all(len(x)==4 for x in (prep['plans'],prep['routes'],strict['rows'],compiled['slots'],model_report['rows'],
        old['rows'],oldcompiled['slots'])),'all4 fixed plans/tasks/original evidence, no subset')
    rows=[]
    for i,(plan,route,actual,slot,mr,oldslot) in enumerate(zip(prep['plans'],prep['routes'],strict['rows'],compiled['slots'],model_report['rows'],oldcompiled['slots'],strict=True)):
        key=producer.ORDER[i];root=key[0]
        require(slot['index']==mr['index']==actual['index']==route['index']==route['old_index']==i and
            (plan['root_slot'],plan['condition'])==(slot['root_slot'],slot['condition'])==
            (actual['root_slot'],actual['condition'])==key and plan['arm']==actual['arm']==producer.ARM and
            raw(actual['plan'])==raw(plan) and all(plan[k]==slot[k] for k in ('task_id','task_sha256','blind_sha256')) and
            actual['task_id']==plan['task_id'] and actual['selection_digest']==plan['selection_digest'],
            'fixed ordered mixed plan/raw/task/condition identity')
        require(plan['schema']==producer.PRODUCER+'.model_plan' and plan['whole_run_report_sha256']==producer.REPORT_SHA and
            plan['selection_digest']==producer.core.digest({k:v for k,v in plan.items() if k not in ('selection_digest','whole_run_report_sha256')}) and
            raw(mr['plan'])==raw({k:v for k,v in plan.items() if k!='whole_run_report_sha256'}) and
            plan['selection_uses_physics'] is plan['P1_admission'] is False and plan['raw_steps']==10 and
            raw_bits(plan['commands'])==raw_bits(plan['prefix_commands']+plan['suffix_commands']),
            'original model-only surrogate plan seal and literal raw witness')
        require(raw(slot)==raw(oldslot) and raw(prep['source_roots'][root])==raw(oldprep['source_roots'][root]),
            'same exact original encoded task/context/goal/Q and native root/goal source')
        result=mr['result'];cs=result['candidates'];winner=result['winner'];j=plan['selected_index']
        require(len(cs)==222 and [c['index'] for c in cs]==list(range(222)) and
            raw(winner)==raw(min(cs,key=lambda c:(c['objective'],c['index']))) and winner['index']==j and
            raw_bits(winner['commands'])==raw_bits(plan['commands']) and winner['objective']==plan['objective'],
            'original stable stage+ridge winner over all222, not terminal or outcome selection')
        model.base.validate_q(slot['q']);goal=bridge.latent(slot['packet']['goal']);pred=bridge.latent(winner['predicted_terminal'])
        require(bridge.mse(pred,goal)==winner['nonlinear_terminal_cost']==plan['model_Q'],
            'original nonlinear predicted endpoint Q, not surrogate objective')
        mode=route['mode'];require(mode==actual['route']==('execute' if i<2 else 'inherit'),'exact fixed certified route')
        if mode=='inherit':
            evidence=old['rows'][i]
            require(actual['status']=='inherited_saved_physical' and raw(actual['inherited_evidence'])==raw(evidence) and
                actual['trajectory']==actual['checkpoints']==[] and actual['own_returned']==0 and
                all(raw(actual[k])==raw(evidence[k]) for k in ('final_state','final_metrics','native_success','terminated','truncated')) and
                raw_bits(evidence['plan']['commands'])==raw_bits(plan['commands']) and
                all(evidence['plan'][k]==plan[k] for k in ('task_id','task_sha256','blind_sha256','source_report_sha256')),
                'whole original inherited row only; no partial trajectory/PNG or action splice')
            directory=Path(prep['arguments']['old_output'])
        else:
            require(actual['inherited_evidence'] is None,'new physical row cannot borrow old PNG')
            evidence=actual;directory=Path(output).resolve()
        require(evidence['status'] in ('horizon_completed','native_stopped') and
            evidence['index']==i and (evidence['root_slot'],evidence['condition'])==key and
            evidence['task_id']==plan['task_id'],'actual whole source row in same slot/condition')
        if evidence['initial_exact']:
            meta=prep['source_roots'][root]
            require(all(raw(evidence[k])==raw(meta[k]) for k in ('root_state','goal','root_sha256')),
                'actual native root and physical task goal, not latent/renderer proxy')
        steps=[f['step'] for f in evidence['checkpoints']]
        require(len(set(steps))==len(steps) and sorted(steps)==[s for s in (0,5,10) if s not in evidence['missing_checkpoints']] and
            sorted(evidence['missing_checkpoints'])==[s for s in (0,5,10) if s not in steps],
            'whole source actual checkpoint and missing denominator')
        cp=[f for f in evidence['checkpoints'] if f['step']==10];source=None
        if cp:
            f=cp[0];trajectory=[t for t in evidence['trajectory'] if t['phase']=='plan' and t['step']==10]
            require(evidence['own_returned']==10 and evidence['initial_exact'] is True and len(trajectory)==1 and
                trajectory[0]['condition']==key[1] and trajectory[0]['overlay_index']==(1 if key[1]=='on' else 0) and
                all(raw(trajectory[0][k])==raw(f[k]) for k in ('step','png','file_sha256','pixel_sha256','state','snapshot')) and
                raw(f['state'])==raw(evidence['final_state']) and Path(f['png']).name==f['png'] and f['png'] not in ('','.','..'),
                'actual source cp10, condition/state/snapshot/file joins, never early padding')
            source=dict(deepcopy(f),path=str(directory/f['png']))
        rows.append(dict(index=i,root_slot=root,condition=key[1],arm=plan['arm'],task_id=plan['task_id'],task_sha256=plan['task_sha256'],
            blind_sha256=plan['blind_sha256'],selection_digest=plan['selection_digest'],selected_index=j,
            full10=deepcopy(plan['commands']),raw_float32_sha256=hashlib.sha256(raw_bits(plan['commands'])).hexdigest(),
            context_sha256=frozen.selector.sha_object(slot['packet']['root_context']),goal_sha256=frozen.selector.sha_object(slot['packet']['goal']),
            q_sha256=model.base.digest(slot['q']),goal192=goal.tolist(),predicted192=pred.tolist(),predicted_Q=plan['model_Q'],
            original_objective=plan['objective'],model_prefix_uphill_from_anchor=winner['stage_cost']>model.base.cost(
                slot['packet']['root_context']['latent_history'][-1],goal,slot['q']),source_route=mode,
            physical_row_status=actual['status'],source_row_status=evidence['status'],source_row_sha256=frozen.selector.sha_object(evidence),
            source_physical_report_sha256=PHYSICAL_REPORT_SHA if mode=='execute' else producer.OLD_REPORT_SHA,
            evaluation={k:deepcopy(evidence[k]) for k in ('final_state','final_metrics','native_success','terminated','truncated')},
            source=source,status='not_started' if source is not None else 'missing_physical_checkpoint10',latent=None))
    return rows


def build():
    require(sha(producer.__file__)==PRODUCER_SHA and sha(frozen.__file__)==FROZEN_SHA,'frozen original mixed adapter and image low-level')
    require(sha(SUMMARY)==SUMMARY_SHA and sha(PREPARE)==PREPARE_SHA,'actual MAIN summary and mixed physical preparation')
    strict=producer.consume(PHYSICAL_OUTPUT,PHYSICAL_REPORT_SHA)
    validate_summary(read(SUMMARY,producer.PREP_CAP),strict)
    report=read(PHYSICAL_OUTPUT/'report.json');binding=report['execution_binding']
    require(Path(binding['prepared'])==PREPARE and binding['prepare_sha256']==PREPARE_SHA,'actual mixed execution prepare binding')
    p=read(PREPARE,producer.PREP_CAP);args=p['arguments']
    compiled=read(args['producer_prepared']);model_report=read(Path(args['producer_output'])/'report.json',model.CAP)
    oldout=Path(args['old_output']);oldreport=read(oldout/'report.json',producer.CAP);oldbinding=oldreport['execution_binding']
    require(sha(oldout/'report.json')==producer.OLD_REPORT_SHA and
        sha(oldbinding['prepared'])==oldbinding['prepare_sha256']==strict['bindings'][oldbinding['prepared']],
        'actual authenticated old allmode whole-row authority')
    oldp=read(oldbinding['prepared']);oldcompiled=read(oldp['arguments']['producer_prepared'])
    rows=fixed_rows(p,strict,compiled,model_report,oldp,oldcompiled,PHYSICAL_OUTPUT)
    bindings=deepcopy(strict['bindings'])
    producer.merge(bindings,{str(SUMMARY):SUMMARY_SHA,str(PREPARE):PREPARE_SHA,str(Path(frozen.__file__).resolve()):FROZEN_SHA})
    producer.merge(bindings,bridge.rgb.MODELS)
    files=own_files()+frozen.own_files()+[Path(bridge.__file__).resolve(),Path(bridge.rgb.__file__).resolve(),
        Path(bridge.pixels.__file__).resolve(),Path(bridge.native.__file__).resolve(),Path(bridge.api.__file__).resolve()]
    producer.merge(bindings,{str(f):sha(f) for f in files})
    model.kernel.validate_normalizer_source(ROOT)
    require(Path(compiled['policy']).resolve()==bridge.rgb.POLICY.resolve() and
        compiled['expected_model_hash']==bridge.rgb.MODEL_TENSOR_SHA,'same frozen original encoder/projector and representation')
    for row in rows:
        if row['source'] is not None:bridge.load_pixels(row['source'])
    bridge.verify_bindings(bindings);n=sum(r['source'] is not None for r in rows)
    return dict(schema=NAME+'.prepare',contract=contract(),rows=rows,bindings=bindings,policy=compiled['policy'],
        expected_model_hash=compiled['expected_model_hash'],shapes=bridge.rgb.config_shapes(bridge.rgb.read(bridge.rgb.POLICY/'config.json')),
        expected_calls={k:n if k in ('encoder','projector') else 0 for k in bridge.KINDS},
        action_mean=model.kernel.ACTION_MEAN.tolist(),action_std=model.kernel.ACTION_STD.tolist(),
        original_mixed_physical_counts=strict['counts'],inherited_allmode_physical_counts=p['old_physical_consume']['counts'],
        original_mixed_model_resources=model_report['resources'],original_mixed_math_resources=model_report['math_resources'])


def write(path,value):
    path=Path(path).absolute();payload=raw(value)
    require(path.name.startswith(NAME+'_') and not path.exists() and not path.is_symlink() and
        len(payload)<=producer.PREP_CAP and shutil.disk_usage(path.parent).free>=FLOOR+len(payload),'new own bounded metadata only')
    with path.open('xb') as f:require(f.write(payload)==len(payload),'full metadata');f.flush();os.fsync(f.fileno())
    bridge.rgb.syncdir(path.parent);return dict(path=str(path),sha256=sha(path),status=value.get('status','prepared'))


def verify(path,expected):
    require(sha(path)==expected,'external actual own prepare SHA');p=read(path,producer.PREP_CAP)
    require(raw(p)==raw(build()),'entire original-route preparation reconstructed');return p


class Store(bridge.rgb.Store):
    def __init__(self,output,budget):
        require(Path(output).name.startswith(NAME+'_') and shutil.disk_usage(Path(output).parent).free>=CAP+FLOOR,'new own full-cap output')
        super().__init__(output,budget);self.index=0
    def event(self,e):self.put(f'event_{self.index:04}.json',e);self.index+=1


process=frozen.process  # Policy/schema-free all4 encoder dispatcher; no authority conversion.


def run(prepared,expected,output,authorization):
    require(authorization==AUTH,'MAIN explicit HOST CUDA before I/O; no CPU fallback')
    budget=bridge.Budget();p=verify(prepared,expected);budget.check();store=Store(output,budget)
    rows=deepcopy(p['rows']);errors=[];trace=guard=None;model_obj=None;before=after=None;source_ok=False;flags=None
    store.put('binding.json',dict(prepared=str(Path(prepared).resolve()),prepare_sha256=expected));store.put('initial_roster.json',rows)
    try:
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import sdpa_kernel,SDPBackend
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
        require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        flags=torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model_obj=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        bridge.native.frozen(model_obj);before=bridge.native.tensor_hash(model_obj);require(before==p['expected_model_hash'],'original frozen model')
        trace=bridge.Trace(model_obj,torch,store,store.check,p['expected_calls'])
        guard=bridge.native.Ledger(1,lambda e:store.event(dict(kind='guard_event',event=e)),
            lambda:torch.cuda.synchronize(device),lambda:not torch.is_grad_enabled())
        transform=p0.image_transform()
        def encode(pix,row,scope,ordinal):
            guard.image=ordinal
            with bridge.physical.saved.overlay.traced_scope(trace,scope,row):
                x=p0._image_tensor(pix,transform=transform,device=device)
                row['preprocessed']=trace.save(x,f'r{row["index"]}_preprocessed')
                bridge.same(x.detach().cpu().numpy(),bridge.pixels.preprocess_pixels(pix),'independent original pixels->preprocessing')
                out=model_obj.encode({'pixels':x})['emb'];require(tuple(out.shape)==(1,1,192),'original native192 only')
                return out[0,0].detach().cpu().numpy().copy()
        with trace,frozen.encode_guard(model_obj,torch,guard,errors),torch.inference_mode(),\
            torch.autocast(device_type='cuda',enabled=False),sdpa_kernel(SDPBackend.MATH):
            process(p,rows,store,encode,store.check)
    except BaseException as e:errors.append(bridge.physical.kernel.error('control_encode',e))
    finally:
        if flags is not None:torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=flags
        try:
            if model_obj is not None:bridge.native.frozen(model_obj);after=bridge.native.tensor_hash(model_obj)
            bridge.verify_bindings(p['bindings']);require(sha(prepared)==expected,'prepare unchanged');budget.check();source_ok=True
        except BaseException as e:errors.append(bridge.physical.kernel.error('final_closure_model_limits',e))
        entries=[] if trace is None else trace.entries;guards=[] if guard is None else guard.entries;blocked=[] if guard is None else guard.blocked
        complete=(not errors and not blocked and source_ok and before==after==p['expected_model_hash'] and
            bridge.counts(entries)=={k:dict(attempted=v,completed=v) for k,v in p['expected_calls'].items()})
        seal=dict(rows=rows,modules=entries,guards=guards,blocked=blocked,errors=errors)
        h=store.put('native_seal.json',seal,final=True)
        report=dict(schema=NAME+'.report',status='old_mixed4_control_encoded_not_admitted' if complete else 'partial_or_failed',
            prepared=str(Path(prepared).resolve()),prepare_sha256=expected,contract=contract(),**seal,counts=bridge.counts(entries),
            model_hash_before=before,model_hash_after=after,model_unchanged=before is not None and before==after,
            source_verified=source_ok,native_seal_sha256=h,artifact_sha256=deepcopy(store.hashes),event_count=store.index,
            elapsed_seconds=time.monotonic()-budget.started,P1_admission=False,scientific_admission=False,freshness=False)
        store.put('report.json',report,final=True)
    require(complete,'partial evidence retained, no retry '+str(errors))
    return dict(report=str(store.path/'report.json'),report_sha256=sha(store.path/'report.json'))


def analyze(spec,row,load):
    out=frozen.analyze(spec,row,load)  # Pure arithmetic row ABI, not a receipt or schema adapter.
    out.update(original_objective=spec['original_objective'],source_route=spec['source_route'],
        source_row_sha256=spec['source_row_sha256'],source_physical_report_sha256=spec['source_physical_report_sha256'])
    return out


def validate(p,r,events,load):
    require(p['schema']==NAME+'.prepare' and r['schema']==NAME+'.report' and p['contract']==r['contract']==contract() and
        len(p['rows'])==len(r['rows'])==4 and r['errors']==r['blocked']==[],'own control schemas and full4 clean report')
    modules=[];guards=[];want=[];ordinal=0
    for spec,row in zip(p['rows'],r['rows'],strict=True):
        require(raw({k:v for k,v in row.items() if k not in ('status','latent','pixels','preprocessed','modules')})==
            raw({k:v for k,v in spec.items() if k not in ('status','latent')}),'entire original route/raw/task/goal/frame identity')
        if spec['source'] is None:require(row==spec,'genuinely missing source only, never padded');continue
        require(row['status']=='encoded','available encoded exactly once');pix=load(row['pixels'])
        bridge.same(pix,bridge.load_pixels(spec['source']),'actual original source cp10 PNG pixels')
        pre=load(row['preprocessed']);bridge.same(pre,bridge.pixels.preprocess_pixels(pix),'independent original pixels->preprocessing')
        z=load(row['latent']);bridge.latent(z);scope=dict(row=row['index'],step=10,phase='encode')
        mods=row['modules'];require(len(mods)==2 and [m['resource'] for m in mods]==['encoder','projector'] and
            all(m['scope']==scope and m['completed'] is True and len(m['inputs'])==1 for m in mods),'full native chain')
        en,proj=mods;ein=load(en['inputs'][0]);eout=load(en['output']);pin=load(proj['inputs'][0]);pout=load(proj['output'])
        bridge.same(ein,pre.reshape(1,3,224,224),'exact encoder input bits');require(eout.shape==(1,257,192),'original full hidden257')
        bridge.same(pin,eout[:,0],'original CLS bridge');require(pout.shape==(1,192),'original projector shape')
        bridge.same(pout[0],z,'original projector->latent bits');want.append(dict(kind='image_attempt',scope=scope,guard_image=ordinal))
        for mod in mods:
            g=dict(index=len(guards),image=ordinal,resource=mod['resource'],completed=True);guards.append(g)
            attempt={k:mod[k] for k in ('index','scope','resource')}|dict(completed=False)
            want += [dict(kind='module_attempt',entry=attempt),dict(kind='guard_event',event=dict(kind='module_attempt',entry=dict(g,completed=False))),
                dict(kind='guard_event',event=dict(kind='module_return',entry=g)),dict(kind='module_return',entry=mod)]
        modules.extend(mods);want.append(dict(kind='image_return',scope=scope,row=row));ordinal+=1
    require(modules==r['modules'] and [m['index'] for m in modules]==list(range(len(modules))) and r['guards']==guards and
        events==want and r['counts']==bridge.counts(modules)=={k:dict(attempted=v,completed=v) for k,v in p['expected_calls'].items()},
        'exact entire event/array/guard chronology/counts, unknown events never filtered')
    return [analyze(spec,row,load) for spec,row in zip(p['rows'],r['rows'],strict=True)]


def consume(output,expected):
    output=Path(output).resolve();require(sha(output/'report.json')==expected,'external actual control encoder report SHA')
    r=read(output/'report.json');p=verify(r['prepared'],r['prepare_sha256'])
    require(r['schema']==NAME+'.report' and r['status']=='old_mixed4_control_encoded_not_admitted' and
        r['model_unchanged'] is r['source_verified'] is True and r['model_hash_before']==r['model_hash_after']==p['expected_model_hash'] and
        r['P1_admission'] is r['scientific_admission'] is r['freshness'] is False and 0<=r['elapsed_seconds']<=WALL,'complete original model CUDA report')
    require({f.name for f in output.iterdir()}==set(r['artifact_sha256'])|{'report.json'} and
        sum(f.stat().st_size for f in output.iterdir())<=CAP,'whole output census/cap')
    def blob(n):return bridge.rgb.flat(output,n).read_bytes()
    for n,h in r['artifact_sha256'].items():require(hashlib.sha256(blob(n)).hexdigest()==h,'every actual artifact SHA')
    require(read(output/'binding.json')==dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256']) and
        read(output/'initial_roster.json')==p['rows'],'entire original roster and execution binding')
    require(sha(output/'native_seal.json')==r['native_seal_sha256'] and read(output/'native_seal.json')==
        {k:r[k] for k in ('rows','modules','guards','blocked','errors')},'native seal before saved evaluation')
    rows=validate(p,r,[read(output/f'event_{i:04}.json') for i in range(r['event_count'])],lambda a:bridge.rgb.load_array(a,blob))
    bridge.verify_bindings(p['bindings']);require(sha(output/'report.json')==expected,'final report unchanged')
    return dict(schema=NAME+'.consume',status='old_mixed4_control_RGB_verified_not_admitted',rows=rows,
        report_sha256=expected,physical_report_sha256=PHYSICAL_REPORT_SHA,physical_summary_sha256=SUMMARY_SHA,
        counts=r['counts'],fixed_denominator=4,known_roots=2,fresh_roots=0,whole_rows_inherited=2,total_FLOPs=None,
        **{k:p[k] for k in ('original_mixed_physical_counts','inherited_allmode_physical_counts',
            'original_mixed_model_resources','original_mixed_math_resources')},
        consume_new_neural_calls=0,consume_new_physics=0,P1_admission=False,scientific_admission=False,freshness=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','verify','run','consume'))
    for k in ('output','prepared','prepare-sha','report-sha','receipt'):parser.add_argument('--'+k)
    parser.add_argument('--authorize-host-cuda',action='store_true');a=parser.parse_args(argv)
    if a.command=='prepare':v=write(a.output,build())
    elif a.command=='verify':p=verify(a.prepared,a.prepare_sha);v=dict(status='verified_no_execution',expected_calls=p['expected_calls'])
    elif a.command=='run':
        require(a.authorize_host_cuda,'MAIN host CUDA flag');v=run(a.prepared,a.prepare_sha,a.output,AUTH)
    else:
        require(a.receipt,'new own receipt required');v=write(a.receipt,consume(a.output,a.report_sha))
    print(raw(v).decode(),end='',flush=True);return 0


if __name__=='__main__':raise SystemExit(main())
