"""Eight NEW suffix terminal PNGs -> frozen native192; MAIN HOST CUDA only.

No prediction/AD/physics, no root/mid re-encoding, no scientific admission.
The parent's graph/Q are not predictions of the new actions.
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
from experiments.spectral_reachability import agent_p2_contextual_midfeedback_recompile_20261028_v1 as intervention

bridge=intervention.bridge
physical=bridge.physical
require,sha,raw=physical.require,physical.sha,physical.raw
ROOT=bridge.ROOT
NAME='agent_p2_midfeedback_terminal_rgb_20261028_v1'
AUTH='MAIN_NEW_MIDFEEDBACK_SUFFIX_TERMINAL_RGB_HOST_CUDA_ONLY'
INTERVENTION_SHA='e195749c68747f3d1d820cb62e53d0b109865c4d3c87ead9f040610c953d94ac'


def contract():
    return dict(fixed_denominator=8,known_roots=2,fresh_roots=0,checkpoint=10,
        scope='new suffix terminal only; original goal192, actual same condition; no new choice',
        max_calls=dict(encoder=8,projector=8,predict=0,action_encoder=0,pred_proj=0,AD=0),
        native_hidden=[1,257,192],latent=192,batch=1,no_dedup=True,no_padding=True,
        missing='null, never encode early terminal as checkpoint10 or replace on with off',
        mapping='new_suffix_Q/context/witness only; parent Q/graph are parent-only',
        Q='original identity192 float64 mean squared distance; no goal re-encode or fit',
        CUDA_only=True,TF32=False,autocast=False,SDP='MATH',new_physics=0,new_optimizer=0,training=False,Qfit=False,
        total_FLOPs=None,equal_FLOPs=False,wall_seconds=600,output_cap_bytes=bridge.CAP,
        final_reserve_bytes=bridge.FINAL,disk_floor_bytes=bridge.FLOOR,
        P1_admission=False,P2_admission=False,physical_release=False,scientific_pass=False)


def fixed_rows(plans,strict,old_rgb,output):
    """Pure join of complete authorities. Terminal outcomes cannot change plans."""
    require(strict['schema']==intervention.NAME+'.physical_consume' and
        strict['status']=='paired_known_intervention_recorded_NOT_science' and strict['fixed_denominator']==8
        and strict['known_roots']==2 and strict['P1_admission'] is strict['P2_admission'] is strict['physical_release'] is False,
        'entire actual new physical strict authority')
    require(len(plans)==len(strict['rows'])==len(old_rgb['rows'])==8 and
        old_rgb['schema']==bridge.NAME+'.consume' and old_rgb['status']=='actual_RGB_and_same_suffix_saved_verified_NOT_science',
        'all8 original/new arms, no subset')
    rows=[]
    for i,(plan,pair,old) in enumerate(zip(plans,strict['rows'],old_rgb['rows'],strict=True)):
        actual=pair['intervention'];control=pair['control'];key=(*physical.saved.producer.ORDER[i//2],physical.ARMS[i%2])
        require((plan['root_slot'],plan['condition'],plan['arm'])==key and
            all((x['root_slot'],x['condition'],x['arm'])==key and x['index']==i for x in (actual,control,old))
            and pair['index']==i and actual['selection_digest']==plan['selection_digest']
            and actual['compiler_selection_digest']==plan['compiler_selection_digest']
            and control['selection_digest']==old['selection_digest']==plan['parent_selection_digest'],
            'exact8 original/new plan/root/condition/arm and model-only seals')
        require('predicted_Q' not in plan and 'witness' not in plan and
            plan['selection_digest']==physical.saved.digest({k:v for k,v in plan.items() if k!='selection_digest'}),
            'new plan own seal, no parent graph/Q pretending to be new prediction')
        require(physical.bits(plan['commands'])==plan['raw_float32_bits'],'new full10 rawfloat32 bits')
        witness=plan['new_suffix_witness'];context=plan['new_suffix_context'];intervention.F.core.context(context)
        require(witness['scope']=='actual-mid suffix5 only, NOT model root-to-terminal graph'
            and witness['raw5']==plan['commands'][5:] and witness['output_context']==context and witness['goal']==plan['goal']
            and witness['Q']=='identity192' and witness['independent_suffix_replay_exact'] is True,
            'new actual-mid suffix witness/context/goal, not original imagined terminal')
        goal=bridge.latent(plan['goal']);pred=bridge.latent(context['latent_history'][-1])
        require(bridge.mse(pred,goal)==plan['new_suffix_Q'],'exact new endpoint identity-Q')
        cp=[f for f in actual['rgb_encoding_inputs'] if f['step']==10]
        require(len(cp)==(0 if 10 in actual['missing_checkpoints'] else 1),'exact explicit checkpoint10 missing denominator')
        source=None
        if cp:
            source=deepcopy(cp[0]);require(actual['own_returned']==10 and actual['full_horizon'] is True,'checkpoint10 not early terminal')
            require(Path(source['png']).name==source['png'],'flat original terminal PNG')
            source['path']=str(Path(output).resolve()/source['png'])
            require(source['condition']==plan['condition'] and source['compiler_selection_digest']==plan['compiler_selection_digest']
                and source['task_id']==plan['task_id'] and source['original_task_file']==plan['original_task_file']
                and source['original_task_sha256']==plan['original_task_sha256']
                and source['original_blind_path']==plan['blind_path'] and source['original_blind_sha256']==plan['blind_sha256']
                and source['encoder_identity']==plan['encoder_identity'] and source['encoding_performed'] is False
                and source['model_bound_by_prepare'] is True and physical.bits(source['raw_action_history10'])==
                source['raw_action_history_bits']==plan['raw_float32_bits'],'actual new full10 history/model/task/condition PNG binding')
        checkpoints=old['checkpoints'];require([c['step'] for c in checkpoints]==[0,5,10],'old full checkpoint denominator')
        rows.append(dict(index=i,root_slot=key[0],condition=key[1],arm=key[2],task_id=plan['task_id'],
            selection_digest=plan['selection_digest'],parent_selection_digest=plan['parent_selection_digest'],
            compiler_selection_digest=plan['compiler_selection_digest'],full10=plan['commands'],raw_float32_bits=plan['raw_float32_bits'],
            new_suffix_Q=plan['new_suffix_Q'],new_suffix_context=context,new_suffix_witness=witness,predicted192=pred.tolist(),goal192=goal.tolist(),
            parent_predicted_Q=plan['parent_predicted_Q'],old_fixed_actual_mid_predicted_Q=old['real_mid_same_suffix_predicted_Q'],
            old_fixed_actual_Q=checkpoints[2]['latent_Q'],physical_pair=pair,source=source,
            status='not_started' if source is not None else 'missing_physical_checkpoint10',latent=None))
    return rows


def build(physical_output,physical_report_sha,physical_receipt,physical_receipt_sha):
    require(sha(intervention.__file__)==INTERVENTION_SHA,'frozen recompile producer/strict consumer')
    output=Path(physical_output).resolve();receipt=Path(physical_receipt).resolve()
    physical.kernel.hash_value(physical_report_sha);physical.kernel.hash_value(physical_receipt_sha)
    require(sha(receipt)==physical_receipt_sha,'external actual physical receipt SHA, not future authority')
    strict=intervention.consume_physical(output,physical_report_sha)
    require(raw(strict)==raw(physical.read(receipt)),'whole actual physical strict reconstruction')
    report=physical.read(output/'report.json');b=report['execution_binding']
    p,pp,_=intervention.physical_material(b['feedback_compile_output'],b['feedback_compile_sha256'])
    old_rgb=physical.read(p['arguments']['rgb_receipt'])
    rows=fixed_rows(pp['plans'],strict,old_rgb,output)
    bindings=dict(p['bindings']);physical.merge(bindings,pp['bindings'])
    physical.merge(bindings,{str(receipt):physical_receipt_sha,str(output/'report.json'):physical_report_sha,
        b['prepared']:b['prepare_sha256']})
    compile_output=Path(b['feedback_compile_output'])
    physical.merge(bindings,{str(compile_output/'report.json'):b['feedback_compile_sha256']})
    compiled=physical.read(compile_output/'report.json')
    actual_compile_counts={axis:{k:{state:sum(r[axis][k][state] for r in compiled['rows']) for state in ('attempted','completed')}
        for k in compiled['rows'][0][axis]} for axis in ('resources','AD_API_resources','predict_API_resources')}
    for f in (*output.iterdir(),*compile_output.iterdir()):physical.merge(bindings,{str(f):sha(f)})
    for row in rows:
        if row['source'] is not None:bridge.load_pixels(row['source'])
    files=bridge.source_files()+intervention.own_files()+[
        Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]
    physical.merge(bindings,{str(f):sha(f) for f in files});physical.merge(bindings,bridge.rgb.MODELS)
    shapes=bridge.rgb.config_shapes(bridge.rgb.read(bridge.rgb.POLICY/'config.json'))
    bridge.verify_bindings(bindings)
    n=sum(row['source'] is not None for row in rows)
    return dict(schema=NAME+'.prepare',arguments=dict(physical_output=str(output),physical_report_sha=physical_report_sha,
        physical_receipt=str(receipt),physical_receipt_sha=physical_receipt_sha),contract=contract(),rows=rows,bindings=bindings,
        shapes=shapes,expected_calls={k:n if k in ('encoder','projector') else 0 for k in bridge.KINDS},
        policy=p['policy'],expected_model_hash=p['expected_model_hash'],
        inherited_compiler_contract=p['contract'],inherited_RGB_counts=p['inherited_RGB_counts'],
        inherited_original_compiler=p['inherited_compiler_resources'],inherited_original_physical=p['inherited_physical_counts'],
        actual_new_physical_counts=strict['counts'],actual_new_compiler_counts=actual_compile_counts,
        actual_new_compile_wall_seconds=compiled['elapsed_seconds'])


def write(path,value):
    path=Path(path).absolute();payload=raw(value)
    require(path.name.startswith(NAME+'_') and not path.exists() and not path.is_symlink() and len(payload)<=bridge.META_CAP
        and shutil.disk_usage(path.parent).free>=bridge.FLOOR+len(payload),'own exclusive bounded metadata')
    with path.open('xb') as f:require(f.write(payload)==len(payload),'full metadata');f.flush();os.fsync(f.fileno())
    bridge.rgb.syncdir(path.parent);return dict(path=str(path),sha256=sha(path),status=value.get('status','prepared'))


def verify(path,expected):
    require(sha(path)==expected,'external own prepare SHA');p=physical.read(path)
    require(raw(p)==raw(build(**p['arguments'])),'entire prepare reconstruction');return p


class Store(bridge.rgb.Store):
    def __init__(self,output,budget):
        require(Path(output).name.startswith(NAME+'_'),'own new flat output');super().__init__(output,budget);self.index=0
    def event(self,e):self.put(f'event_{self.index:04}.json',e);self.index+=1


def process(p,rows,store,encode,check=lambda:None):
    require(len(rows)==8,'all8 source rows including missing')
    ordinal=0
    for row in rows:
        if row['source'] is None:continue
        check();scope=dict(row=row['index'],step=10,phase='encode');row['status']='attempted'
        store.event(dict(kind='image_attempt',scope=scope,guard_image=ordinal))
        pixels=bridge.load_pixels(row['source']);row['pixels']=store.array(f'r{row["index"]}_pixels',pixels)
        z=encode(pixels,row,scope,ordinal);bridge.latent(z)
        row['latent']=store.array(f'r{row["index"]}_latent',z);row['status']='encoded'
        store.event(dict(kind='image_return',scope=scope,row=deepcopy(row)));ordinal+=1


def run(prepared,expected,output,authorization):
    require(authorization==AUTH,'MAIN explicit CUDA before I/O')
    budget=bridge.Budget();p=verify(prepared,expected);budget.check();store=Store(output,budget)
    rows=deepcopy(p['rows']);errors=[];trace=guard=None;model=None;before=after=None;source_ok=False
    store.put('binding.json',dict(prepared=str(Path(prepared).resolve()),prepare_sha256=expected));store.put('initial_roster.json',rows)
    try:
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import sdpa_kernel,SDPBackend
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
        require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        bridge.native.frozen(model);before=bridge.native.tensor_hash(model);require(before==p['expected_model_hash'],'original frozen model')
        trace=bridge.Trace(model,torch,store,store.check,p['expected_calls'])
        guard=bridge.native.Ledger(2,lambda e:store.event(dict(kind='guard_event',event=e)),
            lambda:torch.cuda.synchronize(device),lambda:not torch.is_grad_enabled())
        transform=p0.image_transform()
        def encode(pixels,row,scope,ordinal):
            guard.image=ordinal
            with physical.saved.overlay.traced_scope(trace,scope,row):
                x=p0._image_tensor(pixels,transform=transform,device=device)
                row['preprocessed']=trace.save(x,f'r{row["index"]}_preprocessed')
                bridge.same(x.detach().cpu().numpy(),bridge.pixels.preprocess_pixels(pixels),'original preprocessing exactly')
                out=model.encode({'pixels':x})['emb'];require(tuple(out.shape)==(1,1,192),'native192 only')
                return out[0,0].detach().cpu().numpy().copy()
        with trace,bridge.native.guard(model,torch,guard,errors),torch.inference_mode(),\
            torch.autocast(device_type='cuda',enabled=False),sdpa_kernel(SDPBackend.MATH):
            process(p,rows,store,encode,store.check)
    except BaseException as e:errors.append(physical.kernel.error('encode',e))
    finally:
        try:
            if model is not None:bridge.native.frozen(model);after=bridge.native.tensor_hash(model)
            bridge.verify_bindings(p['bindings']);require(sha(prepared)==expected,'prepare unchanged');budget.check();source_ok=True
        except BaseException as e:errors.append(physical.kernel.error('final_closure_model_limits',e))
        entries=[] if trace is None else trace.entries;guards=[] if guard is None else guard.entries;blocked=[] if guard is None else guard.blocked
        complete=(not errors and not blocked and source_ok and before==after==p['expected_model_hash'] and
            bridge.counts(entries)=={k:dict(attempted=v,completed=v) for k,v in p['expected_calls'].items()})
        # native seal precedes any analysis of the new encoded outputs.
        seal=dict(rows=rows,modules=entries,guards=guards,blocked=blocked,errors=errors)
        h=store.put('native_seal.json',seal,final=True)
        report=dict(schema=NAME+'.report',status='terminal_encoded_science_blocked' if complete else 'partial_or_failed',
            prepared=str(Path(prepared).resolve()),prepare_sha256=expected,contract=contract(),**seal,
            counts=bridge.counts(entries),model_hash_before=before,model_hash_after=after,model_unchanged=before is not None and before==after,
            source_verified=source_ok,native_seal_sha256=h,artifact_sha256=deepcopy(store.hashes),event_count=store.index,
            elapsed_seconds=time.monotonic()-budget.started,P1_admission=False,P2_admission=False,physical_release=False)
        store.put('report.json',report,final=True)
    require(complete,'partial retained/no retries '+str(errors))
    return dict(report=str(store.path/'report.json'),report_sha256=sha(store.path/'report.json'))


def analyze(spec,row,load):
    z=load(row['latent']) if row['latent'] is not None else None
    actual=bridge.mse(z,spec['goal192']) if z is not None else None
    old=spec['old_fixed_actual_Q'];oldpred=spec['old_fixed_actual_mid_predicted_Q'];pred=spec['new_suffix_Q']
    return dict(index=spec['index'],root_slot=spec['root_slot'],condition=spec['condition'],arm=spec['arm'],
        selection_digest=spec['selection_digest'],parent_selection_digest=spec['parent_selection_digest'],
        new_suffix_predicted_Q=pred,new_suffix_actual_Q=actual,old_fixed_actual_mid_predicted_Q=oldpred,old_fixed_actual_Q=old,
        new_minus_old_predicted_Q=pred-oldpred if oldpred is not None else None,
        new_minus_old_actual_Q=actual-old if actual is not None and old is not None else None,
        new_Q_error=pred-actual if actual is not None else None,old_fixed_Q_error=oldpred-old if oldpred is not None and old is not None else None,
        new_endpoint_MSE=bridge.mse(z,spec['predicted192']) if z is not None else None,
        latent_missing=z is None,physical_pair=spec['physical_pair'],physical_benefit_claim=False)


def validate(p,r,events,load):
    require(r['contract']==p['contract']==contract() and len(r['rows'])==8 and r['errors']==r['blocked']==[],'all8 clean terminal report')
    modules=[];guards=[];want=[];ordinal=0
    for spec,row in zip(p['rows'],r['rows'],strict=True):
        require({k:v for k,v in row.items() if k not in ('status','latent','pixels','preprocessed','modules')}==
            {k:v for k,v in spec.items() if k not in ('status','latent')},'original plan/new witness/goal/frame identity')
        if spec['source'] is None:require(row==spec,'missing not replaced or padded');continue
        require(row['status']=='encoded','available encoded once');pix=load(row['pixels'])
        bridge.same(pix,bridge.load_pixels(spec['source']),'actual new terminal PNG pixels')
        pre=load(row['preprocessed']);bridge.same(pre,bridge.pixels.preprocess_pixels(pix),'exact preprocessed input')
        z=load(row['latent']);bridge.latent(z);scope=dict(row=row['index'],step=10,phase='encode')
        mods=row['modules'];require(len(mods)==2 and [m['resource'] for m in mods]==['encoder','projector']
            and all(m['scope']==scope and m['completed'] is True and len(m['inputs'])==1 for m in mods),'full scoped actual encoder/projector')
        en,proj=mods
        physical.saved.overlay.visual_bridge(pre,load(en['inputs'][0]),load(en['output']),load(proj['inputs'][0]),load(proj['output']),z,[1,257,192])
        want.append(dict(kind='image_attempt',scope=scope,guard_image=ordinal))
        for mod in mods:
            g=dict(index=len(guards),image=ordinal,resource=mod['resource'],completed=True);guards.append(g)
            attempt={k:mod[k] for k in ('index','scope','resource')}|dict(completed=False)
            want += [dict(kind='module_attempt',entry=attempt),dict(kind='guard_event',event=dict(kind='module_attempt',entry=dict(g,completed=False))),
                dict(kind='guard_event',event=dict(kind='module_return',entry=g)),dict(kind='module_return',entry=mod)]
        modules.extend(mods);want.append(dict(kind='image_return',scope=scope,row=row));ordinal+=1
    require(modules==r['modules'] and [m['index'] for m in modules]==list(range(len(modules))) and r['guards']==guards
        and events==want and r['counts']==bridge.counts(modules)=={k:dict(attempted=v,completed=v) for k,v in p['expected_calls'].items()},
        'entire native+forbidden-API guard chronology/counts, no unknown filtered')
    return [analyze(spec,row,load) for spec,row in zip(p['rows'],r['rows'],strict=True)]


def consume(output,expected):
    output=Path(output).resolve();require(sha(output/'report.json')==expected,'external actual encoder report SHA')
    r=physical.read(output/'report.json');p=verify(r['prepared'],r['prepare_sha256'])
    require(r['schema']==NAME+'.report' and r['status']=='terminal_encoded_science_blocked' and
        r['model_unchanged'] is r['source_verified'] is True and r['model_hash_before']==r['model_hash_after']==p['expected_model_hash']
        and r['P1_admission'] is r['P2_admission'] is r['physical_release'] is False and 0<=r['elapsed_seconds']<=600,'complete frozen CUDA only')
    require({f.name for f in output.iterdir()}==set(r['artifact_sha256'])|{'report.json'} and
        sum(f.stat().st_size for f in output.iterdir())<=bridge.CAP,'complete bounded output census')
    def blob(n):return bridge.rgb.flat(output,n).read_bytes()
    for n,h in r['artifact_sha256'].items():require(hashlib.sha256(blob(n)).hexdigest()==h,'all artifact file SHA')
    require(physical.read(output/'binding.json')==dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256']) and
        physical.read(output/'initial_roster.json')==p['rows'],'original entire execution roster/binding')
    require(sha(output/'native_seal.json')==r['native_seal_sha256'] and physical.read(output/'native_seal.json')==
        {k:r[k] for k in ('rows','modules','guards','blocked','errors')},'native output sealed before saved analysis')
    rows=validate(p,r,[physical.read(output/f'event_{i:04}.json') for i in range(r['event_count'])],
        lambda e:bridge.rgb.load_array(e,blob))
    bridge.verify_bindings(p['bindings']);require(sha(output/'report.json')==expected,'final report unchanged')
    return dict(schema=NAME+'.consume',status='new_suffix_terminal_RGB_verified_NOT_science',rows=rows,report_sha256=expected,
        counts=r['counts'],fixed_denominator=8,known_roots=2,fresh_roots=0,total_FLOPs=None,
        inherited_compiler_contract=p['inherited_compiler_contract'],inherited_RGB_counts=p['inherited_RGB_counts'],
        inherited_original_compiler=p['inherited_original_compiler'],inherited_original_physical=p['inherited_original_physical'],
        actual_new_compiler_counts=p['actual_new_compiler_counts'],actual_new_compile_wall_seconds=p['actual_new_compile_wall_seconds'],
        actual_new_physical_counts=p['actual_new_physical_counts'],consume_new_neural_calls=0,consume_new_physics=0,
        P1_admission=False,P2_admission=False,physical_release=False)


def main():
    q=argparse.ArgumentParser(description=__doc__);q.add_argument('command',choices=('prepare','verify','run','consume'))
    for k in ('output','prepared','prepare-sha','physical-output','physical-report-sha','physical-receipt','physical-receipt-sha','report-sha','receipt'):q.add_argument('--'+k)
    q.add_argument('--authorize-host-cuda',action='store_true');a=q.parse_args()
    if a.command=='prepare':v=write(a.output,build(a.physical_output,a.physical_report_sha,a.physical_receipt,a.physical_receipt_sha))
    elif a.command=='verify':p=verify(a.prepared,a.prepare_sha);v=dict(status='verified_no_execution',expected_calls=p['expected_calls'])
    elif a.command=='run':
        require(a.authorize_host_cuda,'MAIN explicit CUDA flag');v=run(a.prepared,a.prepare_sha,a.output,AUTH)
    else:
        require(a.receipt,'own saved receipt required');v=write(a.receipt,consume(a.output,a.report_sha))
    print(raw(v).decode(),end='',flush=True)


if __name__=='__main__':main()
