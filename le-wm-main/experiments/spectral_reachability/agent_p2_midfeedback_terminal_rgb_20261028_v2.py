"""Terminal8 RGB for safeguarded compiler v2; MAIN HOST CUDA only.

Shares frozen v1 tensor/encode validators, NOT v1 authority or compiler. No
dedup of incumbent selections; no predictions, AD, physics or policy changes.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import shutil
import time
from experiments.spectral_reachability import agent_p2_midfeedback_terminal_rgb_20261028_v1 as core
from experiments.spectral_reachability import agent_p2_contextual_midfeedback_recompile_20261028_v2 as intervention

NAME='agent_p2_midfeedback_terminal_rgb_20261028_v2'
AUTH='MAIN_SAFEGUARDED_V2_TERMINAL_RGB_HOST_CUDA_ONLY'
ROOT=core.ROOT
bridge,physical=core.bridge,core.physical
require,sha,raw=core.require,core.sha,core.raw
INTERVENTION_SHA='e459cdc5de507353873346d6f86bc223afed291982904c4c4c6ce9bdfb7eaeab'
CORE_SHA='e7bcd2d9a4f0920eb4c715ddd0d095b22d4d6ba3b2b1bd6c49599836b6ab68ac'


def contract():
    c=core.contract()
    c.update(compiler_schema=intervention.NAME+'.compile',physical_schema=intervention.NAME+'.physical_consume',
        scope='safeguarded v2 selected suffix terminal8, including incumbent selections; no dedup/no new selection',
        incumbent_selected='encode independently even when commands match parent; never reuse old latent',
        safeguard='model-Q only, NOT true latent-Q/ranking/native-task guarantee')
    return c


def own_files():return [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]


def fixed_rows(plans,strict,old_rgb,output):
    require(strict['schema']==intervention.NAME+'.physical_consume','v2 physical strict authority required, v1 rejected')
    require(len(plans)==len(strict['rows'])==8,'all8 v2 plans, no selection/missing dedup')
    for plan,pair,old in zip(plans,strict['rows'],old_rgb['rows'],strict=True):
        s=plan['safeguard']
        require(s==pair['safeguard'] and type(s['selected_index']) is int and s['selected_index'] in range(5)
            and s['no_modeled_worsening'] is True and s['domain']['legal'] is True
            and s['chart']==intervention.contract()['chart'] and s['selected_Q']==plan['new_suffix_Q']<=s['incumbent_Q']
            and s['incumbent_Q']==old['real_mid_same_suffix_predicted_Q'],'exact v2 selected/old model-only guard identity')
    # Pure row-ABI compatibility view only. The actual v2 authority is checked
    # above and independently reconstructed in build; its SHA/schema never
    # changes. The frozen frame/tensor validator is not an authority issuer.
    view=dict(strict,schema=core.intervention.NAME+'.physical_consume')
    rows=core.fixed_rows(plans,view,old_rgb,output)
    for row,plan in zip(rows,plans,strict=True):
        row['safeguard']=deepcopy(plan['safeguard'])
        row['selected_index']=plan['safeguard']['selected_index']
        row['incumbent_selected']=row['selected_index']==0
    return rows


def build(physical_output,physical_report_sha,physical_receipt,physical_receipt_sha):
    require(sha(intervention.__file__)==INTERVENTION_SHA and sha(core.__file__)==CORE_SHA,'both frozen implementations, no globals patched')
    output=Path(physical_output).resolve();receipt=Path(physical_receipt).resolve()
    physical.kernel.hash_value(physical_report_sha);physical.kernel.hash_value(physical_receipt_sha)
    require(sha(receipt)==physical_receipt_sha,'external actual v2 physical receipt, no future SHA')
    strict=intervention.consume_physical(output,physical_report_sha)
    require(raw(strict)==raw(physical.read(receipt)),'whole actual v2 physical strict reconstruction')
    report=physical.read(output/'report.json');b=report['execution_binding']
    p,pp,_=intervention.physical_material(b['feedback_compile_output'],b['feedback_compile_sha256'])
    require(p['schema']==intervention.NAME+'.prepare' and p['contract']==intervention.contract(),'actual safeguarded v2 compiler prepare')
    rows=fixed_rows(pp['plans'],strict,physical.read(p['arguments']['rgb_receipt']),output)
    bindings=dict(p['bindings']);physical.merge(bindings,pp['bindings'])
    physical.merge(bindings,{str(receipt):physical_receipt_sha,str(output/'report.json'):physical_report_sha,b['prepared']:b['prepare_sha256']})
    compiled_output=Path(b['feedback_compile_output']);compiled=physical.read(compiled_output/'report.json')
    require(compiled['schema']==intervention.NAME+'.compile' and compiled['status']=='all8_safeguarded_science_blocked'
        and len(compiled['rows'])==8,'v2 selected predictions, not v1 unconditional solve')
    physical.merge(bindings,{str(compiled_output/'report.json'):b['feedback_compile_sha256']})
    counts={axis:{k:{state:sum(r[axis][k][state] for r in compiled['rows']) for state in ('attempted','completed')}
        for k in compiled['rows'][0][axis]} for axis in ('resources','AD_API_resources','predict_API_resources')}
    for f in (*output.iterdir(),*compiled_output.iterdir()):physical.merge(bindings,{str(f):sha(f)})
    for row in rows:
        if row['source'] is not None:bridge.load_pixels(row['source'])
    files=bridge.source_files()+intervention.own_files()+core.intervention.own_files()+own_files()+[
        Path(core.__file__).resolve(),ROOT/'tests'/('test_'+core.NAME+'.py'),ROOT/'innerdocs'/(core.NAME+'.md')]
    physical.merge(bindings,{str(f):sha(f) for f in files});physical.merge(bindings,bridge.rgb.MODELS)
    shapes=bridge.rgb.config_shapes(bridge.rgb.read(bridge.rgb.POLICY/'config.json'));bridge.verify_bindings(bindings)
    n=sum(r['source'] is not None for r in rows)
    return dict(schema=NAME+'.prepare',arguments=dict(physical_output=str(output),physical_report_sha=physical_report_sha,
        physical_receipt=str(receipt),physical_receipt_sha=physical_receipt_sha),contract=contract(),rows=rows,bindings=bindings,
        shapes=shapes,expected_calls={k:n if k in ('encoder','projector') else 0 for k in bridge.KINDS},
        policy=p['policy'],expected_model_hash=p['expected_model_hash'],inherited_compiler_contract=p['contract'],
        inherited_RGB_counts=p['inherited_RGB_counts'],inherited_original_compiler=p['inherited_compiler_resources'],
        inherited_original_physical=p['inherited_physical_counts'],actual_new_physical_counts=strict['counts'],
        actual_new_compiler_counts=counts,actual_new_compile_wall_seconds=compiled['elapsed_seconds'])


def write(path,value):
    path=Path(path).absolute();payload=raw(value)
    require(path.name.startswith(NAME+'_') and not path.exists() and not path.is_symlink() and len(payload)<=bridge.META_CAP
        and shutil.disk_usage(path.parent).free>=bridge.FLOOR+len(payload),'own new bounded metadata')
    with path.open('xb') as f:require(f.write(payload)==len(payload),'full metadata');f.flush();os.fsync(f.fileno())
    bridge.rgb.syncdir(path.parent);return dict(path=str(path),sha256=sha(path),status=value.get('status','prepared'))


def verify(path,expected):
    require(sha(path)==expected,'own v2 prepare SHA');p=physical.read(path)
    require(raw(p)==raw(build(**p['arguments'])),'whole v2 preparation reconstructed');return p


class Store(core.Store):
    def __init__(self,output,budget):
        require(Path(output).name.startswith(NAME+'_'),'new v2 output only')
        bridge.rgb.Store.__init__(self,output,budget);self.index=0


process=core.process  # Policy-free per-row encoder dispatcher, no schema global.


def run(prepared,expected,output,authorization):
    require(authorization==AUTH,'MAIN explicit CUDA before I/O')
    budget=bridge.Budget();p=verify(prepared,expected);budget.check();store=Store(output,budget)
    rows=deepcopy(p['rows']);errors=[];trace=guard=None;model=None;before=after=None;source_ok=False;flags=None
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
                bridge.same(x.detach().cpu().numpy(),bridge.pixels.preprocess_pixels(pixels),'exact original preprocessing')
                out=model.encode({'pixels':x})['emb'];require(tuple(out.shape)==(1,1,192),'native192 only')
                return out[0,0].detach().cpu().numpy().copy()
        with trace,bridge.native.guard(model,torch,guard,errors),torch.inference_mode(),\
            torch.autocast(device_type='cuda',enabled=False),sdpa_kernel(SDPBackend.MATH):
            process(p,rows,store,encode,store.check)
    except BaseException as e:errors.append(physical.kernel.error('encode',e))
    finally:
        if flags is not None:torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=flags
        try:
            if model is not None:bridge.native.frozen(model);after=bridge.native.tensor_hash(model)
            bridge.verify_bindings(p['bindings']);require(sha(prepared)==expected,'prepare unchanged');budget.check();source_ok=True
        except BaseException as e:errors.append(physical.kernel.error('final_closure_model_limits',e))
        entries=[] if trace is None else trace.entries;guards=[] if guard is None else guard.entries;blocked=[] if guard is None else guard.blocked
        complete=(not errors and not blocked and source_ok and before==after==p['expected_model_hash'] and
            bridge.counts(entries)=={k:dict(attempted=v,completed=v) for k,v in p['expected_calls'].items()})
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


def validate(p,r,events,load):
    require(p['schema']==NAME+'.prepare' and r['schema']==NAME+'.report' and r['contract']==p['contract']==contract(),'exact v2 encoder schemas/contract')
    require(len(p['rows'])==8 and all(row['selected_index']==row['safeguard']['selected_index'] and
        row['incumbent_selected']==(row['selected_index']==0) and row['safeguard']['selected_Q']==row['new_suffix_Q']
        <=row['safeguard']['incumbent_Q'] and row['safeguard']==row['physical_pair']['safeguard'] for row in p['rows']),
        'selected v2 model-only identity survives native encoding')
    # Shared tensor validator has its own v1 contract constant. Project only
    # that constant into non-mutating local views AFTER exact v2 checks. All
    # original rows, events, native arrays and counters are passed unchanged.
    result=core.validate(dict(p,contract=core.contract()),dict(r,contract=core.contract()),events,load)
    for row,spec in zip(result,p['rows'],strict=True):
        row.update(safeguard=deepcopy(spec['safeguard']),selected_index=spec['selected_index'],
            incumbent_selected=spec['incumbent_selected'],model_safeguard_guarantees_actual_Q=False,
            model_safeguard_guarantees_prediction_ranking=False,model_safeguard_guarantees_native_task=False)
    return result


def consume(output,expected):
    output=Path(output).resolve();require(sha(output/'report.json')==expected,'actual v2 encoder report SHA')
    r=physical.read(output/'report.json');p=verify(r['prepared'],r['prepare_sha256'])
    require(r['schema']==NAME+'.report' and r['status']=='terminal_encoded_science_blocked'
        and r['model_unchanged'] is r['source_verified'] is True and r['model_hash_before']==r['model_hash_after']==p['expected_model_hash']
        and r['P1_admission'] is r['P2_admission'] is r['physical_release'] is False and 0<=r['elapsed_seconds']<=600,'whole frozen v2 CUDA only')
    require({f.name for f in output.iterdir()}==set(r['artifact_sha256'])|{'report.json'} and
        sum(f.stat().st_size for f in output.iterdir())<=bridge.CAP,'whole output census/cap')
    def blob(n):return bridge.rgb.flat(output,n).read_bytes()
    for n,h in r['artifact_sha256'].items():require(hashlib.sha256(blob(n)).hexdigest()==h,'every actual artifact SHA')
    require(physical.read(output/'binding.json')==dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256']) and
        physical.read(output/'initial_roster.json')==p['rows'],'original v2 execution binding/full roster')
    require(sha(output/'native_seal.json')==r['native_seal_sha256'] and physical.read(output/'native_seal.json')==
        {k:r[k] for k in ('rows','modules','guards','blocked','errors')},'native seal before saved analysis')
    rows=validate(p,r,[physical.read(output/f'event_{i:04}.json') for i in range(r['event_count'])],lambda e:bridge.rgb.load_array(e,blob))
    bridge.verify_bindings(p['bindings']);require(sha(output/'report.json')==expected,'final report unchanged')
    return dict(schema=NAME+'.consume',status='safeguarded_v2_terminal_RGB_verified_NOT_science',rows=rows,report_sha256=expected,
        counts=r['counts'],fixed_denominator=8,known_roots=2,fresh_roots=0,total_FLOPs=None,
        **{k:p[k] for k in ('inherited_compiler_contract','inherited_RGB_counts','inherited_original_compiler','inherited_original_physical',
            'actual_new_compiler_counts','actual_new_compile_wall_seconds','actual_new_physical_counts')},
        consume_new_neural_calls=0,consume_new_physics=0,P1_admission=False,P2_admission=False,physical_release=False)


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
