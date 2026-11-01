"""GRASP bounded-H3 selected-plan physical adapter -> generic selected-RGB core, MAIN CUDA only.

Four fixed images (two opaque slots x step5/10). No physics, prediction or AD.
Adapter-prepare validation is explicit; generic physical.verify is never called.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import os
from pathlib import Path
from experiments.spectral_reachability import p1_constructive_selected_rgb_cuda_20261027_v2 as rgb
from experiments.spectral_reachability import p1_constructive_grasp_physical_20261027_v1 as adapter

NAME='p1_constructive_grasp_rgb_cuda_20261027_v1'
ROOT=adapter.ROOT
REPORT=ROOT/'results/p1_constructive_grasp_physical_20261027_v1_main01/report.json'
REPORT_SHA='bf7f15cf9d5aad4ca7923c4a116f6703c5e559b496c50488de55a4a94394f606'
PHYSICAL_PREPARE_SHA='6c617a16ff889ab06673b6dc9feb5f44fb7829aeb01bc1d21b9dfcb211d6d3c9'
SELECTION_SHA='ab6dcf51965091c388b0004037fc554f9fa7992011869f8cbdcbc8c31868e8f9'
ADAPTER_SHA='49df48cb1207459121445d2308d30928672aba3e90bad010210fd62ecd7e1c60'
RGB_SHA='e1448739bbae827a2e3b1f62bf50f0f27aa06d47aef60163483fa289e6eb432e'
require,sha,read=rgb.require,rgb.sha,rgb.read
enc=rgb.enc
EXPORTS,EXPORT_SHAS=adapter.compiler.EXPORTS,adapter.compiler.EXPORT_SHAS


def check_prepare(prepared,manifest,expected):
    """Saved identity only: do not open sourcepack/certificate or run material()."""
    require(prepared['schema']==adapter.NAME+'.prepare','own adapter prepare schema, not generic')
    require(manifest==expected and prepared['plans']==expected['plans'] and
        prepared['source_slots']==expected['roots'] and
        prepared['sourcepack_sha256']==adapter.SOURCEPACK_SHA and
        prepared['contract']==adapter.physical.contract(2,adapter.WALL),'exact own selected physical contract')
    a=prepared['adapter']
    require(a['source_sha256']==ADAPTER_SHA and a['compiler_report']==str(adapter.REPORT) and
        a['compiler_report_sha256']==adapter.REPORT_SHA and
        a['compiler_prepare_sha256']==adapter.PREPARE_SHA and
        a['generic_executor_sha256']==adapter.PHYSICAL_SHA and
        a['command_authority']=='frozen GRASP own consume only; no physical outcome selection',
        'frozen GRASP command authority')


def assemble(report,events,prepared,tasks):
    """Normalize only the schema tag in RAM for the sealed generic array join.

    The journal must first match the ORIGINAL adapter prepare byte-content.
    The original hash-bound records are never changed or written back.
    """
    require(prepared['schema']==adapter.NAME+'.prepare' and events[0]['prepare']==prepared,
        'original adapter prepare matches journal before schema view')
    view=deepcopy(prepared);view['schema']=adapter.physical.NAME+'.prepare'
    ev=deepcopy(events);ev[0]['prepare']=deepcopy(view)
    return rgb.assemble(report,ev,view,tasks,[adapter.ARM])


def build():
    require(REPORT_SHA is not None and PHYSICAL_PREPARE_SHA is not None,
        'MAIN actual physical report/prepare SHA pending; no prepare or CUDA')
    adapter.compiler.core.hash_value(REPORT_SHA);adapter.compiler.core.hash_value(PHYSICAL_PREPARE_SHA)
    require(sha(adapter.__file__)==ADAPTER_SHA and sha(rgb.__file__)==RGB_SHA and
        sha(rgb.old.__file__)==rgb.OLD_SHA and sha(adapter.physical.__file__)==adapter.PHYSICAL_SHA,
        'frozen adapter and generic RGB/physics sources')
    report=read(REPORT,REPORT_SHA)
    bp=REPORT.parent/'execution_binding.json';binding=read(bp)
    require(binding['prepare_sha256']==PHYSICAL_PREPARE_SHA,'MAIN external physical adapter prepare SHA')
    pp=Path(binding['prepared']).absolute();prep=read(pp,PHYSICAL_PREPARE_SHA)
    require(prep['manifest_sha256']==SELECTION_SHA,'MAIN fixed physical selection SHA')
    mp=Path(prep['manifest']).absolute();manifest=read(mp,prep['manifest_sha256'])
    # Own consume is the original model-command authority; no physical feedback
    # or success field participates in mapping either branch to encoding.
    expected,inherited=adapter.build(adapter.REPORT,adapter.REPORT_SHA)
    check_prepare(prep,manifest,expected)
    bindings={str(REPORT):REPORT_SHA,str(bp):sha(bp),str(pp):PHYSICAL_PREPARE_SHA,
        str(mp):prep['manifest_sha256']}
    bindings.update(inherited)
    # Seal original closure bytes, but never parse evaluator-only source files.
    for path,h in prep['bindings'].items():
        require(sha(path)==h,'original physical closure seal');bindings[path]=h
    tasks=[]
    for slot,(path,h) in enumerate(zip(EXPORTS,EXPORT_SHAS,strict=True)):
        interface=rgb.planner.load_export(path,h);export=read(path,h)
        require(type(export['source_slot']) is int and export['source_slot']==slot and
            export['blind_sha256']==export['blind_source_sha256']==adapter.BLIND_SHAS[slot] and
            interface['task_id']==adapter.TASK_IDS[slot],'own approved blind task export identity')
        import numpy as np
        require(np.array_equal(interface['gn_task']['projection'],np.eye(192)),'original identity mean Q; no fit')
        tasks.append(dict(source_slot=slot,task_export=path,task_sha256=h,
            blind_sha256=adapter.BLIND_SHAS[slot],interface=interface));bindings[path]=h
    jp=REPORT.parent/'journal.jsonl';events=rgb.old.journal(jp,report['journal_sha256'])
    bindings[str(jp)]=report['journal_sha256'];images=assemble(report,events,prep,tasks)
    for item in images:
        c=item['checkpoint'];f=rgb.old.local(REPORT.parent,c['png'])
        rgb.old.png(f,c['file_sha256'],c['pixel_sha256']);item['png_path']=str(f);bindings[str(f)]=c['file_sha256']
    for path,h in enc.MODELS.items():require(sha(path)==h,'frozen model/config bytes');bindings[path]=h
    files=rgb.planner.source_files()+[Path(rgb.__file__),Path(rgb.old.__file__),Path(adapter.__file__),
        Path(adapter.physical.__file__),Path(__file__),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]
    bindings.update({str(Path(f).resolve()):sha(f) for f in files})
    for path,h in bindings.items():require(sha(path)==h,'final saved input/source TOCTOU')
    return dict(schema=NAME+'.prepare',images=images,bindings=bindings,policy=str(enc.POLICY),
        physical_report=str(REPORT),physical_report_sha256=REPORT_SHA,physical_prepare_sha256=PHYSICAL_PREPARE_SHA,
        expected_model_hash=enc.MODEL_TENSOR_SHA,contract=dict(roots=2,arms=[adapter.ARM],branches=2,images=4,
        encoder=4,projector=4,predict=0,action_encoder=0,AD=0,physics=0,training=False,no_dedup=True,
        Q='identity192_no_fit',selection=False,P1_admission=False,P2_admission=False,official_equivalence=False,
        prediction='unknown_not_recomputed',inherited_costs='compiler/physics/original encoding separately paid',
        numerics=dict(SDP='MATH',matmul_TF32=False,cudnn_TF32=False),
        output_cap_bytes=rgb.CAP,disk_floor_bytes=rgb.FLOOR))


def prepare(output):
    return adapter.write_new(output,build())


def verify(path,expected):
    p=read(path,expected);require(p==build(),'complete own RGB adapter reconstruction');return p


def run(prepared,expected,output):
    # Own verifier, not rgb.verify() / generic physical.verify(). Encoding,
    # PNG decode, ledger, NPY writes and metrics use unchanged generic core.
    p=verify(prepared,expected);store=rgb.Store(output);errors=[];model=None
    before=after=host=None;source_ok=False;ledger=enc.Ledger(1,store.event)
    rows=[dict(**{k:i[k] for k in ('index','branch','root_slot','source_slot','arm','role','task_id','task_export','task_sha256','blind_sha256')},
        status='not_started',provenance=i) for i in p['images']]
    complete=False
    try:
        store.put('binding.json',dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected))
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import SDPBackend,sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
        host=require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        enc.frozen(model);before=enc.tensor_hash(model);require(before==p['expected_model_hash'],'original frozen model tensor SHA')
        transform=p0.image_transform();ledger.sync=lambda:torch.cuda.synchronize(device);ledger.no_grad=lambda:not torch.is_grad_enabled()
        with sdpa_kernel(SDPBackend.MATH),enc.guard(model,torch,ledger,errors):
            def callback(a):
                with torch.inference_mode():
                    x=p0._image_tensor(a,transform=transform,device=device)
                    require(x.device.type=='cuda' and tuple(x.shape)==(1,1,3,224,224),'native CUDA RGB batch1')
                    z=model.encode({'pixels':x})['emb']
                    require(z.device.type=='cuda' and z.dtype==torch.float32 and tuple(z.shape)==(1,1,192),'native projected CLS192')
                    return z[0,0].detach().cpu().numpy().copy()
            rgb.encode_images(p,callback,ledger,store,rows)
    except BaseException as exc:
        errors.insert(0,enc.error('execution',exc))
        try:store.event(dict(kind='execution_error',error=errors[0]))
        except BaseException as e:errors.append(enc.error('error_journal',e))
    finally:
        ledger.image=None
        for row in rows:
            if row['status']=='attempted':row['status']='failed_partial'
        if model is not None:
            try:enc.frozen(model);after=enc.tensor_hash(model)
            except BaseException as e:errors.append(enc.error('final_model',e))
        try:verify(prepared,expected);source_ok=True
        except BaseException as e:errors.append(enc.error('final_source',e))
        complete=not errors and not ledger.blocked and source_ok and before is not None and before==after and all(r['status']=='encoded' for r in rows) and ledger.counts()=={k:dict(attempted=4,completed=4) for k in ('encoder','projector')}
        hashes={}
        try:artifacts=list(store.output.iterdir())
        except BaseException as e:errors.append(enc.error('artifact_listing',e));complete=False;artifacts=[]
        for f in artifacts:
            if f.is_file():
                try:hashes[f.name]=sha(f)
                except BaseException as e:errors.append(enc.error('artifact_hash',e));complete=False
        report=dict(schema=NAME+'.report',status='grasp_selected_rgb_encoded_not_admitted' if complete else 'partial_or_failed',
            prepare_sha256=expected,physical_report_sha256=REPORT_SHA,physical_prepare_sha256=PHYSICAL_PREPARE_SHA,
            rows=rows,resources=ledger.counts(),errors=errors,first_error=errors[0] if errors else None,
            source_verified=source_ok,model_hash_before=before,model_hash_after=after,
            model_unchanged=before==after if before is not None else None,host=host,artifact_sha256=hashes,
            contract=p['contract'],new_predict=0,new_AD=0,new_physics=0,training=False,P1_admission=False,P2_admission=False,
            official_equivalence=False)
        try:store.put('report.json',report,final=True)
        except BaseException as e:errors.append(enc.error('final_report',e));complete=False;print(rgb.canonical(dict(errors=errors)).decode(),flush=True)
    require(complete,'partial output retained; no retry: '+str(errors));return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','verify','run'))
    for k in ('prepared','prepare-sha','output'):p.add_argument('--'+k)
    a=p.parse_args()
    if a.command=='prepare':r=prepare(a.output)
    elif a.command=='verify':r=dict(status='verified_no_execution',images=len(verify(a.prepared,a.prepare_sha)['images']))
    else:r=run(a.prepared,a.prepare_sha,a.output)
    print(rgb.canonical(r).decode(),end='')


if __name__=='__main__':main()
