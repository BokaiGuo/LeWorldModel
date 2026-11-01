"""Saved-only terminal20 supplement to immutable v1 attribution evidence.

No encoder/runtime/compiler import, torch, model loading, AD, or physics.
Byte/receipt validation is not re-execution of the recorded neural forwards.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.spectral_reachability import p4_feedback_attribution_audit_20261027_v1 as v1
from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal, verify_seal

TAG='p4_feedback_attribution_audit_20261027_v2'
ENCODER='p4_terminal20_cuda_encoding_20261027_v1'
REPO=Path(__file__).resolve().parents[2]
DEFAULT_V1=REPO/'results/p4_feedback_attribution_audit_20261027_v1_verify02/report.json'
DEFAULT_RECEIPT=REPO/f'results/{ENCODER}_cuda_main01/receipt.json'
PREPARE_SHA='54b8aaee4d5cb58844eac0b6741e86a4f8b6eb4dadc9acfb178e93e6905a3495'
ENCODER_SHA='b0656a1e3feab4d277d5d3159e8a4fbea5267ee390670291df08be2a3fafdcd2'
V1_SHA='f73e10ad32f33908d0199331834c45de9fe8e6f957dfa7eab582fc302e483216'
ARMS=('imagined','actual')
require=v1.require


def canonical(v):
    return (json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()


class Inputs(v1.Inputs):
    """Small reads inherited; model bindings streaming-only, never deserialized."""
    def __init__(self):
        super().__init__(); self.large_hashes={}

    def hash(self,path):
        path=Path(path).resolve()
        require(path.is_file() and path.stat().st_size<=1024*1024*1024,'bounded model/source file')
        h=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
        digest=h.hexdigest()
        require(str(path) not in self.large_hashes or self.large_hashes[str(path)]==digest,'streamed input changed')
        self.large_hashes[str(path)]=digest
        return digest

    def stable(self):
        super().stable()
        for path,expected in list(self.large_hashes.items()): require(self.hash(path)==expected,'streamed input changed')


def validate_entries(receipt,journal):
    require(receipt['status']=='two_terminal20_latents_saved' and receipt['schema']==ENCODER+'.derived_receipt', 'completed derived schema/status')
    require(receipt['historical_runs']==1 and not receipt['errors'] and receipt['first_error'] is None and
        not receipt['blocked'] and receipt['forbidden_attempts']=={},'derived errors or blocked work')
    for key in ('predict_calls','AD_calls','physics_steps'):
        require(type(receipt[key]) is int and receipt[key]==0,'zero forbidden resource: '+key)
    for key in ('training','P1_admission','P4_admission','efficacy_verified'):
        require(receipt[key] is False,'scope flag: '+key)
    require(receipt['model_unchanged'] is True and receipt['model_hash_before']==receipt['model_hash_after'] and
        isinstance(receipt['model_hash_before'],str) and len(receipt['model_hash_before'])==64,'model state receipts')
    require(receipt['host']['device']=='cuda:0' and receipt['host']['allocation_sum']==4.0,'HOST CUDA allocation receipt')
    expected=[dict(arm=arm,kind=kind,index=i*2+j,completed=True)
              for i,arm in enumerate(ARMS) for j,kind in enumerate(('encoder','existing_projector'))]
    require(receipt['entries']==expected and receipt['counts']=={kind:dict(attempted=2,completed=2)
        for kind in ('encoder','existing_projector')},'encoder/projector entry budgets')
    require(len(receipt['rows'])==2 and {r['arm'] for r in receipt['rows']}==set(ARMS),'derived arm census')
    expected_journal=[]
    for arm in ARMS:
        for e in [v for v in expected if v['arm']==arm]:
            expected_journal.extend([dict(e,event='attempt',completed=False),dict(e,event='return_or_error')])
        expected_journal.append(dict(event='endpoint',row=next(v for v in receipt['rows'] if v['arm']==arm)))
    require(journal==expected_journal,'exact entry/endpoint journal sequence')


def validate_endpoint(row,endpoint,original_observation,root,native):
    require(row['status']=='encoded_and_saved' and row['arm'] in ARMS and row['root_sha256']==root and
        row['primitive_step']==20,'root/arm/step identity')
    for key,value in endpoint.items(): require(row[key]==value,'prepared endpoint field: '+key)
    require(original_observation['arm']==row['arm'] and original_observation['primitive_step']==20 and
        original_observation['event_index']==row['event_index'] and original_observation['state']==row['state'] and
        original_observation['rgb_path']==row['rgb_path'] and original_observation['rgb_sha256']==row['rgb_file_sha256'] and
        original_observation['pixel_sha256']==row['pixel_sha256'],'original observation binding')
    require(row['latent_shape']==[192] and row['latent_dtype']=='float32' and native.shape==(192,) and
        native.dtype==np.dtype('float32') and np.isfinite(native).all(),'native float32 finite192 array')


def terminal_metrics(predicted,observed15,observed20,goal,q):
    """Independent numeric decomposition, retaining signed cross term."""
    pred,old,real=[v1.array(v,goal.shape) for v in (predicted,observed15,observed20)]
    diff=real-pred; projected=q@diff; residual=q@(goal-pred)
    predicted_cost=float(np.mean(residual**2))
    real_cost=float(np.mean((q@(goal-real))**2))
    start_cost=float(np.mean((q@(goal-old))**2))
    square=float(np.mean(projected**2)); cross=float(-2*np.mean(residual*projected))
    require(np.isclose(real_cost-predicted_cost,square+cross,rtol=1e-10,atol=1e-10),'terminal residual identity')
    return dict(predicted_Q_cost=predicted_cost,observed15_Q_cost=start_cost,observed20_Q_cost=real_cost,
        observed_Q_cost_change=real_cost-start_cost,observed_minus_predicted_Q_cost=real_cost-predicted_cost,
        prediction_error=dict(raw_l2=float(np.linalg.norm(diff)),raw_mse=float(np.mean(diff**2)),
            projected_l2=float(np.linalg.norm(projected)),projected_mse=square,
            goal_cross_term=cross,observed_minus_predicted_cost=real_cost-predicted_cost,
            identity_error=square+cross-(real_cost-predicted_cost)))


def audit(v1_report=DEFAULT_V1,receipt_path=DEFAULT_RECEIPT):
    inputs=Inputs()
    old=inputs.json(v1_report); receipt_path=Path(receipt_path).resolve()
    r=inputs.json(receipt_path); folder=receipt_path.parent
    require(old['schema']==v1.TAG and old['status']=='saved_diagnostics_recorded_not_admission' and
        old['inputs_unchanged'] is True and old['P1_admission'] is False and old['P4_admission'] is False,'v1 scope/status')
    require(inputs.hash(v1.__file__)==V1_SHA and old['inputs'][str(Path(v1.__file__).resolve())]==V1_SHA,'immutable v1 source binding')
    for path,h in old['inputs'].items(): require(inputs.hash(path)==h,'v1 historical input SHA: '+path)
    prior=[run for run in old['runs'] if run['report_path']==r['source_report']]
    require(len(prior)==1 and prior[0]['arithmetic_verified'] is True,'unique v1/source runtime join')
    original=Path(r['source_report']).resolve(); p=inputs.json(original.parent/'protocol.json')
    require(inputs.hash(original)==r['source_report_sha256'],'source report SHA')
    # Re-run immutable saved-data v1, not its CLI; no new model/environment.
    rebuilt=v1.audit_report(original,inputs)
    require(rebuilt==prior[0],'v1 full arithmetic/history/budget reproduction')
    execution=inputs.json(folder/'execution_binding.json')
    prepared=Path(execution['prepared']).resolve()
    require(execution['prepare_sha256']==r['prepare_sha256']==PREPARE_SHA and
        inputs.hash(prepared/'prepare.json')==PREPARE_SHA,'independently retained prepare SHA')
    prep=inputs.json(prepared/'prepare.json'); ep=inputs.json(prepared/'protocol.json')
    require(prep['status']=='prepared_not_executed' and prep['schema']==ENCODER+'.prepare' and
        inputs.hash(prepared/'protocol.json')==prep['protocol_sha256'],'encoder prepared protocol binding')
    require(inputs.json(prepared/'launch.json')==dict(output=str(folder),prepare_sha256=PREPARE_SHA),'exclusive launch/output binding')
    seal=prep['source_seal']
    verify_seal(output_dir=seal['output_dir'],expected_manifest_sha256=seal['manifest_sha256'])
    manifest=inputs.json(Path(seal['output_dir'])/'manifest.json')
    require(inputs.json(Path(seal['output_dir'])/'args.json')['protocol_sha256']==prep['protocol_sha256'],'encoder seal args/protocol')
    for e in manifest['files']:
        require(ep['sources_sha256'][e['source']]==e['sha256'] and inputs.hash(e['source'])==e['sha256'] and
                inputs.hash(Path(seal['output_dir'])/e['archive'])==e['sha256'],'encoder source/archive SHA')
    encoder_path=str(REPO/f'experiments/spectral_reachability/{ENCODER}.py')
    require(ep['sources_sha256'][encoder_path]==ENCODER_SHA,'retained frozen encoder source SHA')
    for path,h in {**ep['bindings'],**ep['model_bindings'],**ep['sources_sha256'],**ep['historical_source_bindings']}.items():
        require(inputs.hash(path)==h,'encoder bound input/model/source changed: '+path)
    require(ep['schema']==ENCODER+'.protocol' and ep['source_report']==str(original) and
        ep['source_report_sha256']==r['source_report_sha256'] and ep['root_sha256']==r['root_sha256']==p['root_sha256'] and
        ep['goal_q_source']==str(original.parent/'protocol.json'),'derived source/root/goal identity')
    models={str(Path(p['policy'])/name):p['bindings'][str(Path(p['policy'])/name)] for name in ('weights.pt','config.json')}
    require(r['model_bindings']==ep['model_bindings']==models,'original frozen model binding')
    require(hashlib.sha256(canonical([p['goal'],p['q']])).hexdigest()==ep['goal_q_sha256']==r['goal_q_sha256'],'goal/Q array SHA')
    c=ep['contract']
    require(c['historical_runs']==1 and c['arms']==list(ARMS) and c['primitive_step']==20 and c['latent_dim']==192 and
        c['encoder_cap']==c['existing_projector_cap']==2 and c['per_arm_each_cap']==1 and c['device']=='cuda:0' and
        c['cpu_neural_fallback'] is False and c['predict_cap']==c['AD_cap']==c['physics_cap']==0,'encoder budget contract')
    journal_raw=inputs.raw(folder/'encoding.events.jsonl')
    require(hashlib.sha256(journal_raw).hexdigest()==r['journal_sha256'],'derived journal SHA')
    journal=[v1.strict_json(line) for line in journal_raw.splitlines()]
    validate_entries(r,journal)
    events=[v1.strict_json(line) for line in inputs.raw(original.parent/'runtime.events.jsonl').splitlines()]
    require(len(ep['endpoints'])==2 and {e['arm'] for e in ep['endpoints']}==set(ARMS),'prepared endpoint census')
    goal=v1.array(p['goal'],(192,)); q=v1.array(p['q'],(192,192))
    rows=[]
    for arm in ARMS:
        row=next(v for v in r['rows'] if v['arm']==arm)
        endpoint=next(v for v in ep['endpoints'] if v['arm']==arm)
        observation=v1.one([e for e in events if e.get('arm')==arm and e.get('primitive_step')==20],'observation')
        latent_path=Path(row['latent_path']).resolve()
        require(latent_path==folder/f'{arm}.terminal20.native192.npy','latent array path identity')
        raw=inputs.raw(latent_path); require(len(raw)<4096 and hashlib.sha256(raw).hexdigest()==row['latent_sha256'],'latent array file SHA')
        native=np.load(io.BytesIO(raw),allow_pickle=False)
        validate_endpoint(row,endpoint,observation,p['root_sha256'],native)
        png=inputs.raw(row['rgb_path'])
        require(hashlib.sha256(png).hexdigest()==row['rgb_file_sha256'],'terminal PNG SHA')
        with Image.open(io.BytesIO(png)) as image:
            require(image.mode=='RGB' and image.size==(224,224),'terminal RGB format')
            rgb=np.asarray(image).copy()
        require(hashlib.sha256(rgb.tobytes()).hexdigest()==row['pixel_sha256'],'terminal pixel SHA')
        selection=v1.one([e for e in events if e.get('arm')==arm and e.get('boundary')==15],'selection')
        context=v1.one([e for e in events if e.get('arm')==arm and e.get('boundary')==15],'planning_context')
        predicted=selection['successor']['latent_history'][-1]
        observed15=context['observed_context']['latent_history'][-1]
        metrics=terminal_metrics(predicted,observed15,native,goal,q)
        require(np.isclose(metrics['observed20_Q_cost'],row['original_Q_cost'],rtol=0,atol=1e-12),'derived saved Q cost')
        historical=next(v for v in prior[0]['rounds'] if v['arm']==arm and v['boundary']==15)
        require(historical['next_latent_status']=='unconfirmed_not_saved' and historical['prediction_error'] is None and
            historical['observed_Q_cost_change'] is None,'preserve v1 missing-at-that-time history')
        require(np.isclose(metrics['predicted_Q_cost'],historical['predicted_Q_cost'],rtol=0,atol=1e-12),'v1 prediction identity')
        rows.append(dict(root_sha256=p['root_sha256'],arm=arm,boundary=15,end=20,round=3,
            v1_historical_status=historical['next_latent_status'],v2_latent_status='derived_real_RGB_native192_verified',
            latent_path=str(latent_path),latent_sha256=row['latent_sha256'],pixel_sha256=row['pixel_sha256'],
            selection_sha256=selection['selection_sha256'],metrics=metrics,
            physical_change=deepcopy(historical['physical_change']),physical_after=deepcopy(historical['physical_after']),
            conditioning=historical['prediction_conditioning'],causal_attribution=False))
    inputs.stable()
    return dict(schema=TAG,status='saved_terminal20_supplement_verified',rows=rows,
        v1_report=str(Path(v1_report).resolve()),derived_receipt=str(receipt_path),inputs={**inputs.hashes,**inputs.large_hashes},
        historical_runs=1,located_runs=old['located_runs'],missing_runs=old['missing_runs'],
        two_run_evidence_verified=False,inputs_unchanged=True,v1_history_preserved=True,
        terminal20_actual_minus_imagined_Q_cost=rows[1]['metrics']['observed20_Q_cost']-rows[0]['metrics']['observed20_Q_cost'],
        original_endpoint_comparison=prior[0]['actual_minus_imagined'],
        derived_historical_resources=r['counts'],new_neural_calls=0,new_physics_steps=0,new_AD_calls=0,
        P1_admission=False,P4_admission=False,efficacy_verified=False,
        limitations=['one historical run, two arms, one reused root; no second batch or independent generalization',
            'source/model/PNG/array/entry receipts verified from saved bytes; encoder and model-state hashes not recomputed by neural execution',
            'imagined-arm error includes accumulated rolling-context drift; not an isolated one-step observed-start error',
            'latent/Q and physical pose have different units; no unique causal explanation or P1/P4 admission'])


def write(path,value):
    raw=canonical(value); require(len(raw)<2*1024*1024,'bounded audit output')
    with Path(path).open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


def run(output,v1_report=DEFAULT_V1,receipt_path=DEFAULT_RECEIPT):
    output=Path(output).resolve()
    require(not output.exists() and output.name.startswith(TAG),'new exclusive v2 output required')
    output.mkdir(exist_ok=False)
    sources=[Path(__file__),REPO/f'tests/test_{TAG}.py',REPO/f'innerdocs/{TAG}.md',
             Path(v1.__file__),REPO/'experiments/spectral_reachability/d2p_prerun_source_seal.py']
    seal=create_seal(files=sources,args_json=json.dumps(dict(v1_report=str(Path(v1_report).resolve()),
        receipt_path=str(Path(receipt_path).resolve()),new_neural_calls=0,new_physics_steps=0)),output_dir=output/'source_seal')
    try:
        result=audit(v1_report,receipt_path)
        result['audit_source_seal']=seal
        result['source_verification']=verify_seal(output_dir=seal['output_dir'],expected_manifest_sha256=seal['manifest_sha256'])
        write(output/'report.json',result)
        return result
    except BaseException as exc:
        write(output/'failure.json',dict(schema=TAG,status='audit_fail_closed',type=type(exc).__name__,message=str(exc),audit_source_seal=seal))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v1-report',type=Path,default=DEFAULT_V1)
    parser.add_argument('--receipt-path',type=Path,default=DEFAULT_RECEIPT)
    parser.add_argument('--output',type=Path,required=True)
    result=run(**vars(parser.parse_args()))
    print(json.dumps(dict(status=result['status'],rows=len(result['rows']),P4_admission=result['P4_admission'])))
