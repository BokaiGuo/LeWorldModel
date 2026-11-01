"""Saved-only known2 delivered-mean inverse/domain/roundtrip; no NN or physics."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
from experiments.spectral_reachability import agent_lewm_finalmean_paid_evaluator_20261029_v1 as paid
from experiments.spectral_reachability import pusht_d2p_packet_contract as packets

NAME='agent_lewm_delivered_mean_raw_domain_20261029_v1'
ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'results/agent_lewm_finalmean_paid_evaluator_20261029_v1_cuda_MAIN01/report.json'
REPORT_SHA='e7332cc12f75a8081405bfeed3da9db038ecab00d9234ebbb24f387c182dbebc'
RECEIPT=ROOT/'results/agent_lewm_finalmean_paid_evaluator_20261029_v1_consume_MAIN01.json'
RECEIPT_SHA='622754ccbd87a860598040d65af35572bb0acab0ee7be435e3a2604cf6564ee3'
PREPARE_SHA='f88321adc69f1175d9a02533867d6107aa49beb9c7c7534871320256b85b1313'
PINS={Path(paid.__file__):'5e569c07037f23a2c2b2403a49f0bd090738f7b1da31327bf97657b1c151dd72',
      Path(packets.__file__):'eda631348e929b45591807dac0adcbb72d10d84b7d48a7a558a492eab9c04aef',
      ROOT/'experiments/run_pusht_counterfactual.py':'0a4eb4592951139e596b35becede17d175df7b48a2fde16679ab0aac7c4dce34',
      ROOT/'.venv/lib/python3.12/site-packages/stable_worldmodel/envs/pusht/env.py':'c6d2104adc368ff1c352edc0e35ac74da9f07fda3a748ddbef0ba6b7a2d1daa2'}
CAP,FLOOR,WALL=8*1024**2,200*1024**2,900
need,sha,read,raw,exact,durable=paid.need,paid.sha,paid.read,paid.raw,paid.exact,paid.durable
load_task=paid.old.inputs.load_export
load_array=paid.old.load_array
task_identity=paid.old.inputs.core.task_id


def contract():
    return dict(roots=[0,1],known_only=True,fresh=False,condition='off',packets=2,raw_shape=[10,2],token_shape=[1,2,10],
        normalize_authority='P2 normalize_live_packet + original task float32 mean/std; NOT Hi scaler',
        raw_inverse='float32 separate multiply std10 then add mean10, time-major2x5x2; no FMA',
        roundtrip='float32 separate subtract mean10 then divide std10; supplied original P2 stats, no /100',
        raw_bounds=[-1.,1.],domain_gate='exact inclusive per element; no tolerance',
        normalize_live_packet_outside_domain='undefined/rejected; algebraic roundtrip diagnostic only',
        clipping=False,tanh=False,reselection=False,fit=False,training_provenance_verified=False,
        bitwise='dtype+shape+all bytes; signed zero distinguished',ULP='monotone float32 order with +/-0 same ULP',
        nonexact_roundtrip='new paid deployment-token replay required; old mean cost cannot certify changed tokens',
        verification='pinned complete paid/native receipts + byte-bound metadata/task/source; no rerun fullstrict',
        actual_CUDA_normalizer_execution=False,whole_known2_denominator_preserved=True,
        encoder=0,predictor=0,AD=0,physics=0,training=0,output_cap=CAP,disk_floor=FLOOR,wall_seconds=WALL,
        physical_release=False,scientific_admission=False)


def constant_stats(path):
    """Extract supplied P2 literals without importing the torch/scikit-learn module."""
    values={}
    for n in ast.parse(Path(path).read_text()).body:
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('PUBLIC_ACTION_MEAN','PUBLIC_ACTION_STD'):
            need(isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='np.asarray' and len(n.value.args)==1 and
                [(k.arg,ast.unparse(k.value)) for k in n.value.keywords]==[('dtype','np.float32')],'literal float32 P2 constants')
            values[n.targets[0].id]=np.asarray(ast.literal_eval(n.value.args[0]),np.float32)
    need(set(values)=={'PUBLIC_ACTION_MEAN','PUBLIC_ACTION_STD'},'both original stats')
    mean,std=values['PUBLIC_ACTION_MEAN'],values['PUBLIC_ACTION_STD']
    need(mean.shape==std.shape==(2,) and np.isfinite(mean).all() and np.isfinite(std).all() and (std>0).all(),'original stats shape/finite/positive')
    return mean,std


def packed(a):
    need(type(a) is np.ndarray and a.dtype==np.float32 and np.isfinite(a).all(),'finite original float32')
    b=np.ascontiguousarray(a).tobytes()
    return dict(dtype=a.dtype.str,shape=list(a.shape),values=a.tolist(),uint32_bits=a.view(np.uint32).tolist(),
        raw_bytes_hex=b.hex(),raw_bytes_sha256=hashlib.sha256(b).hexdigest(),nbytes=len(b))


def differences(a,b):
    summary=paid.compare(a,b)  # frozen pure NumPy comparator, never producer NN/math
    def ordered(x):
        u=x.view(np.uint32).astype(np.int64)
        return np.where(u>=0x80000000,0x100000000-u,u+0x80000000)
    return dict(summary,per_element_bit_equal=(a.view(np.uint32)==b.view(np.uint32)).tolist(),
        per_element_ULP=np.abs(ordered(a)-ordered(b)).tolist(),
        per_element_difference_float64=(b.astype(np.float64)-a.astype(np.float64)).tolist())


def evaluate(root,tokens,mean,std,task):
    need(type(root) is int and root in (0,1) and tokens.dtype==np.float32 and tokens.shape==(1,2,10) and
        np.isfinite(tokens).all() and mean.dtype==std.dtype==np.float32 and mean.shape==std.shape==(2,) and
        np.isfinite(mean).all() and np.isfinite(std).all() and (std>0).all(),'fixed root/f32 tokens/stats')
    need(exact(np.asarray(task['action_mean'],np.float32),mean) and exact(np.asarray(task['action_std'],np.float32),std),'task/scaler match before arithmetic')
    mean10=np.tile(mean,5);std10=np.tile(std,5)
    # Explicit ufunc stages force separately rounded float32, no fused multiply/add.
    product=np.multiply(tokens[0],std10,dtype=np.float32)
    raw_packets=np.add(product,mean10,dtype=np.float32)
    commands=raw_packets.reshape(10,2)
    need(np.isfinite(commands).all(),'finite inverse, no replacing overflow')
    lower=commands<np.float32(-1);upper=commands>np.float32(1);outside=lower|upper
    legal=not bool(outside.any());accepted=[]
    for p in commands.reshape(2,5,2):
        try:
            checked=packets.packet_commands(p)
            need(np.array_equal(checked,p.astype(np.float64)),'packet checker must not alter commands')
            accepted.append(True)
        except ValueError:
            need(bool((np.abs(p)>1).any()),'unexpected original packet checker rejection');accepted.append(False)
    need(legal==all(accepted),'original packet domain and exact element mask agree')
    subtracted=np.subtract(raw_packets,mean10,dtype=np.float32)
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
        roundtrip=np.divide(subtracted,std10,dtype=np.float32)[None]
    need(np.isfinite(roundtrip).all(),'finite roundtrip; no manufactured finite values')
    if legal:
        # Original P2 saved-array helper is a bound NumPy mirror of the optional
        # torch normalize_live_packet. This is NOT actual CUDA normalization.
        reference=paid.old.inputs.core.normalized_tokens(task,commands)[None]
        need(exact(roundtrip,reference),'exact original saved normalize helper')
    diff=differences(tokens,roundtrip);changed=not diff['bitwise']
    status='blocked_raw_domain' if not legal else ('requires_paid_deployment_token_replay' if changed else 'saved_domain_and_roundtrip_exact_only')
    violations=[]
    for r,c in np.argwhere(outside):
        violations.append(dict(raw_index=int(r),packet=int(r)//5,offset=int(r)%5,axis=int(c),
            value=float(commands[r,c]),uint32_bits=int(commands.view(np.uint32)[r,c]),
            side='below_lower' if lower[r,c] else 'above_upper'))
    return dict(root=root,status=status,delivered_tokens=packed(tokens),mean2=packed(mean),std2=packed(std),
        tiled_mean10=packed(mean10),tiled_std10=packed(std10),inverse_product=packed(product),
        raw_packets=packed(raw_packets),raw10=packed(commands),roundtrip_subtracted=packed(subtracted),roundtrip_tokens=packed(roundtrip),
        domain=dict(legal=legal,lower=-1.,upper=1.,raw_min=float(commands.min()),raw_max=float(commands.max()),
            per_element_in_box=(~outside).tolist(),packet_accepted=accepted,outside_count=int(outside.sum()),violations=violations),
        roundtrip_comparison=diff,roundtrip_is_legal_normalizer_input=legal,
        roundtrip_scope='P2 saved arithmetic' if legal else 'algebraic diagnostic only; normalize_live_packet would reject raw',
        deployment_token_replay_required=changed,old_mean_cost_certifies_roundtrip_tokens=bool(legal and not changed),
        old_mean_cost_must_not_be_reused_for_changed_tokens=changed,
        old_slice_mean_evaluated=False,physical_release=False,scientific_admission=False)


def collect():
    """Metadata and explicitly referenced NPY/task bytes only; no owner consume."""
    sources={str(p):h for p,h in PINS.items()}
    need(all(sha(p)==h for p,h in sources.items()),'frozen scaler/domain/evaluator sources')
    report=read(REPORT,REPORT_SHA);receipt=read(RECEIPT,RECEIPT_SHA)
    need(report['schema']==paid.NAME+'.report' and report['status']=='known2_paid_finalmean_evaluated' and report['errors']==[] and
        report['all_predictions_sealed'] is report['source_verified'] is True and report['protocol']==paid.protocol() and
        report['prepare_sha256']==PREPARE_SHA and len(report['rows'])==2,'actual full paid mean job')
    prepared=read(report['prepared'],PREPARE_SHA)
    need(prepared['schema']==paid.NAME+'.prepare' and prepared['protocol']==paid.protocol() and len(prepared['slots'])==2 and
        report['model_before']==report['model_after']==prepared['expected_model_hash'] and
        report['upstream']==prepared['upstream'] and report['inherited']==prepared['inherited'],'exact paid prepare/model/upstream')
    need(receipt['schema']==paid.NAME+'.consume' and receipt['status']=='known2_paid_finalmean_saved_verified' and
        receipt['report_sha256']==REPORT_SHA and receipt['upstream']==prepared['upstream'] and
        receipt['resources']==paid.protocol()['resources'] and receipt['inherited']==prepared['inherited'] and len(receipt['rows'])==2 and
        receipt['old_slice_mean_evaluated'] is receipt['physical_release'] is receipt['scientific_admission'] is False,'actual complete paid receipt authority')
    sources.update(prepared['sources'])
    sources.update({str(REPORT):REPORT_SHA,str(RECEIPT):RECEIPT_SHA,str(report['prepared']):PREPARE_SHA})
    upstream=prepared['upstream']
    need(upstream['report_sha256']==paid.REPORT_SHA and upstream['receipt_sha256']==paid.RECEIPT_SHA and
        upstream['prepare_sha256']==paid.PREPARE_SHA,'original installed CEM authority pins')
    original=read(upstream['report'],upstream['report_sha256']);original_receipt=read(upstream['receipt'],upstream['receipt_sha256'])
    original_prepare=read(upstream['prepared'],upstream['prepare_sha256'])
    sources.update({str(upstream[k]):upstream[h] for k,h in (('report','report_sha256'),('receipt','receipt_sha256'),('prepared','prepare_sha256'))})
    need(original['schema']==paid.old.NAME+'.report' and original['status']=='all_known2_native_normalized_CEM_recorded' and original['errors']==[] and
        original['prepare_sha256']==upstream['prepare_sha256'] and original['prepared']==upstream['prepared'] and
        original['protocol']==paid.old.protocol() and original['model_before']==original['model_after']==prepared['expected_model_hash'] and
        original_receipt['schema']==paid.old.NAME+'.consume' and original_receipt['status']=='all_known2_native_normalized_CEM_saved_verified' and
        original_receipt['report_sha256']==upstream['report_sha256'] and
        original_prepare['schema']==paid.old.NAME+'.prepare' and original_prepare['protocol']==paid.old.protocol() and
        original_prepare['expected_model_hash']==prepared['expected_model_hash'] and
        len(original['rows'])==len(original_receipt['rows'])==len(original_prepare['slots'])==2,'complete original mean delivery evidence')
    mean,std=constant_stats(ROOT/'experiments/run_pusht_counterfactual.py')
    rows=[]
    def array(folder,rec,inventory):
        value=load_array(folder,rec,inventory);sources[str(Path(folder)/rec['file'])]=rec['sha256'];return value
    for i,(row,attest,slot,nativerow,native_attest,taskslot) in enumerate(zip(report['rows'],receipt['rows'],prepared['slots'],original['rows'],original_receipt['rows'],original_prepare['slots'],strict=True)):
        need(row['root']==attest['root']==slot['root']==nativerow['root']==native_attest['root']==taskslot['source_slot']==i and
            row['status']=='paid_finalmean_evaluated' and attest['returned_mean_evaluated_in_new_job'] is True and
            attest['old_slice_mean_evaluated'] is False and nativerow['status']=='normalized_mean_sealed' and
            native_attest['returned_mean_evaluated'] is False,'both fixed original roots; no deployable-only subset')
        tokens=array(REPORT.parent,row['inputs']['actions'],report['artifacts'])
        refs=[nativerow['actions'],nativerow['mean'],nativerow['callbacks'][-1]['mean'],slot['refs']['actions']]
        for rec in refs:need(exact(tokens,array(Path(upstream['report']).parent,rec,original['artifacts'])),'all delivered/evaluated/last callback mean bytes identical')
        need(slot['refs']['actions']==nativerow['actions'] and taskslot['task_export'] in sources and
            sources[taskslot['task_export']]==taskslot['task_sha256'],'original task/source closure')
        task=load_task(taskslot['task_export'],taskslot['task_sha256'])['gn_task']
        for key,expected in (('emb',np.asarray(task['root_context']['latent_history'],np.float32)[None]),
            ('warm',np.asarray(task['root_context']['action_history'],np.float32)[None,None]),
            ('goal_emb',np.asarray(task['goal'],np.float32)[None,None])):
            need(exact(array(REPORT.parent,row['inputs'][key],report['artifacts']),expected),'paid context/warm/goal matched original task')
        need(exact(np.asarray(task['action_mean'],np.float32),mean) and exact(np.asarray(task['action_std'],np.float32),std), 'actual P2 task/scaler match')
        sources[taskslot['task_export']]=taskslot['task_sha256']
        rows.append(dict(root=i,tokens=tokens,mean=mean.copy(),std=std.copy(),task=task,
            identity=dict(task=taskslot['task_export'],task_sha256=taskslot['task_sha256'],
                task_id=task_identity(task),paid_actions=row['inputs']['actions'],native_actions=nativerow['actions'])))
    for path in (Path(__file__),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')):sources[str(path)]=sha(path)
    need(all(sha(p)==h for p,h in sources.items()),'all current metadata/task/model/scaler/source bytes; no inference')
    return dict(rows=rows,sources=sources,authority=dict(paid_report=str(REPORT),paid_report_sha256=REPORT_SHA,
        paid_receipt=str(RECEIPT),paid_receipt_sha256=RECEIPT_SHA,paid_prepared=report['prepared'],paid_prepare_sha256=PREPARE_SHA,
        native=upstream,normalizer_source=str(ROOT/'experiments/run_pusht_counterfactual.py'),
        packet_source=str(Path(packets.__file__)),training_provenance_verified=False,Hi_original_scaler=False,
        trust='complete saved receipts + current byte closure; no repeated fullstrict or actual CUDA normalization'),inherited=report['inherited'])


def rebuild():
    started=time.monotonic();data=collect();rows=[]
    for item in data['rows']:
        value=evaluate(item['root'],item['tokens'],item['mean'],item['std'],item['task']);value['identity']=item['identity'];rows.append(value)
    need([r['root'] for r in rows]==[0,1],'whole known2 denominator')
    need(time.monotonic()-started<WALL,'saved-analysis900s')
    return dict(schema=NAME+'.report',status='known2_raw_domain_and_roundtrip_analyzed',contract=contract(),rows=rows,
        all_roots_in_box=all(r['domain']['legal'] for r in rows),all_roundtrips_bitexact=all(r['roundtrip_comparison']['bitwise'] for r in rows),
        authority=data['authority'],sources=data['sources'],inherited=data['inherited'],
        cohort_rule='any future experiment must retain both known roots and blocked slots; no subset authorization',
        physical_release=False,scientific_admission=False)


def own(path):
    p=Path(path).absolute();need(p.parent==ROOT/'results' and p.name.startswith(NAME+'_') and p.suffix=='.json' and
        not any(x.is_symlink() for x in (p,*p.parents)),'exclusive own direct results JSON');return p


def analyze(output):
    p=own(output);need(not p.exists(),'never overwrite');v=rebuild();b=raw(v)
    need(len(b)<=CAP and shutil.disk_usage(p.parent).free>=FLOOR+len(b),'output8MiB/floor200MiB')
    durable(p,b);return dict(status=v['status'],path=str(p),sha256=sha(p))


def verify(report,h):
    p=own(report);need(p.stat().st_size<=CAP,'bounded saved report');v=read(p,h)
    need(raw(v)==p.read_bytes() and raw(v)==raw(rebuild()),'canonical whole saved numerical/source reconstruction')
    return dict(status='known2_raw_domain_roundtrip_saved_verified',report_sha256=h)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    a=sub.add_parser('analyze');a.add_argument('--output',required=True)
    a=sub.add_parser('verify');a.add_argument('--report',required=True);a.add_argument('--report-sha',required=True)
    args=parser.parse_args();v=analyze(args.output) if args.command=='analyze' else verify(args.report,args.report_sha)
    print(raw(v).decode(),end='')


if __name__=='__main__':main()
