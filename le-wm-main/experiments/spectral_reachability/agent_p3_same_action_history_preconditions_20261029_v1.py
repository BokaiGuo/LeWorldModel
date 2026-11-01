"""Saved prepare/inspect only: eight actual-history jobs and strict old-R IO eligibility.

No producer import, upstream fullstrict, model, torch, physics or execution entrypoint.
"""
from __future__ import annotations

import argparse
import base64
from copy import deepcopy
import os
from pathlib import Path
import shutil

import numpy as np

from experiments.spectral_reachability import agent_p3_same_action_error_decomposition_20261029_v1 as saved

NAME = 'agent_p3_same_action_history_preconditions_20261029_v1'
PLANNER = 'agent_p3_shared_panel_20261029_v1'
ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT/'innerdocs/agent_p3_same_action_next_protocol_20261029_v1.md'
PROTOCOL_SHA = 'a66ddd4a8417de5b4dbab17f78ea9032d609393cca85697e4dc7eb609d8c0d49'
PINS = {
    ROOT/'experiments/spectral_reachability'/(saved.ENC+'.py'): saved.ENC_SHA,
    Path(saved.__file__).resolve(): '7fd089b6801ad45a230ab25a8a565c24978df7902a5b83ff32e0b4d9f96c3398',
    ROOT/'experiments/spectral_reachability'/(PLANNER+'.py'):
        '8cc8f796f313d4cc3f209ce231ebcaec4808f096acc02781cd6df734800dc464',
    PROTOCOL: PROTOCOL_SHA,
}
require, canonical, sha = saved.require, saved.canonical, saved.sha
META = 128*1024**2
JOURNAL_CAP = 128*1024**2
LINE_CAP = 8*1024**2
FLOOR = 200*1024**2
KEYS = ('encoder_prepared','encoder_report','encoder_receipt',
        'planner_prepared','planner_report','planner_receipt')


def digest(value): return sha(canonical(value))


def f32(value, shape):
    def numeric(x):
        return all(numeric(v) for v in x) if type(x) is list else type(x) in (int,float)
    require(type(value) is list and numeric(value), 'literal numeric array, no bool')
    a = np.asarray(value, np.float64)
    require(a.shape == shape and np.isfinite(a).all() and
            np.array_equal(a,a.astype(np.float32).astype(np.float64)), 'exact finite float32 shape')
    return a.astype(np.float32)


def same(a,b,why):
    require(a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes(),why)


def context(value):
    require(type(value) is dict and set(value)=={'layout','context_length','latent_dim',
            'action_dim','latent_history','action_history'}, 'complete context schema')
    require(type(value['context_length']) is int and value['context_length']==3
            and type(value['latent_dim']) is int and value['latent_dim']==192
            and type(value['action_dim']) is int and value['action_dim']==10
            and value['layout']=='states_with_between_state_actions_v1','native context identity')
    return f32(value['latent_history'],(3,192)), f32(value['action_history'],(2,10))


def normalization(task, raw10):
    mean,std = f32(task['action_mean'],(2,)),f32(task['action_std'],(2,))
    require((std>0).all() and (np.abs(raw10)<=1).all(),'original normalizer and rawbox')
    return ((raw10-mean)/std).reshape(2,10)


def unblob(value):
    require(type(value) is dict and set(value)=={'shape','dtype','sha256','data'} and
            value['dtype']=='<f4' and type(value['shape']) is list and
            all(type(x) is int and 0<x<=4096 for x in value['shape']), 'native blob ABI')
    data=base64.b64decode(value['data'],validate=True)
    require(len(data)<=LINE_CAP and len(data)==int(np.prod(value['shape']))*4 and
            sha(data)==value['sha256'],'native blob bytes/SHA')
    a=np.frombuffer(data,dtype='<f4').reshape(value['shape'])
    require(np.isfinite(a).all(),'native finite blob')
    return a


def read_selected_journal(path, expected, expected_tail, scope):
    """Authenticate whole chain, keep only exact selected H3 stream; no fullstrict."""
    path=Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size<=JOURNAL_CAP,
            'required saved R journal missing or oversized')
    import hashlib
    h=hashlib.sha256();tail='0'*64;selected=[];byte_count=0
    with path.open('rb') as f:
        i=0
        while True:
            line=f.readline(LINE_CAP+1)
            if not line:break
            byte_count+=len(line);h.update(line)
            require(len(line)<=LINE_CAP and byte_count<=JOURNAL_CAP and line.endswith(b'\n'),
                    'one bounded complete native journal record')
            record=saved.decode(line)
            require(set(record)=={'index','previous','payload','sha256'} and
                    type(record['index']) is int and record['index']==i and record['previous']==tail and
                    digest({k:v for k,v in record.items() if k!='sha256'})==record['sha256'],
                    'native journal full hashchain')
            tail=record['sha256'];e=record['payload']
            require(type(e) is dict and type(e['kind']) is str,'native event ABI')
            if e.get('scope')==scope:
                selected.append(dict(record_index=i,payload=e))
            i+=1
    require(h.hexdigest()==expected and tail==expected_tail,'R journal report SHA/tail')
    return selected, dict(path=str(path.resolve()),sha256=expected,tail=tail,bytes=byte_count,records=i)


SEQUENCE = [('program_packet',('latent','history','raw','mean','std','token','outgoing')),
    ('io_input',('action_encoder:0',)),('io_return',('action_encoder:output',)),
    ('io_input',('predict:latent','predict:act_emb')),
    ('io_input',('predictor:0','predictor:1')),('io_return',('predictor:output',)),
    ('io_input',('pred_proj:0',)),('io_return',('pred_proj:output',)),
    ('io_return',('predict:output',)),('program_boundary',('latent','history'))]


def certify_R(node, witness, selected):
    """All three edges of the original winner selected replay, actual native full IO."""
    result=node['receipt'];winner=witness['winner'];req=witness['request']
    require(node['status']==result['status']=='sealed' and node['error'] is None and
            result['error'] is None and result['selected_horizon']==3 and
            canonical(node['request'])==canonical(req) and result['request_sha256']==digest(req)
            and canonical(result)==canonical(witness['result']), 'original sealed b5 request/result')
    require(canonical(result['resource_receipt'])==canonical(node['resource_receipt']),
            'native resource receipt identity')
    panel=next(p for p in result['panels'] if p['horizon']==3)
    require(canonical(panel['winner'])==canonical(winner) and
            canonical(panel['selected_path'])==canonical(winner['whole']), 'fixed original H3 winner replay')
    scope=dict(slot=node['source_slot'],mode=node['mode'],horizon=3,index=winner['index'],phase='selected')
    require(len(selected)==30,'exact three-edge selected replay, no unknown/spliced events')
    task=witness['task'];state,history=context(req['task']['root_context'])
    actions=f32(winner['actions'],(15,2));mean,std=f32(task['action_mean'],(2,)),f32(task['action_std'],(2,))
    require((std>0).all() and (np.abs(actions)<=1).all(),'original normalizer/raw domain')
    operations=node['resource_receipt']['events'];certificates=[];seen=set()
    require(all(type(rec['record_index']) is int for rec in selected) and
            all(a['record_index']<b['record_index'] for a,b in zip(selected,selected[1:])),
            'strict original journal chronology')
    for step in range(3):
        stream=selected[10*step:10*step+10];arrays=[]
        for rec,(kind,keys) in zip(stream,SEQUENCE,strict=True):
            e=rec['payload']
            require(set(e)=={'kind','scope','phase','step','operation_index','arrays'} and
                    e['kind']==kind and e['scope']==scope and e['phase']=='whole' and
                    type(e['step']) is int and e['step']==step and set(e['arrays'])==set(keys),
                    'selected whole lifecycle exact chronology/fields')
            arrays.append({k:unblob(e['arrays'][k]) for k in keys})
        packet,ae_in,ae_out,pr_in,inner_in,inner_out,proj_in,proj_out,pr_out,boundary=arrays
        require(stream[0]['payload']['operation_index'] is None and stream[-1]['payload']['operation_index'] is None,
                'packet boundary have no invented operation')
        for resource,positions,input_shape in (('action_encoder',(1,2),(1,3,10)),
                ('predict',(3,8),(1,3,192)),('predictor',(4,5),(1,3,192)),('pred_proj',(6,7),(3,192))):
            ids=[stream[k]['payload']['operation_index'] for k in positions]
            require(type(ids[0]) is int and ids[0]==ids[1] and ids[0] not in seen,
                    'unique matched actual operation IDs')
            seen.add(ids[0]);ops=[op for op in operations if op['index']==ids[0]]
            require(len(ops)==1 and ops[0]['resource']==resource and ops[0]['scope']==scope
                    and ops[0]['status']=='returned' and ops[0]['error'] is None
                    and ops[0]['shape']==list(input_shape),'actual returned native resource join')
        raw=actions[5*step:5*step+5];token=((raw-mean)/std).reshape(10)
        outgoing=np.concatenate((history,token[None]),axis=0)[None]
        for k,want in dict(latent=state[None],history=history[None],raw=raw,mean=mean,std=std,
                           token=token,outgoing=outgoing).items():same(packet[k],want,'literal program packet '+k)
        same(ae_in['action_encoder:0'],outgoing,'actual action encoder input')
        encoded=ae_out['action_encoder:output'];require(encoded.shape==(1,3,192),'encoded action shape')
        same(pr_in['predict:act_emb'],encoded,'predict encoded actions')
        same(pr_in['predict:latent'],state[None],'predict rolled full history')
        same(inner_in['predictor:0'],state[None],'predictor history')
        same(inner_in['predictor:1'],encoded,'predictor actions')
        hidden=inner_out['predictor:output'];require(hidden.shape==(1,3,192),'predictor full3 output')
        same(proj_in['pred_proj:0'],hidden.reshape(3,192),'projection input reshape')
        full=pr_out['predict:output'];require(full.shape==(1,3,192),'native full3 forecast')
        same(proj_out['pred_proj:output'],full.reshape(3,192),'actual native prediction output')
        state=np.concatenate((state[1:],full[0,-1:]),axis=0);history=outgoing[0,1:]
        same(boundary['latent'],state[None],'shifted output history')
        same(boundary['history'],history[None],'shifted action history')
        path_state,path_history=context(winner['whole'][step])
        same(state,path_state,'original winner path latent bytes');same(history,path_history,'original path action bytes')
        certificates.append(dict(step=step,record_indices=[e['record_index'] for e in stream],
            predict_operation_index=stream[3]['payload']['operation_index'],
            input_latent_sha256=sha(pr_in['predict:latent'].tobytes()),
            input_act_emb_sha256=sha(encoded.tobytes()),full3_output_sha256=sha(full.tobytes()),
            predicted192=full[0,-1].tolist(),full3_output=full.tolist(),
            native_tensor_bytes=sum(a.nbytes for group in arrays for a in group.values())))
    return certificates


def contract():
    return dict(fixed_slots=8,known_roots=2,fresh_roots=0,starts=[10,15],ends=[15,20],
        banks=['endpoints','midpoints'],new_A_predict_cap=8,new_R_predict_cap=0,
        encoder=0,projector=0,AD=0,physics=0,training=False,no_retry=True,no_refund=True,
        runtime_included=False,wall_seconds=600,output_cap_bytes=64*1024**2,
        final_reserve_bytes=8*1024**2,disk_floor_bytes=FLOOR,metadata_cap_bytes=META,
        cuda_allocated_bytes=3*1024**3,cuda_reserved_bytes=4*1024**3,
        native_tensor_bytes_per_A=25840,all8_native_tensor_bytes=206720,
        total_FLOPs=None,equal_total_compute=False,P1_admission=False,P3_admission=False,
        C3=False,physical_release=False)


def resolve(p,r,receipt,planner_p,planner_r,planner_receipt,load_latent,load_journal):
    """Pure saved resolver; missing/invalid branch evidence yields explicit blocked slots."""
    require(planner_p['schema']==PLANNER+'.prepare' and planner_r['schema']==PLANNER+'.report'
            and planner_receipt['schema']==PLANNER+'.consume' and planner_receipt['complete'] is True
            and planner_receipt['status']=='saved_shared_panel_verified_NOT_admitted', 'original planner authority')
    require(planner_p['expected_model_hash']==p['expected_model_hash'] and
            all(planner_receipt[k] is False for k in ('P1_admission','P3_admission','C3')),
            'original frozen model/science boundary')
    # Pure arithmetic/identity checks only; does not invoke encoder or upstream consume.
    saved.analyze_documents(p,r,receipt,load_latent)
    chains=receipt['actual_context_chains'];require(len(chains)==4,'all actual context chains')
    jobs=[];journals={}
    for b,plan in enumerate(p['plans']):
        witness=plan['source_witness'];task=witness['task'];slot=b//2;bank=('endpoints','midpoints')[b%2]
        require(canonical(task)==canonical(planner_p['tasks'][slot]),'original planner task/normalizer/goal')
        chain=chains[b]
        require(type(chain['plan_index']) is type(chain['root_slot']) is int and
                (chain['plan_index'],chain['root_slot'],chain['bank'])==(b,slot,bank)
                and [x['step'] for x in chain['checkpoints']]==[5,10,15,20],'complete actual chain identities')
        commands=f32(plan['commands'],(20,2));raw10=f32(witness['request']['raw10'],(10,2))
        actual=deepcopy(witness['request']['task']['root_context']);context(actual)
        expected_actual={}
        for j,cp in enumerate((5,10,15,20)):
            spec=p['rows'][4*b+j];z=receipt['rows'][4*b+j]['actual192']
            history=np.concatenate((raw10,commands[5:cp]))[-10:]
            if z is None:actual=None
            elif actual is not None:
                actual=dict(actual,latent_history=(deepcopy(actual['latent_history']) if j==0 else
                    deepcopy(actual['latent_history'][1:])+[z]),action_history=normalization(task,history).tolist())
                if j==0:actual['latent_history'][-1]=z
            item=chain['checkpoints'][j]
            require(item['context_missing'] is (actual is None) and canonical(item['context'])==canonical(actual),
                    'actual full history independently reconstructed')
            require(canonical(spec['normalized_action_history'])==canonical(normalization(task,history).tolist()),
                    'once-only original normalizer actual history')
            expected_actual[cp]=deepcopy(actual)
        episode=f's{slot}_adaptive_{bank}'
        nodes=[n for n in planner_r['nodes'] if n['episode_id']==episode and n['boundary']==5]
        R=None;reason=None;ref=None
        try:
            require(len(nodes)==1,'exact original adaptive b5 node missing')
            node=nodes[0];require(node['source_slot']==slot and node['mode']=='adaptive_'+bank,'R source node identity')
            scope=dict(slot=slot,mode=node['mode'],horizon=3,index=witness['winner']['index'],phase='selected')
            selected,ref=load_journal(node,scope)
            R=certify_R(node,witness,selected)
            journals[episode]=ref
        except (ValueError,KeyError,OSError,TypeError,StopIteration) as exc:
            reason=type(exc).__name__+': '+str(exc)
        for start in (10,15):
            step=(start-5)//5;end=start+5;ca=expected_actual[start]
            cr=witness['winner']['whole'][step-1];zr,ar=context(cr)
            future=commands[start:end];history=np.concatenate((raw10,commands[5:start]))[-10:]
            same(ar,normalization(task,history),'rolled action history normalization')
            target=receipt['rows'][4*b+(end-5)//5]['actual192']
            reasons=[] if reason is None else ['R_unverified: '+reason]
            if ca is None:reasons.append('A_full_actual_context_missing')
            if target is None:reasons.append('actual_endpoint_label_missing')
            if ca is not None:
                _,aa=context(ca);same(aa,ar,'A/R same causal action history bytes')
            token=((future-f32(task['action_mean'],(2,)))/f32(task['action_std'],(2,))).reshape(10)
            outgoing=np.concatenate((ar,token[None]),axis=0)[None]
            jobs.append(dict(index=len(jobs),plan_index=b,root_slot=slot,bank=bank,task_id=plan['task_id'],
                start=start,end=end,status='ready' if not reasons else 'blocked',blocked_reasons=reasons,
                selection_digest=plan['selection_digest'],witness_sha256=witness['witness_sha256'],
                model_hash=p['expected_model_hash'],goal192=task['goal'],task_binding=task['binding'],
                full20=plan['commands'],full20_rawbits=commands.tobytes().hex(),
                future_raw5=future.tolist(),future_raw5_bits=future.tobytes().hex(),
                causal_raw10=history.tolist(),causal_raw10_bits=history.tobytes().hex(),
                normalizer=dict(mean=task['action_mean'],std=task['action_std'],dtype='float32',applications=1),
                R_context=cr,R_context_sha256=digest(cr),A_context=ca,
                A_context_sha256=None if ca is None else digest(ca),
                A_predict_input=dict(latent=None if ca is None else [ca['latent_history']],
                    outgoing_actions=outgoing.tolist(),raw5=future.tolist()),
                R_reuse_verified=R is not None,R_proof=None if R is None else R[step],R_journal=ref,
                label192=target,label_row_index=4*b+(end-5)//5,
                byte_budget=dict(A_native_tensor_bytes=25840,native_full3_output_bytes=2304),
                new_R_calls=0,new_A_calls_cap=1))
    require(len(jobs)==8,'all eight slots retained')
    payload_bytes=len(canonical(jobs))
    require(payload_bytes<=META,'eight job metadata fits cap')
    bounded_output=payload_bytes+275760+80*64*1024+8*1024**2
    require(bounded_output<=64*1024**2,'future complete IO and metadata byte bound')
    return dict(jobs=jobs,selected_journals=journals,
        fixed_denominator=8,ready=sum(j['status']=='ready' for j in jobs),
        blocked=sum(j['status']=='blocked' for j in jobs),job_serialized_bytes=payload_bytes,
        IO_budget=dict(all8_native_tensor_bytes=206720,base64_data_bytes_upper_bound=275760,
            metadata_cap_bytes=META,output_cap_bytes=64*1024**2,output_feasible=True,
            event_count=80,event_metadata_cap_bytes=64*1024,projected_bounded_output_bytes=bounded_output,
            remaining_output_bytes_for_JSON_metadata=64*1024**2-275760,
            assumption='future runtime uses the same10event per packet native trace; fixed no extra replay'))


def own_files():
    return [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]


def build(**arguments):
    require(set(arguments)=={k+s for k in KEYS for s in ('','_sha')},'six explicit paths and SHA inputs')
    bound={}
    def read(path,h):
        require(Path(path).is_absolute() and not Path(path).is_symlink(),'literal absolute regular input path')
        path=str(Path(path).resolve());data=saved.checked_bytes(path,h,META);bound[path]=h
        return saved.decode(data)
    p,r,receipt,pp,pr,pc=[read(arguments[k],arguments[k+'_sha']) for k in KEYS]
    for key,rep,rc in (('encoder',r,receipt),('planner',pr,pc)):
        require(Path(rep['prepared']).resolve()==Path(arguments[key+'_prepared']).resolve() and
                rep['prepare_sha256']==arguments[key+'_prepared_sha'] and
                rc['report_sha256']==arguments[key+'_report_sha'],'explicit actual authority chain')
    require(pc['prepare_sha256']==arguments['planner_prepared_sha'] and
            Path(pc['prepared']).resolve()==Path(arguments['planner_prepared']).resolve(), 'planner receipt prepare join')
    for path,h in PINS.items():saved.checked_bytes(path,h,META);bound[str(path)]=h
    # These exact upstream planner files must be in encoder's inherited physical closure.
    for key in ('planner_prepared','planner_report','planner_receipt'):
        require(p['bindings'][str(Path(arguments[key]).resolve())]==arguments[key+'_sha'],
                'encoder closure authenticates exact planner authority')
    folder=Path(arguments['encoder_report']).resolve().parent
    def latent(d):
        a=saved.load_latent(folder,d,r['artifact_sha256']);bound[str(folder/d['file'])]=d['sha256'];return a
    planner_dir=Path(arguments['planner_report']).resolve().parent
    def journal(n,scope):
        path=planner_dir/(n['episode_id']+'_b05')/'events.jsonl'
        data,ref=read_selected_journal(path,n['native_journal_sha256'],n['native_journal_tail'],scope)
        bound[str(path)]=ref['sha256'];return data,ref
    result=resolve(p,r,receipt,pp,pr,pc,latent,journal)
    for f in own_files():bound[str(f)]=sha(f.read_bytes())
    for f,h in bound.items():saved.checked_bytes(f,h,JOURNAL_CAP)
    return dict(schema=NAME+'.prepare',arguments=arguments,contract=contract(),bindings=bound,**result,
        status='all8_saved_preconditions_ready_NOT_execution' if result['blocked']==0 else 'blocked_saved_preconditions',
        evidence_scope=dict(inherits_MAIN_complete_encoder_and_planner_receipts=True,
            full_upstream_strict_rebuilt_here=False,new_neural_parity=False,
            checked_here='authority bytes, actual histories, selected R native IO/journal/resource joins, byte budgets'),
        inherited_resources=dict(encoder=receipt['counts'],planner=pr['per_episode_neural']),
        resources_here=dict(predict=0,encoder=0,projector=0,AD=0,physics=0),
        P1_admission=False,P3_admission=False,C3=False,physical_release=False)


def inspect(prepared,expected):
    p=saved.decode(saved.checked_bytes(prepared,expected,META))
    require(p['schema']==NAME+'.prepare','own prepare schema')
    require(canonical(p)==canonical(build(**p['arguments'])),'deterministic saved precondition reconstruction')
    return dict(schema=NAME+'.inspect',status=p['status'],prepare_sha256=expected,
        fixed_denominator=8,ready=p['ready'],blocked=p['blocked'],
        slots=[{k:j[k] for k in ('index','plan_index','start','end','status','blocked_reasons','R_reuse_verified')}
               for j in p['jobs']],production_execution=False)


def write(path,value):
    path=Path(path).absolute();data=canonical(value)+b'\n'
    require(path.name.startswith(NAME+'_') and len(data)<=META and not path.is_symlink()
            and shutil.disk_usage(path.parent).free>=len(data)+FLOOR,'exclusive own metadata cap/floor')
    with path.open('xb') as f:
        require(f.write(data)==len(data),'full metadata write');f.flush();os.fsync(f.fileno())
    fd=os.open(path.parent,os.O_DIRECTORY|os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return dict(path=str(path),sha256=sha(data),status=value['status'])


def main():
    q=argparse.ArgumentParser(description=__doc__);subs=q.add_subparsers(dest='command',required=True)
    prep=subs.add_parser('prepare')
    for k in KEYS:
        prep.add_argument('--'+k.replace('_','-'),required=True)
        prep.add_argument('--'+k.replace('_','-')+'-sha',required=True)
    prep.add_argument('--output',required=True)
    check=subs.add_parser('inspect');check.add_argument('--prepared',required=True);check.add_argument('--prepare-sha',required=True)
    args=vars(q.parse_args());cmd=args.pop('command')
    if cmd=='prepare':output=args.pop('output');v=write(output,build(**args))
    else:v=inspect(args['prepared'],args['prepare_sha'])
    print(canonical(v).decode())


if __name__=='__main__':main()
