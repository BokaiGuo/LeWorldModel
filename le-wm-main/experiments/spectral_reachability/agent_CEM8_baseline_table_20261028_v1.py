"""Project Hegel's strict saved CEM8 analysis into long tables; no analysis runner.

Only standard-library metadata/CSV work. No model, consumer, simulator or numpy.
Input analysis/source/report SHAs must be actual external MAIN authorities.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil

NAME = 'agent_CEM8_baseline_table_20261028_v1'
ANALYZER = 'agent_CEM8_physical_analysis_20261028_v1'
CEM = 'agent_moving_CEM8_20261028_v1'
TEACHER = 'capability_suffix_factorial_cuda_MAIN_20261027_v2'
ORDER = [(s,c,a) for s in (0,1) for c in ('off','on') for a in ('line_search_GN','full10_conditional_D2P')]
METHODS = ('cem','spectral_teacher_full')
ENDPOINTS = ('endpoint','horizon10')
CAP, FLOOR = 32*1024**2, 200*1024**2
METRICS = ('joint_xy_error','native7_distance','native7_reward','agent_xy_error','object_xy_error',
           'velocity_error','angle_wrapped_error','joint_xy_wrapped_angle_error')


def require(ok,message):
    if not ok: raise ValueError(message)


def actual_hash(h):
    require(type(h) is str and len(h)==64 and h!='0'*64 and all(c in '0123456789abcdef' for c in h), 'actual nonzero SHA256')


def raw(value):
    return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()


def sha(path):
    p=Path(path); require(p.is_absolute() and p.resolve()==p and p.is_file() and not p.is_symlink(), 'canonical regular absolute file')
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''): h.update(block)
    return h.hexdigest()


def read_bound(path,expected):
    actual_hash(expected); p=Path(path)
    require(sha(p)==expected and p.stat().st_size<=CAP,'saved input SHA/cap')
    def pairs(items):
        out={}
        for k,v in items:
            require(k not in out,'duplicate JSON key'); out[k]=v
        return out
    body=p.read_bytes(); require(len(body)<=CAP and hashlib.sha256(body).hexdigest()==expected,'read SHA/cap')
    value=json.loads(body,object_pairs_hook=pairs,parse_constant=lambda _:require(False,'nonfinite JSON'))
    require(type(value) is dict,'JSON object'); raw(value); return value


def checked_count(value):
    require(type(value) is int and value>=0,'nonnegative literal resource count'); return value


def ledger_axes(entries,predict_name):
    """Count saved ledger entry/return and true batch dimension, not declarations."""
    axes={}
    for label,resource in (('predict_API_calls',predict_name),('action_encoder_API_calls','action_encoder'),('pred_proj_API_calls','pred_proj')):
        selected=[e for e in entries if e['resource']==resource]
        require(selected and all(type(e['completed']) is bool for e in selected),'complete native ledger metadata')
        axes[label]=dict(attempted=len(selected),completed=sum(e['completed'] for e in selected))
        if resource==predict_name:
            sizes=[]
            for e in selected:
                shape=e['inputs'][0]['shape']
                require(len(shape)==3 and shape[1:]==[3,192] and type(shape[0]) is int and shape[0]>0,'actual predictor batch dimension')
                sizes.append(shape[0])
            axes['predictor_forward_samples']=dict(attempted=sum(sizes),completed=sum(n for n,e in zip(sizes,selected,strict=True) if e['completed']))
            axes['predictor_batch_sizes']=sorted(set(sizes))
    return axes


def budgets(cem,teacher):
    require(cem['schema']==CEM+'.report' and teacher['schema']==TEACHER+'.report' and
            len(cem['rows'])==8 and len(teacher['rows'])==24 and cem['errors']==teacher['errors']==[], 'actual complete compiler8/24 ABI')
    result=[]
    for i,key in enumerate(ORDER):
        row=cem['rows'][i]
        require(type(row['index']) is type(row['source_slot']) is int and row['index']==i and
                (row['source_slot'],row['condition'],row['prefix_arm'])==key and row['status']=='suffix_sealed','CEM8 ordinal/key')
        ca=ledger_axes(row['module_ledger'],'predict')
        for k,label in (('predict','predict_API_calls'),('action_encoder','action_encoder_API_calls'),('pred_proj','pred_proj_API_calls')):
            require(row['resources'][k]==ca[label],'actual CEM ledger/resource join')
        # Explicit producer declarations, not inference from absent hooks.
        ca.update(jacobian_calls=cem['contract'].get('jacobian'),mathematical_VJP_directions=cem['contract'].get('VJP'),
                  planning_SVD_calls=cem['contract'].get('SVD'),AD_forward_API_calls=cem['contract'].get('AD'))
        ordinal=3*i+1
        require(type(teacher['rows'][ordinal]['index']) is int and teacher['rows'][ordinal]['index']==ordinal and
                teacher['rows'][ordinal]['status']=='sealed','teacher_full exact ordinal')
        entries=[e for e in teacher['ledger'] if e['scope']['plan_index']==ordinal]
        ta=ledger_axes(entries,'predictor')
        ops=[e for e in teacher['operations'] if e['case']==ordinal]
        require(all(e['completed'] is True and e['kind'] in ('jacobian','svd') for e in ops),'returned fixed-helper operations')
        j=sum(e['kind']=='jacobian' for e in ops)
        ta.update(jacobian_calls=j,planning_SVD_calls=sum(e['kind']=='svd' for e in ops),
                  mathematical_VJP_directions=None,AD_forward_API_calls=None)
        # Direction count is a mathematical contract unit, NOT backward API count.
        if teacher['contract'].get('jacobians')==24 and teacher['contract'].get('mathematical_VJP_directions')==4608:
            ta['mathematical_VJP_directions']=j*192
        for method,axes in zip(METHODS,(ca,ta),strict=True):
            for k in ('jacobian_calls','mathematical_VJP_directions','planning_SVD_calls','AD_forward_API_calls'):
                if axes[k] is not None: checked_count(axes[k])
            axes.update(backward_engine_API_calls=None,JVP_calls=None,encoder_API_calls=None,projector_API_calls=None,
                        model_physics_calls=None,wall_seconds=None,FLOPs=None)
            result.append(dict(anchor_index=i,method=method,compiler_ordinal=i if method=='cem' else ordinal,
                resource_row_id=str(i)+'|'+method,
                scope='new suffix selection including nominal/AD-forward/candidates/replay; excludes inherited work',
                axes=axes,provenance='actual saved module ledger; J/VJP/AD/SVD explicit declarations or returned helper receipts'))
    return result


def project(analysis,cem,teacher):
    """RAM projection seam, not an alternative physical/scientific validator."""
    require(analysis['schema']==ANALYZER+'.analysis' and analysis['status']=='saved_fixed8_analysis_not_admitted' and
            type(analysis['fixed_denominator']) is int and analysis['fixed_denominator']==len(analysis['rows'])==8 and
            analysis['known_roots']==2 and analysis['fresh_roots']==0,'strict Hegel fixed8 format only')
    require(all(analysis[k] is False for k in ('P1_admission','scientific_admission','independent_confirmation','equal_total_compute','equal_FLOPs','freshness')),
            'no scientific/compute upgrade')
    require([len(analysis['consumes'][k]['rows']) for k in ('cem8','spectral24')]==[8,24],'complete strict consumes, not observed4')
    resource_rows=budgets(cem,teacher); rows=[]
    for i,pair in enumerate(analysis['rows']):
        require(type(pair['index']) is type(pair['root_slot']) is type(pair['spectral_ordinal']) is int and pair['index']==i and
                (pair['root_slot'],pair['condition'],pair['prefix_arm'])==ORDER[i] and pair['spectral_ordinal']==3*i+1,'stable eight-anchor long-table order')
        for k in ('task_id','selection_digest','anchor_sha256'): actual_hash(pair[k])
        for m in METHODS:
            ordinal=i if m=='cem' else 3*i+1
            saved=analysis['consumes']['cem8' if m=='cem' else 'spectral24']['rows'][ordinal]
            model=cem['rows'][i]['plan'] if m=='cem' else None
            plan=saved['selected_suffix']['plan']
            require(saved['task_id']==pair['task_id'] and saved['selection_digest']==pair['selection_digest'] and
                    plan['task_id']==pair['task_id'] and (m=='cem' or saved['variant']==plan['variant']=='teacher_full'), 'full task/selection/method provenance')
            if model is not None: require(all(plan[k]==v for k,v in model.items()),'actual CEM model plan join')
            else: require(plan['winner_index']==teacher['rows'][ordinal]['result']['winner']['index'] and
                          plan['predicted_Q']==teacher['rows'][ordinal]['result']['winner']['Q'], 'own teacher winner/modelQ join')
            value=pair[m]; n=value['returned_suffix_steps']; checked_count(n)
            require(n<=5 and n==len(saved['responses']) and value['status']==saved['status'],'actual suffix return count, not full5 fill')
            for endpoint in ENDPOINTS:
                metric=value[endpoint]
                require(not pair['identity_unknown'] or metric is None,'identity unknown must remain unscored')
                require(endpoint!='horizon10' or n==5 or metric is None,'no padding early-stop horizon10')
                flat=dict(anchor_index=i,root_slot=pair['root_slot'],condition=pair['condition'],prefix_arm=pair['prefix_arm'],method=m,
                    endpoint_kind=endpoint,task_id=pair['task_id'],selection_digest=pair['selection_digest'],anchor_sha256=pair['anchor_sha256'],
                    physical_status=value['status'],actual_suffix_returned_steps=n,suffix_attempted_steps=saved.get('suffix_attempted'),
                    identity_unknown=deepcopy(pair['identity_unknown']),missing_checkpoints=deepcopy(value['missing_checkpoints']),
                    first_error=deepcopy(value['first_error']),secondary_errors=deepcopy(value['secondary_errors']),
                    observed_endpoint_metrics=deepcopy(value['observed_endpoint']),
                    physical_report_sha256=analysis['consumes']['cem8' if m=='cem' else 'spectral24']['physical_report_sha256'],
                    compiler_report_sha256=analysis['consumes']['cem8' if m=='cem' else 'spectral24']['compiler_report_sha256'],
                    metric_available=metric is not None,model_Q=plan['model_Q'] if m=='cem' else plan['predicted_Q'],
                    selected_plan_sha256=hashlib.sha256(raw(plan)).hexdigest())
                budget=resource_rows[2*i+(m=='spectral_teacher_full')]
                flat.update(resource_row_id=budget['resource_row_id'],model_budget_axes=deepcopy(budget['axes']),
                            budget_repeated_across_endpoint_kinds=True)
                for k in METRICS:
                    v=None if metric is None else metric[k]
                    require(v is None or (type(v) in (int,float) and math.isfinite(v)), 'finite metric or null')
                    flat[k]=v
                for k,source in (('native_success','native_success'),('wrapped_joint_success','wrapped_allxy_success')):
                    v=None if metric is None else metric['native'][source]
                    require(v is None or type(v) is bool,'literal success/null'); flat[k]=v
                rows.append(flat)
    return dict(schema=NAME+'.table',status='strict_analysis_projection_not_admitted',fixed_anchor_denominator=8,
        methods=list(METHODS),endpoint_kinds=list(ENDPOINTS),rows=rows,resource_rows=resource_rows,
        resource_groups=dict(
            cem8=dict(compiler_resources=deepcopy(cem['resources']),wall_seconds=cem.get('wall_seconds'),
                      declared_contract=deepcopy(cem['contract'])),
            factorial_all24=dict(axes=ledger_axes(teacher['ledger'],'predictor'),predict_API_calls=teacher['predict_API'],
                                 wall_seconds=teacher.get('wall_seconds'),declared_contract=deepcopy(teacher['contract']),contains_teacher8=True),
            teacher8=dict(subset_of='factorial_all24',compiler_ordinals=list(range(1,24,3)),add_to_parent=False),
            inherited_shared_once=dict(declarations=deepcopy(cem['contract'].get('shared_resource_axes')),
                teacher8_subset_of='factorial_all24',add_to_parent=False,total_historical_cost=None,
                note='shared encodes/source/root-prefix history reused; no per-row allocation, subtraction, refund or total')),
        physical_resources_once={k:deepcopy(analysis['consumes'][k]['counts']) for k in ('cem8','spectral24')},
        interpretation='known2 roots/8 anchors; endpoint and H10 distinct; budgets differ, not equal total compute or FLOPs',
        known_roots=2,fresh_roots=0,P1_admission=False,scientific_admission=False,equal_total_compute=False,equal_FLOPs=False,
        metric_authority=ANALYZER,metric_recomputed_here=False,new_neural_calls=0,new_physics_calls=0)


def load(analysis_path,analysis_sha,analyzer_source_sha,cem_path,cem_sha,teacher_path,teacher_sha):
    analysis=read_bound(analysis_path,analysis_sha); actual_hash(analyzer_source_sha)
    source=Path(__file__).resolve().with_name(ANALYZER+'.py')
    bindings=deepcopy(analysis['input_sha256'])
    require(bindings[str(source)]==analyzer_source_sha==sha(source),'MAIN actual analyzer source SHA and strict inventory')
    for path,h in bindings.items(): actual_hash(h); require(sha(path)==h,'complete strict-analysis source/input inventory')
    for key,path,h in (('cem8',cem_path,cem_sha),('spectral24',teacher_path,teacher_sha)):
        require(bindings[str(Path(path))]==h==analysis['consumes'][key]['compiler_report_sha256'], 'actual compiler SHA in strict analysis closure')
    cem,teacher=read_bound(cem_path,cem_sha),read_bound(teacher_path,teacher_sha)
    result=project(analysis,cem,teacher)
    for path,h in bindings.items(): require(sha(path)==h,'final strict input closure drift')
    require(sha(analysis_path)==analysis_sha,'analysis drift')
    bindings[str(Path(analysis_path))]=analysis_sha
    for path in (Path(__file__).resolve(),Path(__file__).resolve().parents[2]/'tests'/('test_'+NAME+'.py'),
                 Path(__file__).resolve().parents[2]/'innerdocs'/(NAME+'.md')): bindings[str(path)]=sha(path)
    result.update(input_sha256=bindings,input_directories=deepcopy(analysis['input_directories']),
                  analysis_path=str(Path(analysis_path)),analysis_sha256=analysis_sha,analyzer_source_sha256=analyzer_source_sha)
    for row in result['rows']: row.update(analysis_sha256=analysis_sha,analyzer_source_sha256=analyzer_source_sha)
    require(len(raw(result))<=CAP,'complete export cap, no thinning'); return result


def csv_bytes(result):
    """One row per anchor/method/endpoint. Literal null/false, never blank/zero fill."""
    out=io.StringIO(newline=''); fields=list(result['rows'][0]); writer=csv.DictWriter(out,fieldnames=fields,lineterminator='\n')
    writer.writeheader()
    for row in result['rows']:
        writer.writerow({k:json.dumps(v,separators=(',',':'),allow_nan=False) if not isinstance(v,str) else v for k,v in row.items()})
    return out.getvalue().encode()


def write_new(path,body,result):
    p=Path(path)
    require(p.is_absolute() and p.parent.resolve()==p.parent and p.parent.is_dir() and p.name.startswith(NAME) and
            not p.exists() and not p.is_symlink(),'exclusive new same-prefix output')
    require(len(body)<=CAP and shutil.disk_usage(p.parent).free>=FLOOR+len(body),'output cap/200MiB floor')
    for f in result['input_sha256']:
        require(p!=Path(f),'no input overwrite')
    for directory in result['input_directories']:
        require(not p.is_relative_to(Path(directory)),'no writes inside physical producer output')
    with p.open('xb') as stream:
        require(stream.write(body)==len(body),'complete export'); stream.flush(); os.fsync(stream.fileno())
    fd=os.open(p.parent,os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)
    return hashlib.sha256(body).hexdigest()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('analysis','analysis-sha','analyzer-source-sha','cem-compiler-report','cem-compiler-report-sha',
                 'teacher-compiler-report','teacher-compiler-report-sha','output'): p.add_argument('--'+flag,required=True)
    p.add_argument('--format',choices=('json','csv'),default='json'); a=p.parse_args(argv)
    require(not Path(a.output).exists() and not Path(a.output).is_symlink(),'NEW output before expensive hashes')
    r=load(a.analysis,a.analysis_sha,a.analyzer_source_sha,a.cem_compiler_report,a.cem_compiler_report_sha,a.teacher_compiler_report,a.teacher_compiler_report_sha)
    body=raw(r) if a.format=='json' else csv_bytes(r)
    print(raw(dict(output=a.output,sha256=write_new(a.output,body,r),rows=len(r['rows']),format=a.format,status=r['status'])).decode(),end='')


if __name__=='__main__': main()
