"""Independent saved FD request/context/resource/numeric audit; no CUDA imports.

Does not import the FD producer or call its requests/analyze/compare helpers.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zlib

import numpy as np

from experiments.spectral_reachability import p1_line_search_gn_saved_audit_20261027_v1 as source

NAME = 'p1_line_search_gn_fd_saved_audit_20261027_v1'
ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'results/p1_line_search_gn_fd_cuda_20261027_v1_cuda_main01'
REPORT_SHA = '1dd13d4f8ce4ac2bcdee7e8e360ef283962bae11314efb19f017cbeb92d44163'
PREPARE_SHA = '29b70d36fb5a4f49f22474a9fc5932f566c2eaab7357b4de70d2411389e974b0'
EPS = (1e-3, 3e-3, 1e-2)
require, array = source.require, source.array


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def exact(actual, expected, label):
    require(json.dumps(actual, sort_keys=True, allow_nan=False) == json.dumps(expected, sort_keys=True, allow_nan=False), label)


def decode(raw):
    def pairs(items):
        out = {}
        for k,v in items:
            require(k not in out, 'duplicate JSON key'); out[k] = v
        return out
    def bad(v): raise ValueError('nonfinite JSON constant: '+v)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=bad)


def gunzip(raw, limit=16*1024**2):
    require(len(raw) <= 2*1024**2, 'compressed input cap')
    d = zlib.decompressobj(31); plain = d.decompress(raw, limit+1)
    require(len(plain) <= limit and d.eof and not d.unused_data and not d.unconsumed_tail, 'complete bounded single gzip, no trailing bytes')
    return plain


def directions():
    a, b = np.zeros(20), np.zeros(20); a[0] = b[10] = 1.
    c = np.array([1,-1,1,1,-1,-1,1,-1,-1,1,-1,1,1,-1,1,-1,-1,1,1,-1], float)/np.sqrt(20.)
    return [dict(id=k, vector=v.tolist()) for k,v in zip(('packet0_e0','packet1_e0','fixed20_sign_unit'), (a,b,c))]


def requests(points):
    rows = []
    require(len(points) == 8 and [(p['root_index'],p['iteration']) for p in points] == [(r,i) for r in (0,1) for i in (1,2,3,4)], 'all eight pre-update points')
    for i, p in enumerate(points):
        require(p['point_order'] == i, 'point order')
        base = array(p['at_logits'], (2,10)); require(np.array_equal(base,base.astype(np.float32)), 'literal float32 base')
        common = {k:p[k] for k in ('point_order','root_index','root_sha256','task_id','iteration')}
        x = base.astype(np.float32)
        rows.append(dict(common, order=len(rows), mode='nominal', direction=None, epsilon=None, sign=0,
            logits=x.astype(float).tolist(), commands=np.tanh(x).reshape(10,2).astype(float).tolist()))
        for direction in directions():
            v = np.asarray(direction['vector']).reshape(2,10)
            for eps in EPS:
                for sign in (-1,1):
                    x = (base+sign*eps*v).astype(np.float32)
                    rows.append(dict(common, order=len(rows), mode='central_side', direction=direction['id'], epsilon=eps, sign=sign,
                        logits=x.astype(float).tolist(), commands=np.tanh(x).reshape(10,2).astype(float).tolist()))
    return rows


def reconstruct_context(task, previous, raw_packet, terminal):
    terminal = array(terminal,(192,))
    require(np.array_equal(terminal,terminal.astype(np.float32)), 'native terminal float32 representability')
    token = (np.asarray(raw_packet,np.float32).reshape(10)-np.tile(np.asarray(task['action_mean'],np.float32),5))/np.tile(np.asarray(task['action_std'],np.float32),5)
    require(np.isfinite(token).all(), 'finite original normalizer token')
    return dict(layout='states_with_between_state_actions_v1', context_length=3, latent_dim=192, action_dim=10,
        latent_history=previous['latent_history'][1:]+[terminal.tolist()],
        action_history=previous['action_history'][1:]+[token.astype(float).tolist()])


def consume(events, points, tasks):
    wanted = requests(points); cursor = 0; completed = []; module_index = 0
    counts = {k:dict(attempted=0,completed=0) for k in ('predict','action_encoder')}
    def take():
        nonlocal cursor
        require(cursor < len(events), 'truncated event census')
        e = events[cursor]; cursor += 1; return e
    for req in wanted:
        exact(take(),dict(kind='evaluation_attempt',request=req),'independent request reconstruction')
        previous = tasks[req['root_index']]['root_context']; hashes = []; contexts = []
        for packet in (0,1):
            for resource in ('action_encoder','predict'):
                entry = dict(index=module_index,resource=resource,samples=1,completed=False,evaluation_order=req['order'],packet=packet)
                exact(take(),dict(kind='module_attempt',entry=entry),'ordered predispatch module attempt')
                counts[resource]['attempted'] += 1
                exact(take(),dict(kind='module_completed',entry=dict(entry,completed=True)),'unique synchronized module completion')
                counts[resource]['completed'] += 1; module_index += 1
            e = take()
            require(set(e) == {'kind','evaluation_order','packet','context_sha256','terminal'} and
                e['kind'] == 'native_packet' and e['evaluation_order'] == req['order'] and e['packet'] == packet, 'fullH2 packet receipt identity')
            c = reconstruct_context(tasks[req['root_index']],previous,req['commands'][5*packet:5*packet+5],e['terminal'])
            require(digest(c) == e['context_sha256'], 'full context independently reconstructed from terminal/token/history')
            hashes.append(digest(c)); contexts.append(c); previous = c
        row = dict(req, terminal=previous['latent_history'][-1], context_sha256=hashes)
        exact(take(),dict(kind='evaluation_complete',evaluation=row),'all152 completed evaluations and context receipts')
        if req['mode'] == 'nominal':
            p = points[req['point_order']]
            exact(row['terminal'],p['at_terminal'],'eight exact AD nominal anchors')
            exact(hashes,p['saved_context_sha256'],'both complete original AD contexts anchored')
        completed.append(row)
    require(cursor == len(events) == 1824 and module_index == 608 and len(completed) == 152, 'no missing/extra/refunded events')
    exact(counts,{k:dict(attempted=304,completed=304) for k in counts},'complete raw resource counts')
    return completed, counts


def error_metrics(fd, reference):
    fd, reference = array(fd,(192,)),array(reference,(192,))
    delta = fd-reference; n=float(np.linalg.norm(reference)); f=float(np.linalg.norm(fd)); err=float(np.linalg.norm(delta))
    return dict(absolute_l2=err,absolute_linf=float(np.max(np.abs(delta))),absolute_rmse=float(np.sqrt(np.mean(delta**2))),
        reference_l2=n,fd_l2=f,relative_l2=err/n if n else None,symmetric_relative_l2=err/max(n,f) if max(n,f) else None,
        cosine=float(fd@reference/(n*f)) if n and f else None)


def recompute(points, matrices, rows):
    comparisons, stability = [], []; cursor = 0
    for point in points:
        cursor += 1 # nominal already checked by consume
        j = array(matrices[point['point_order']],(192,20))
        for d in directions():
            v=np.asarray(d['vector']); group=[]
            for eps in EPS:
                minus, plus = rows[cursor:cursor+2]; cursor += 2
                m,p = array(minus['terminal'],(192,)),array(plus['terminal'],(192,))
                fd = (p-m)/(2*eps)
                effective=(np.asarray(plus['logits']).reshape(20)-np.asarray(minus['logits']).reshape(20))/(2*eps)
                reference=j@v; effective_reference=j@effective
                c=dict(point_order=point['point_order'],root_index=point['root_index'],iteration=point['iteration'],direction=d['id'],epsilon=eps,
                    fd=fd.tolist(),saved_Jv=reference.tolist(),effective_float32_Jv=effective_reference.tolist(),effective_direction=effective.tolist(),
                    effective_direction_error_l2=float(np.linalg.norm(effective-v)),midpoint_drift_l2=float(np.linalg.norm((p+m)/2-np.asarray(point['at_terminal']))),
                    requested_direction_metrics=error_metrics(fd,reference),effective_direction_metrics=error_metrics(fd,effective_reference))
                group.append(c)
            comparisons.extend(group)
            stability.append(dict(point_order=point['point_order'],direction=d['id'],adjacent_epsilon_fd_l2=[
                float(np.linalg.norm(np.asarray(a['fd'])-b['fd'])) for a,b in zip(group,group[1:])]))
    require(cursor == len(rows) == 152 and len(comparisons) == 72, 'full numeric denominator')
    return dict(schema='p1_line_search_gn_fd_cuda_20261027_v1.analysis',comparisons=comparisons,epsilon_stability=stability,
        nominal_anchors=8,scientific_pass_threshold=None,derivative_verdict='not_assigned',P1_admission=False)


def numeric_equal(actual, expected):
    """Fixed receipt-arithmetic tolerance only; never a derivative pass threshold."""
    if isinstance(expected,dict):
        require(isinstance(actual,dict) and set(actual)==set(expected),'analysis exact fields')
        for k,v in expected.items(): numeric_equal(actual[k],v)
    elif isinstance(expected,list):
        require(isinstance(actual,list) and len(actual)==len(expected),'analysis full array census')
        for a,e in zip(actual,expected,strict=True): numeric_equal(a,e)
    elif type(expected) is float:
        require(type(actual) in (int,float) and np.isfinite(actual) and np.isclose(actual,expected,rtol=1e-12,atol=1e-12),'independent FD/Jv scalar or vector mismatch')
    else:
        require(type(actual) is type(expected) and actual==expected,'analysis identity/scope mismatch')


def summary(rows):
    def stats(values):
        v=[x for x in values if x is not None]
        return dict(defined=len(v),undefined=len(values)-len(v),minimum=float(min(v)) if v else None,
            median=float(np.median(v)) if v else None,maximum=float(max(v)) if v else None)
    return dict(count=len(rows),requested_relative_l2=stats([r['requested_direction_metrics']['relative_l2'] for r in rows]),
        effective_relative_l2=stats([r['effective_direction_metrics']['relative_l2'] for r in rows]),
        effective_cosine=stats([r['effective_direction_metrics']['cosine'] for r in rows]),
        effective_absolute_linf=stats([r['effective_direction_metrics']['absolute_linf'] for r in rows]))


def execute(report_path=RUN/'report.json', expected=REPORT_SHA):
    require(isinstance(expected,str) and len(expected)==64 and all(c in '0123456789abcdef' for c in expected),'external report SHA required')
    bindings={}
    def bound(path,h):
        path=Path(path).absolute(); require(source.sha(path)==h,'external/bound file SHA: '+str(path))
        require(str(path) not in bindings or bindings[str(path)]==h,'conflicting binding'); bindings[str(path)]=h
        require(path.stat().st_size<=2*1024**2,'small run input')
        raw=path.read_bytes(); require(hashlib.sha256(raw).hexdigest()==h,'read/hash race')
        return raw
    path=Path(report_path).absolute(); folder=path.parent; r=decode(bound(path,expected))
    require(r['schema']=='p1_line_search_gn_fd_cuda_20261027_v1.report' and r['status']=='cuda_fd_completed' and
        r['errors']==[] and r['first_error'] is None and r['forbidden_AD']==[],'complete actual FD report')
    require(r['prepare_sha256']==PREPARE_SHA and r['input_line_search_report_sha256']==source.INPUT_SHA,'fixed external source protocol')
    require(set(r['artifact_sha256'])=={'analysis.json.gz','events.jsonl.gz','binding.json'} and
        {f.name for f in folder.iterdir()}==set(r['artifact_sha256'])|{'report.json'},'exact output file census')
    require(sum(f.stat().st_size for f in folder.iterdir())<=r['output_cap_bytes']==2*1024**2 and r['floor_bytes']==200*1024**2,'actual output cap and declared floor')
    artifacts={name:bound(folder/name,h) for name,h in r['artifact_sha256'].items()}
    binding=decode(artifacts['binding.json']); require(binding['prepare_sha256']==PREPARE_SHA,'execution prepare binding')
    p=decode(bound(binding['prepared'],PREPARE_SHA))
    exact(p['directions'],directions(),'three precommitted directions'); exact(p['epsilons'],list(EPS),'three fixed eps'); exact(p['signs'],[-1,1],'both signs')
    b=source.load_bundle(); saved=source.audit(b)
    for name,h in {**b['bindings'],**p['bindings_sha256']}.items():
        require(source.sha(name)==h,'saved source/model/producer binding: '+name); bindings[name]=h
    tasks={i:t for i,t in enumerate(b['tasks'])}; points=[]; matrices={}
    for root in b['prepare']['roots']:
        events=[e for e in b['events'] if e['root_index']==root['root_index']]
        ds=[e for e in events if e['kind']=='derivative_complete']; require(len(ds)==4,'four original fresh J')
        for it,e in enumerate(ds,1):
            require(e['evaluation_id']==f'iter_{it}_AD','pre-update derivative point')
            d=e['derivative']; j=array(d['raw_terminal_jacobian'],(192,20))
            contexts=[v['context'] for v in events if v['kind']=='jacobian_forward_packet' and v['evaluation_id']==e['evaluation_id']]
            require(len(contexts)==2,'two original AD contexts')
            point=dict(point_order=len(points),root_index=root['root_index'],root_sha256=root['root_sha256'],task_id=root['task_id'],
                task_path=root['task_path'],iteration=it,evaluation_id=e['evaluation_id'],at_logits=d['at_logits'],at_terminal=d['at_terminal'],
                saved_context_sha256=[digest(c) for c in contexts],logit_J_sha256=digest(j.tolist()),
                saved_Jv={v['id']:(j@np.asarray(v['vector'])).tolist() for v in directions()})
            matrices[len(points)]=j; points.append(point)
    exact(p['points'],points,'all eight original fresh logit J/Jv/nominal bindings')
    require(r['model_hash_before']==r['model_hash_after']==p['expected_model_hash']==b['prepare']['expected_model_hash'] and
        r['model_unchanged'] is r['source_verified'] is True,'bound unchanged model receipt')
    for k in ('P1_admission','training','new_Q'): require(r[k] is p[k] is False,'no scope upgrade')
    for k in ('new_AD_calls','new_images','new_physics','independent_sample_increment'): require(type(r[k]) is int and r[k]==0,'no new FD AD/images/physics/sample')
    require(r['derivative_verdict']=='not_assigned' and r['scientific_pass_threshold'] is None,'no derivative pass threshold')
    exact(r['budget'],p['budget'],'budget declaration binding')
    wanted_budget=dict(points=8,directions=3,epsilons=3,central_comparisons=72,signed_H2_evaluations=144,nominal_H2_evaluations=8,
        total_H2_evaluations=152,predict=304,action_encoder=304,FD_predict=288,nominal_predict=16,AD_calls=0,new_jacobians=0,new_vjp_directions=0,
        encoder=0,projector=0,physics_steps=0,inherited_line_search_predict=156,inherited_line_search_action_encoder=156,
        inherited_AD_forward_predicts_already_inside156=16,inherited_jacobians=8,inherited_mathematical_vjp_directions=1536,
        line_search_plus_FD_predict=460,line_search_plus_FD_action_encoder=460)
    require(all(type(r['budget'][k]) is type(v) and r['budget'][k]==v for k,v in wanted_budget.items()),'all new and inherited budgets including ADforward')
    exact(r['inherited_line_search_resources'],saved['resources'],'actual inherited line-search counts')
    for k in ('inherited_B8_resources_once','inherited_image_resources_once'): exact(r[k],p[k],'upstream charged separately once')
    raw=gunzip(artifacts['events.jsonl.gz']); require(raw.endswith(b'\n'),'no torn journal suffix')
    events=[decode(line) for line in raw.splitlines()]
    rows,counts=consume(events,points,tasks); exact(r['resources'],counts,'report vs raw608 module counts')
    require(r['evaluations_completed']==152 and r['saved_validation_calls']==dict(attempted=2,completed=2) and
        r['completed_saved_validation_numpy20x20_solves']==16,'all evaluations and source validation work')
    recomputed=recompute(points,matrices,rows)
    require(r['analysis_sha256']==r['artifact_sha256']['analysis.json.gz'],'analysis hash binding')
    numeric_equal(decode(gunzip(artifacts['analysis.json.gz'])),recomputed)
    comparisons=recomputed['comparisons']
    by_eps=[dict(epsilon=eps,**summary([c for c in comparisons if c['epsilon']==eps])) for eps in EPS]
    by_direction=[dict(direction=d['id'],**summary([c for c in comparisons if c['direction']==d['id']])) for d in directions()]
    grid=[dict(epsilon=eps,direction=d['id'],**summary([c for c in comparisons if c['epsilon']==eps and c['direction']==d['id']])) for d in directions() for eps in EPS]
    for name,h in bindings.items(): require(source.sha(name)==h,'final source/input hash drift')
    return dict(schema=NAME+'.report',status='saved_fd_arithmetic_verified_not_admitted',input_report_sha256=expected,
        bindings_sha256=bindings,denominators=dict(points=8,nominal_anchors=8,evaluations=152,central_comparisons=72,
            reconstructed_full_contexts=304,module_attempts=608,module_completions=608,events=len(events)),
        resources=counts,inherited_line_search_resources=saved['resources'],AD_forward_predicts_already_inside_inherited156=16,
        line_search_plus_FD_predict=460,by_epsilon=by_eps,by_direction=by_direction,by_direction_and_epsilon=grid,
        comparisons=[{k:v for k,v in c.items() if k not in ('fd','saved_Jv','effective_float32_Jv','effective_direction')} for c in comparisons],
        epsilon_stability=recomputed['epsilon_stability'],new_neural_calls=0,new_physical_steps=0,new_AD_calls=0,
        derivative_verdict='not_assigned',scientific_pass_threshold=None,P1_admission=False,independent_confirmation=False,
        limitations=['three sampled directions are not full-J certification or all-semantic correctness',
            'context/history reconstructed from saved model terminals; no neural re-execution',
            'fixed 1e-12 numeric tolerances validate receipt arithmetic, not derivative acceptance',
            'no feasible-ball stationarity, independent-root, physical efficacy, or theoretical-failure verdict'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--report',type=Path,default=RUN/'report.json')
    p.add_argument('--report-sha',default=REPORT_SHA); a=p.parse_args()
    print(json.dumps(execute(a.report,a.report_sha),sort_keys=True,ensure_ascii=False,allow_nan=False))
