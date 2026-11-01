"""Saved Stage C metadata joins and fixed pre-result arithmetic. No native execution."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct

from experiments.spectral_reachability import agent_p3_capability_probe_result_protocol_20261029_v1 as math_contract

NAME = 'agent_p3_capability_probe_result_summary_20261029_v1'
PHYSICAL = 'agent_p3_same_time_capability_probe_physical_20261029_v2'
STAGEB = 'agent_p3_same_time_capability_probe_cuda_20261029_v1'
CAP_SHA = '88587a2d9f610df775c321eda6497bf3602c6ccd993b8779d7f36c6ad4744a4d'
PINS = dict(prepared='8fdecdc9fe5fcb1f00b03c30e9971566da03c2a523ccc652d86065c9b442e95f',
    batch='b70022a70a32c0c5f743684276e42f9040e228ae8296bb6daeae5fb70ce28c1b',
    receipt='4e7d35ec52dba109c9e2c181ad46d754f44bc0c0ef41a362c30a68776c66f3b1',
    capability=CAP_SHA,physical_source='f070b6afe4fe0ebba66e05d366f929467602b98805980f86f27e2344445a8344')
B_RECEIPT_SHA = '682a78366b10a60895c4530f2fc6cc401e6439b625f4ba0a8aa996dc2fed79a7'
B_PINS = dict(prepare='90b3a1a6284c5605594d4341a2f994870e4b8db9c1721ba40f0160244082e870',
    report='a64fbf31cf2601cf6ea26d100ef5cb5c01efbe8b63be6cf33bc4d91d735c25fc',
    completion='2d4f8364867ccfba4bb15f4794f0f665af34483cf4bdb1adf57af4efb1f27574',receipt=B_RECEIPT_SHA)
MATH_SHA = 'a9be0217934fa26b274c1967d1c8d3c5d582ccfddc2a357d8f937aa5332bf273'
FW_SHA = '2351ba5b05c91fb192ae4542e6839ab58b56bcd974170b94994246f61981f346'
U_SHAS = ('6029b37c7a02543a58ef962c4369002939239d3f34bda34bced155204eb3124a',
    '011d086c47cfd2ffd33fc16a895960d83e6cb0308c63168ecfd9bfe41abb8b21',
    '5cd3933ffab7ed42b0124f26414bcba8de3c15075f6fe0d27815e5a4b8c3caa3',
    'e1e2bf9124070256da21b064e2c1316045c9b3dc22190c665f44b9c9a3146422',
    'ebf2b284e92c80905c2da561fc7b69f669d4cfd69bc5cd6f213ef34627ab7c55')
PHYSICAL_FLAGS = tuple(k for k in math_contract.FLAGS if k != 'P2_admission')
COUNTS = dict(constructor=1,close=1,reset=2,set_state=4,goal_render=2,goal_bind=1,
    goal_setter=3,public_step=25,physics_step=254,explicit_render=81)
CAP = 256 * 1024**2
need = math_contract.need


def encoded(v, *, compact=True):
    options = dict(sort_keys=True, allow_nan=False)
    if compact:
        options['separators'] = (',', ':')
    return json.dumps(v, **options).encode()


def digest(v, *, compact=True):
    return hashlib.sha256(encoded(v, compact=compact)).hexdigest()


def same(a, b):
    return encoded(a) == encoded(b)


def absolute(value):
    p = Path(value)
    need(p.is_absolute() and str(p.resolve()) == str(p), 'canonical absolute path')
    return p


def sha(path):
    p = absolute(path)
    need(p.is_file() and not p.is_symlink(), 'regular non-symlink authority file')
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def ref(path):
    return dict(path=str(absolute(path)), sha256=sha(path))


def load(r, seen):
    path = str(absolute(r['path']))
    need(sha(path) == r['sha256'], 'complete external authority SHA: ' + path)
    need(Path(path).stat().st_size <= CAP, 'bounded saved metadata file')
    def pairs(items):
        out = {}
        for k, v in items:
            need(k not in out, 'duplicate JSON key')
            out[k] = v
        return out
    with open(path, 'rb') as f:
        value = json.load(f, object_pairs_hook=pairs,
            parse_constant=lambda x: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
    encoded(value)  # reject NaN/Infinity, including overflow exponents
    need(sha(path) == r['sha256'], 'stable saved metadata read')
    seen[path] = r['sha256']
    return value


def flags(value):
    need(all(value[k] is False for k in PHYSICAL_FLAGS), 'all original scientific flags explicitly false')


def raw_bits(rows, n):
    need(type(rows) is list and len(rows) == n, 'fixed literal raw action length')
    out = []
    for row in rows:
        need(type(row) is list and len(row) == 2, 'raw action pair shape')
        for x in row:
            math_contract.finite(x)
            need(-1 <= x <= 1 and struct.unpack('<f', struct.pack('<f', x))[0] == x,
                 'already float32 native domain, no clip/rechart')
        out.append(struct.pack('<2f', *row).hex())
    return out


def source_closure():
    root = Path(__file__).resolve().parents[2]
    paths = [Path(__file__).resolve(), root/'tests'/('test_'+NAME+'.py'), root/'innerdocs'/(NAME+'.md'),
        Path(math_contract.__file__).resolve(), root/'tests'/('test_'+math_contract.NAME+'.py'),
        root/'innerdocs'/(math_contract.NAME+'.md')]
    return {str(p): sha(str(p)) for p in paths}


def map_row(row, job, prior, reference):
    """Inspect actual saved returns, not a second native event/PNG fullstrict."""
    flags(row)
    need(same(row['job'], job) and row['fixture_only'] is False and row['new_neural_calls'] == 0,
         'entire original row/job identity')
    need(row['status'] in ('horizon_completed','native_stopped','engineering_failed'), 'native row status')
    ops = row['operations']
    need([x['index'] for x in ops] == list(range(len(ops))), 'complete original operation order')
    need(set(row['counts']) == set(COUNTS), 'all native resource axes')
    for k, cap in COUNTS.items():
        selected = [x for x in ops if x['resource'] == k]
        counts = dict(attempted=len(selected), completed=sum(x['status'] == 'returned' for x in selected))
        need(same(counts, row['counts'][k]) and counts['completed'] <= counts['attempted'] <= cap,
             'all saved attempted/completed resources, no refund')
    need(all(x['resource'] in COUNTS and x['status'] in ('pending','failed','returned') for x in ops),
         'known operation axis/status')
    need(row['pending_operation_indices']==[x['index'] for x in ops if x['status']=='pending'],
         'all unresolved callbacks remain pending')
    public = [x for x in ops if x['resource'] == 'public_step']
    expected_bits = raw_bits(job['warm10']+job['raw15'], 25)
    returned = []
    for i, op in enumerate(public):
        phase, step = ('warmup',i+1) if i<10 else ('plan',i-9)
        need((op['phase'],op['step']) == (phase,step) and op['raw_bits'] == expected_bits[i],
             'actual public raw bytes and chronological order')
        need(not returned or not (returned[-1]['result']['terminated'] or returned[-1]['result']['truncated']),
             'no public action after native early stop')
        if phase == 'plan' and step > 5:
            need(51 in row['guards'], 'cp5 causal guard inherited')
        if phase == 'plan' and step > 10:
            need(66 in row['guards'], 'cp10 causal guard inherited')
        if op['status'] == 'returned':
            v = op['result']
            need(v['actual_bits'] == op['raw_bits'] and v['actual_dtype'] == '<f4', 'native latest_action bytes')
            math_contract.vector7(v['state']); math_contract.vector7(v['goal_state'])
            need(type(v['terminated']) is bool and type(v['truncated']) is bool, 'original native flags')
            returned.append(op)
    plan = [x for x in returned if x['phase'] == 'plan']
    need(len(row['responses']) == len(plan), 'all original native responses, not terminal only')
    for response, op in zip(row['responses'], plan, strict=True):
        v = op['result']
        need(response['step'] == op['step'] and same(response['state'],v['state']) and
             response['reward'] == v['reward'] and response['native_success'] is v['terminated'] and
             response['terminated'] is v['terminated'] and response['truncated'] is v['truncated'] and
             response['native_metrics']['native_success'] is v['terminated'] and
             response['native_metrics']['native_distance'] == -v['reward'], 'every original native scalar/label join')
        need(same(v['goal_state'],prior['native_cp10']['goal7']), 'original task goal on every plan return')
    checkpoints = row['checkpoints']
    steps = [x['step'] for x in checkpoints]
    need(steps == sorted(set(steps)) and all(n in (0,5,10,15) for n in steps), 'fixed actual checkpoints')
    need(row['missing_checkpoints'] == [n for n in (0,5,10,15) if n not in steps], 'explicit checkpoint null denominator')
    for cp in checkpoints:
        need(cp['phase']=='plan' and cp['overlay_index']==0 and cp['condition']=='off', 'actual off checkpoint')
        observed = [x['receipt'] for x in row['observations'] if x['receipt']['phase']=='plan'
                    and x['receipt']['step']==cp['step'] and x['receipt']['overlay_index']==0]
        need(len(observed)==1 and same(observed[0],cp), 'checkpoint equals full original observation receipt')
        if cp['step'] in (0,5,10):
            old = next(x['receipt'] for x in reference['observations'] if x['receipt']['phase']=='plan'
                       and x['receipt']['step']==cp['step'] and x['receipt']['overlay_index']==0)
            need(all(same(cp[k],old[k]) for k in ('state','snapshot','file_sha256','pixel_sha256')),
                 'original actual prefix state/full snapshot/PNG identity')
        if cp['step']:
            op = next(x for x in plan if x['step']==cp['step'])
            need(same(cp['state'],op['result']['state']) and same(cp['snapshot'],op['result']['snapshot']),
                 'actual checkpoint tied to public return')
    endpoint = next((x for x in checkpoints if x['step']==15),None)
    need(same(endpoint,row['endpoint15']), 'actual endpoint15 is not imagined/substituted')
    if row['status']=='horizon_completed':
        need(steps==[0,5,10,15] and row['guards']==[51,66] and row['error'] is None and row['cleanup_errors']==[] and
             len(row['observations'])==81 and all(row['counts'][k]==dict(attempted=n,completed=n) for k,n in COUNTS.items()),
             'complete horizon original census')
    elif row['status']=='native_stopped':
        need(returned and (returned[-1]['result']['terminated'] or returned[-1]['result']['truncated']), 'real early stop')
    else:
        need(row['error'] is not None, 'engineering failure preserved')
    last = returned[-1] if returned else None
    v = None if last is None else last['result']
    ep = next((x['result'] for x in plan if x['step']==15),None) if endpoint is not None else None
    return dict(index=job['index'],endpoint=job['endpoint'],root_slot=job['root_slot'],role=job['source_role'],
        probe=job['probe'],status=row['status'],goal7=prior['native_cp10']['goal7'],
        cp15_state7=None if ep is None else ep['state'],cp15_native_reward=None if ep is None else ep['reward'],
        cp15_native_success=None if ep is None else ep['terminated'],
        terminal_control_step=0 if last is None or last['phase']=='warmup' else last['step'],
        terminal_state7=None if v is None else v['state'],terminal_goal7=None if v is None else v['goal_state'],
        terminal_native_success=None if v is None else v['terminated'],
        failure=None if row['error'] is None and row['cleanup_errors']==[] else dict(error=row['error'],cleanup_errors=row['cleanup_errors']))


def build(request, *, fixture=False):
    need(set(request)==set(PINS), 'explicit five authority inputs, no implicit future SHA')
    if not fixture:
        need(all(request[k]['sha256']==h for k,h in PINS.items()), 'MAIN actual fixed authorities')
    closure = source_closure(); seen = {}
    need(closure[str(Path(math_contract.__file__).resolve())]==MATH_SHA, 'unchanged frozen arithmetic kernel')
    p,b,s,c = [load(request[k],seen) for k in ('prepared','batch','receipt','capability')]
    src = request['physical_source']; need(sha(src['path'])==src['sha256'], 'physical source SHA')
    seen[src['path']] = src['sha256']
    need(Path(src['path']).name==PHYSICAL+'.py' and p['bindings'][src['path']]==src['sha256'], 'sealed producer source binding')
    flags(p); flags(b); flags(s)
    need(p['schema']==PHYSICAL+'.prepare' and b['schema']==PHYSICAL+'.batch' and s['schema']==PHYSICAL+'.consume', 'own actual physical schemas')
    need(p['fixture_only'] is fixture and p['known_roots']==2 and p['fresh_roots']==0 and p['stageB_connected'] is True,
         'known2 actual/fixture boundary')
    need(s['prepared']==b['prepared']==request['prepared']['path'] and
         s['prepare_sha256']==b['prepare_sha256']==request['prepared']['sha256'], 'prepare path and full byte identity')
    need(s['fixed_denominator']==10 and len(s['jobs'])==len(b['jobs'])==len(p['jobs'])==10, 'all ten originals')
    need(same(s['source_authorities'],p['authority']) and
         same(s['inherited_stageB_resource_receipt'],p['inherited_stageB_resource_receipt']) and
         same(c['inherited_stageB_resource_receipt'],p['inherited_stageB_resource_receipt']), 'whole upstream authority/resource inheritance')
    need(s['new_neural_calls']==0 and s['FLOPs'] is None, 'no new neural or invented FLOPs')
    authority = p['authority']
    need(same(c['authorities']['stageB_receipt'],authority['stageB']['receipt']), 'capability and physical same stageB receipt')
    if not fixture:
        need({k:v['sha256'] for k,v in authority['stageB'].items()}==B_PINS and authority['full_witness']['sha256']==FW_SHA and
             [x['sha256'] for x in authority['U']]==list(U_SHAS), 'original B/full witness/export authorities')
    for a in authority['stageB'].values():
        need(sha(a['path'])==a['sha256'], 'inherited stageB full file byte seal'); seen[a['path']]=a['sha256']
    w = load(authority['full_witness'],seen)
    need(w['schema']=='agent_p3_actual_cp10_full_witness_20261029_v1.witness' and len(w['rows'])==5,
         'whole five actual history witness')
    rows = []; original = []; seals = []; exports = []
    for e in range(5):
        export = load(authority['U'][e],seen); exports.append(export)
        plan = export['plan']; probe = c['probes'][e]; old = w['rows'][e]
        need(export['schema']==STAGEB+'.witness' and export['receipt']==authority['stageB']['receipt']['path'] and
             export['receipt_sha256']==authority['stageB']['receipt']['sha256'] and
             export['evidence_kind']=='MAIN_SHA_bound_saved_receipt', 'actual paid exported U witness')
        need(digest({k:v for k,v in plan.items() if k!='witness_sha256'})==plan['witness_sha256']==probe['witness_sha256'],
             'compact producer plan hash, distinct from physical spaced codec')
        ubytes=bytes.fromhex(''.join(raw_bits(plan['raw5'],5)))
        need(ubytes.hex()==plan['raw_float32_bits'] and hashlib.sha256(ubytes).hexdigest()==plan['raw_float32_sha256'],
             'saved U bytes/full bitstring independent literal reconstruction')
        need((plan['index'],plan['root_slot'],plan['role'])==(e,*math_contract.ENDPOINTS[e]) and
             plan['native_call_ids']==dict(signal=e+2,whole=7+2*e,segmented=8+2*e), 'original native dual replay call identities')
        ba=authority['stageB']
        need(plan['authority']==dict(prepared=ba['prepare']['path'],prepare_sha256=ba['prepare']['sha256'],
             report=ba['report']['path'],report_sha256=ba['report']['sha256'],
             completion=ba['completion']['path'],completion_sha256=ba['completion']['sha256']), 'whole U downstream paid authority')
        need(all(same(plan[k],probe[k]) for k in ('raw5','raw_float32_sha256','endpoint_identity','source')) and
             plan['endpoint_identity']==old['stageA_task']['identity'] and plan['task']==old['stageA_task']['task'] and
             plan['source']==old['stageA_row']['source'] and plan['causal_raw10']==old['control_raw10'] and
             plan['checkpoint']==10 and plan['probe_checkpoint']==15 and plan['whole']==plan['segmented'],
             'same original actual endpoint/task/dual replay, no reoptimization')
        need(probe['native_cp10']['goal7']==old['source_row']['goal'], 'original native goal')
    for i,(endpoint,root,role,probe) in enumerate(math_contract.JOBS):
        job = p['jobs'][i]; prior=c['probes'][endpoint]; old=w['rows'][endpoint]; plan=exports[endpoint]['plan']
        need((job['index'],job['endpoint'],job['root_slot'],job['source_role'],job['probe'])==(i,endpoint,root,role,probe),
             'fixed ten branch identity/order')
        need(job['selection_digest']==digest({k:v for k,v in job.items() if k!='selection_digest'},compact=False), 'physical spaced selection codec')
        need(job['source_witness_sha256']==authority['full_witness']['sha256'] and same(job['U_witness'],authority['U'][endpoint]), 'source witness pointer')
        raw = raw_bits(job['raw15'],15); raw_bits(job['warm10'],10)
        need(raw_bits(job['warm10'],10)==raw_bits(old['warm_raw10'],10) and raw[:10]==raw_bits(old['control_raw10'],10), 'whole original warm10 and causal raw10')
        need(hashlib.sha256(bytes.fromhex(''.join(raw[:10]))).hexdigest()==prior['endpoint_identity']['raw10_float32_sha256'], 'history byte hash')
        need(raw[10:]==(raw_bits(plan['raw5'],5) if probe=='U' else ['00'*8]*5), 'U exact float32 or positive-zero Z2 tail')
        receipt=s['jobs'][i]; seal=b['jobs'][i]; flags(receipt)
        out=str(absolute(seal['output']))
        need(out==str(Path(request['batch']['path']).parent/f'job{i:02}') and receipt['output']==out, 'all ten fresh non-reused directories')
        binding=load(ref(str(Path(out)/'binding.json')),seen)
        need(binding['schema']==PHYSICAL+'.job_binding' and binding['prepared']==request['prepared']['path'] and
             binding['prepare_sha256']==request['prepared']['sha256'] and binding['fixture_only'] is False and
             same(binding['job'],job) and same(binding['authority'],authority), 'full native job binding')
        need(binding['guard_transport_fields']==['index','previous_sha256','event_sha256','branch','arm','selection_digest','receipt/png'],
             'exact enumerated upstream transport projection')
        if receipt['schema']==PHYSICAL+'.failure_consume':
            need(receipt['status']==seal['status']=='transport_failed_NOT_verified' and receipt['counts'] is receipt['endpoint15'] is None and
                 receipt['unknown_final_callbacks'] is True, 'unverified transport unknowns never zero')
            f=load(dict(path=str(Path(out)/'failure.json'),sha256=seal['failure_sha256']),seen)
            need(receipt['failure_sha256']==seal['failure_sha256'] and same(f,receipt['declarations']) and
                 f['schema']==PHYSICAL+'.publication_failure' and f['status']==receipt['status'] and
                 f['unknown_final_callbacks'] is True and f['endpoint15'] is None and
                 f['binding_sha256']==digest(binding,compact=False), 'complete transport declarations and identity')
            record=dict(index=i,endpoint=endpoint,root_slot=root,role=role,probe=probe,status='transport_failed',
                goal7=prior['native_cp10']['goal7'],cp15_state7=None,cp15_native_reward=None,cp15_native_success=None,
                terminal_control_step=0,terminal_state7=None,terminal_goal7=None,terminal_native_success=None,failure=f)
            original.append(None)
        else:
            need(receipt['schema']==PHYSICAL+'.job_consume' and receipt['fixture_only'] is False, 'whole saved job receipt')
            for k in ('report_sha256','completion_sha256'):
                need(receipt[k]==seal[k], 'actual job report/completion seal')
            r=load(dict(path=str(Path(out)/'report.json'),sha256=seal['report_sha256']),seen)
            completion=load(dict(path=str(Path(out)/'completion.json'),sha256=seal['completion_sha256']),seen)
            need(set(r)=={'schema','binding','row','journal_tail','journal_sha256','elapsed_seconds'} and
                 set(completion)=={'schema','report_sha256','artifact_sha256','bytes_before_completion','inclusive_wall_seconds','wall_scope'},
                 'actual full report/completion metadata ABI')
            need(0<=r['elapsed_seconds']<=completion['inclusive_wall_seconds']<=900 and
                 completion['wall_scope']=='through report hash; completion self-write excluded', 'inherited final publication wall scope')
            need(r['schema']==PHYSICAL+'.report' and completion['schema']==PHYSICAL+'.completion' and
                 completion['report_sha256']==seal['report_sha256'] and same(r['binding'],binding) and
                 same(r['row'],receipt['row']) and completion['artifact_sha256']['report.json']==seal['report_sha256'] and
                 completion['artifact_sha256']['binding.json']==seen[str(Path(out)/'binding.json')], 'entire original row/report/completion metadata')
            record=map_row(receipt['row'],job,prior,old); original.append(receipt['row'])
        rows.append(record); seals.append(seal)
    partial=any(x['status']=='transport_failed' for x in rows)
    need(s['status']==('saved_partial_transport_NOT_verified' if partial else 'all10_saved_native_jobs_reconstructed_NOT_admitted'),
         'top receipt status does not hide partial failures')
    arithmetic=math_contract.evaluate(rows,c)
    result=dict(schema=NAME+'.summary',status='fixture_only_NOT_evidence' if fixture else
        'saved_partial_transport_descriptive_NOT_verified' if partial else 'saved_stageC_descriptive_NOT_admitted',
        request=request,sources=closure,input_sha256=seen,fixture_only=fixture,
        inherited_fullstrict_only=True,upstream_fullstrict_reexecuted=False,
        original_receipts=s['jobs'],original_rows=original,batch_seals=seals,mapped_rows=rows,
        inherited_stageB_resource_receipt=s['inherited_stageB_resource_receipt'],
        native_resource_vectors=[None if r is None else r['counts'] for r in original],
        arithmetic=arithmetic,known_roots=2,fresh_roots=0,new_neural_calls=0,new_physics_calls=0,new_AD_calls=0,
        FLOPs=None,**math_contract.FLAGS)
    need(source_closure()==closure and all(sha(f)==h for f,h in seen.items()), 'stable source and metadata after build')
    return result


def write_new(path, value):
    p=absolute(path); data=encoded(value)+b'\n'
    need(len(data)<=CAP and p.parent.is_dir(), 'bounded output and existing parent')
    with p.open('xb') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    fd=os.open(str(p.parent),os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return ref(str(p))


def verify(path, expected, *, fixture=False):
    value=load(dict(path=path,sha256=expected),{})
    need(value['fixture_only'] is fixture and same(value,build(value['request'],fixture=fixture)), 'whole independent saved rebuild')
    need(sha(path)==expected, 'stable summary after verify')
    return dict(schema=NAME+'.verify',summary=path,summary_sha256=expected,fixture_only=fixture,
        status='fixture_only_NOT_evidence' if fixture else 'saved_analysis_rebuilt_NOT_admitted',**math_contract.FLAGS)


def main(argv=None):
    q=argparse.ArgumentParser(description=__doc__); sub=q.add_subparsers(dest='command',required=True)
    a=sub.add_parser('build')
    for key in PINS:
        a.add_argument('--'+key.replace('_','-'),required=True)
        a.add_argument('--'+key.replace('_','-')+'-sha',required=True)
    a.add_argument('--output',required=True)
    a=sub.add_parser('verify');a.add_argument('--summary',required=True);a.add_argument('--summary-sha',required=True)
    a=q.parse_args(argv)
    if a.command=='build':
        result=write_new(a.output,build({k:dict(path=getattr(a,k),sha256=getattr(a,k+'_sha')) for k in PINS}))
    else:
        result=verify(a.summary,a.summary_sha)
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':
    main()
