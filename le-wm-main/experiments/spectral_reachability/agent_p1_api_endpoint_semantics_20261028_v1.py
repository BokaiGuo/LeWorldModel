"""Saved-only correction of native full3 versus shifted-context comparison semantics."""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
from pathlib import Path
import numpy as np
from experiments.spectral_reachability import agent_p1_teacher_library40_api_equivalence_20261028_v1 as api

NAME = 'agent_p1_api_endpoint_semantics_20261028_v1'
ROOT, rgb, teacher = api.ROOT, api.rgb, api.teacher
require, sha, digest, canonical = api.require, api.sha, api.digest, api.canonical
REPORT = ROOT/'results/agent_p1_teacher_library40_api_equivalence_20261028_v1_cuda_MAIN01/report.json'
REPORT_SHA = '38ad7134ba7f13b7244a9bc9ad311d476ccd67a0df53daf9fa7f314e7876ae3e'
RECEIPT = ROOT/'results/agent_p1_teacher_library40_api_equivalence_20261028_v1_consume_MAIN01.json'
RECEIPT_SHA = 'fc6f7022efc1df77b3eb85ff8262a87d0cbb5cf6bd1f65d6d77f512ae5651d68'


def native3(a):
    require(isinstance(a,np.ndarray) and a.dtype == np.float32 and a.shape == (3,192) and np.isfinite(a).all(),
            'exact saved native float32 full3, never squeeze/broadcast arbitrary shape')
    return a


def discrepancy(a,b):
    require(a.dtype == b.dtype and a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), 'same semantic shape and dtype')
    return dict(api.stageb.discrepancy(a,b), bitwise_equal=a.tobytes() == b.tobytes())


def resolve(source,native,old_context,old_native,goal,actual,legacy):
    """Do not reject route/model differences. Only reject broken saved plumbing."""
    source,native,old_context,old_native = [native3(x) for x in (source,native,old_context,old_native)]
    require(isinstance(goal,np.ndarray) and goal.dtype == np.float32 and goal.shape == (192,), 'original saved native192 goal')
    # Old context consists of two preserved observations plus one native predicted endpoint.
    api.same(old_context[:2],source[1:],'old shifted context retained actual z10/z15')
    api.same(old_context[-1],old_native[-1],'old shifted endpoint equals original native endpoint')
    shifted = np.concatenate((source[1:],native[-1:]),axis=0)
    legacy_recomputed = api.stageb.discrepancy(native,old_context)
    require(legacy['vs_original_prediction'] == legacy_recomputed, 'original erroneous field retained exactly, not silently rewritten')
    terms = api.history.rank.scalar_terms(native[-1],actual,goal)
    require(all(legacy[k] == v for k,v in terms.items()), 'original endpoint-based scalar Q/decomposition independently recomputed')
    old_terms = api.history.rank.scalar_terms(old_context[-1],actual,goal)
    return dict(native_output_semantics=dict(shape=[3,192],input_context_absolute_steps=[5,10,15],
                    rows=[dict(native_row=i,kind='model_prediction_not_preserved_observation',nominal_next_absolute_step=step) for i,step in enumerate((10,15,20))]),
        shifted_context_semantics=[dict(row=0,kind='preserved_observation',absolute_step=10),dict(row=1,kind='preserved_observation',absolute_step=15),dict(row=2,kind='model_prediction',absolute_step=20)],
        endpoint_vs_original_endpoint=discrepancy(native[-1],old_context[-1]),
        native_full3_vs_original_native_full3=discrepancy(native,old_native),
        reconstructed_shifted_context_vs_original_shifted_context=discrepancy(shifted,old_context),
        native_rows_vs_original_native_rows=[discrepancy(native[i],old_native[i]) for i in range(3)],
        preserved_history_vs_original_shifted_rows=[discrepancy(source[i+1],old_context[i]) for i in range(2)],
        endpoint_Q=dict(current=terms,original=old_terms,predicted_Q_delta=terms['predicted_Q']-old_terms['predicted_Q']),
        deprecated_v1_field=dict(field='vs_original_prediction',original_value=deepcopy(legacy['vs_original_prediction']),
            independently_recomputed=legacy_recomputed,status='invalid_cross_semantics_for_drift',
            causal_drift_evidence=False,reason='native rows0/1 are model outputs; old rows0/1 preserve observed input z10/z15; only last rows share endpoint semantics'))


def assemble(rows):
    require(len(rows) == 40,'fixed40, no winner/filter/drop')
    for i,r in enumerate(rows):
        task,rem = divmod(i,10); role,word = divmod(rem,5)
        require((r['index'],r['root_slot'],r['condition'],r['role'],r['library_index'],r['library_label']) ==
            (i,task//2,('off','on')[task%2],('Z','S')[role],word,teacher.LIBRARY[word]) and
            set(r['routes']) == set(api.ROUTES),'original fixed roster and both independently executed routes')
        for route in api.ROUTES:
            require((r['routes'][route]['endpoint_Q']['current']['actual_Q'] is None) == (i == 12), 'missing12 null, all other actuals preserved')
    return dict(rows=rows,fixed_slots=40,route_slots=80,known_roots=2,independent_roots=0,missing_actual=[12],
        per_route={route:dict(endpoint_exact_slots=sum(r['routes'][route]['endpoint_vs_original_endpoint']['exact_equal'] for r in rows),
            endpoint_bitwise_equal_slots=sum(r['routes'][route]['endpoint_vs_original_endpoint']['bitwise_equal'] for r in rows),
            native_full3_exact_slots=sum(r['routes'][route]['native_full3_vs_original_native_full3']['exact_equal'] for r in rows),
            reconstructed_shifted_exact_slots=sum(r['routes'][route]['reconstructed_shifted_context_vs_original_shifted_context']['exact_equal'] for r in rows),
            deprecated_cross_semantics_unequal_slots=sum(not r['routes'][route]['deprecated_v1_field']['original_value']['exact_equal'] for r in rows),
            max_endpoint_abs_difference=max(r['routes'][route]['endpoint_vs_original_endpoint']['max_abs_difference'] for r in rows),
            max_abs_predicted_Q_delta=max(abs(r['routes'][route]['endpoint_Q']['predicted_Q_delta']) for r in rows)) for route in api.ROUTES})


def build():
    r = rgb.read(REPORT,REPORT_SHA); receipt = rgb.read(RECEIPT,RECEIPT_SHA)
    require(receipt == api.consume(REPORT,REPORT_SHA), 'entire original strict consumer/arrays/ledger/counters rebuilt, including deprecated field')
    binding = rgb.read(REPORT.parent/'execution_binding.json'); p = rgb.read(binding['prepared'],binding['prepare_sha256'])
    require(binding['prepare_sha256'] == r['prepare_sha256'] == receipt['prepare_sha256'], 'same actual prepare/report/receipt')
    api.validate_jobs(p['jobs'])
    bindings = {**p['bindings'],**p['sources_sha256'],**p['model_bindings'],str(REPORT):REPORT_SHA,str(RECEIPT):RECEIPT_SHA,
                str(Path(binding['prepared'])):binding['prepare_sha256']}
    for name,h in r['artifact_sha256'].items(): bindings[str(REPORT.parent/name)] = h
    arrays = {}
    def load(folder,a,old=False):
        value = teacher.load_array(folder,a) if old else api.mid.load_array(folder,a)
        path = teacher.flat(folder,a['file'])
        enriched = dict(original_receipt=deepcopy(a),array_sha256=hashlib.sha256(value.tobytes()).hexdigest(),dtype=str(value.dtype),shape=list(value.shape))
        require(str(path) not in arrays or arrays[str(path)] == enriched, 'unique full-array receipt')
        expected = a['file_sha256'] if old else a['sha256']
        require(sha(path) == expected and (str(path) not in bindings or bindings[str(path)] == expected), 'same pinned file receipt, no closure overwrite')
        arrays[str(path)] = enriched; bindings[str(path)] = expected; return value
    cpath = ROOT/rgb.INPUTS['compiler'][0]; compiled = rgb.read(cpath,rgb.INPUTS['compiler'][1]); base = cpath.parent
    rows = []
    for j,old_summary in zip(p['jobs'],receipt['rows'],strict=True):
        i,word = j['index'],j['library_index']; cr = compiled['rows'][i//5]
        old_context = load(j['compiler_folder'],j['old_prediction_receipt'],True)
        old_forward = cr['forwards'][2+word]
        require(old_forward['stage'] == 'evaluate_'+str(word), 'original same-word forward, not nominal/other library')
        old_native = load(base,old_forward['modules'][-1]['output'],True)
        original_endpoint = teacher.check_forward(old_forward,cr['blind_input'],lambda a:load(base,a,True))
        api.same(original_endpoint,old_native[-1],'original full3 module endpoint binding')
        goal = load(base,cr['arrays']['goal'],True)
        api.same(goal,np.asarray(j['goal192'],np.float32),'original saved goal array, no retargeting')
        old_Q = float(np.mean((old_context[-1].astype(np.float64)-goal.astype(np.float64))**2))
        require(old_Q == j['original_predicted_Q'], 'original mean Q, no public criterion/sum substitution')
        actual = None if j['actual_receipt'] is None else load(api.history.rank.REPORT.parent,j['actual_receipt'],True)
        if actual is not None: api.same(actual,np.asarray(j['actual192'],np.float32),'same original actual terminal latent')
        item = {k:deepcopy(j[k]) for k in ('index','root_slot','condition','role','library_index','library_label','task_id','anchor_id','plan_digest','goal_binding','missing_reason')}
        require(all(old_summary[k] == item[k] for k in ('index','root_slot','condition','role','library_label')), 'exact current/legacy slot identity')
        item.update(routes={},original_shifted_context_receipt=deepcopy(j['old_prediction_receipt']),original_native_full3_receipt=deepcopy(old_forward['modules'][-1]['output']))
        for route in api.ROUTES:
            rid = j['job_id']+'.'+route
            new_row = r['rows'][2*i+api.ROUTES.index(route)]
            require(new_row['run_id'] == rid and new_row['job_binding_sha256'] == j['job_binding_sha256'], 'actual literal route/slot binding')
            source = load(REPORT.parent,new_row['arrays']['full3'])
            require(source.shape == (1,3,192), 'original full3 source ABI')
            es = [e for e in r['entries'] if e['run_id'] == rid and e['kind'] == 'predict']
            require(len(es) == 1, 'exactly one actual native predict per route')
            a = es[0]['arrays']['output_output']; native = load(REPORT.parent,a)
            require(native.shape == (1,3,192), 'exact native predict output ABI')
            source_expected = load(base,cr['arrays']['latents'],True)
            api.same(source[0],source_expected,'literal original teacher input source')
            if route == 'public_rollout':
                public = load(REPORT.parent,new_row['arrays']['public_predicted_emb'])
                require(public.shape == (1,1,4,192),'exact public three histories plus endpoint')
                api.same(public[0,0,:3],source[0],'public preserved unshifted source history')
                api.same(public[0,0,-1],native[0,-1],'public same native endpoint')
            else: api.same(load(REPORT.parent,new_row['arrays']['cached_predictions']),native,'cached native return')
            item['routes'][route] = dict(resolve(source[0],native[0],old_context,old_native,goal,actual,old_summary['routes'][route]),
                new_native_prediction_receipt=deepcopy(a),source_history_receipt=deepcopy(new_row['arrays']['full3']))
        rows.append(item)
    own = [Path(__file__).resolve(), ROOT/f'tests/test_{NAME}.py', ROOT/f'innerdocs/{NAME}.md']
    for path in own: bindings[str(path)] = sha(path)
    rgb.verify_bindings(bindings)
    return dict(schema=NAME+'.analysis',status='saved_endpoint_semantics_corrected_not_admitted',bindings=bindings,
        source_files={str(f):bindings[str(f)] for f in own},array_receipts=arrays,input_report_sha256=REPORT_SHA,input_receipt_sha256=RECEIPT_SHA,
        original_all40_routes_exact_equal=receipt['all40_routes_exact_equal'],original_route_rank_errors=deepcopy(receipt['route_rank_errors']),
        causal_drift_claim=False,new_neural_calls=0,new_physics_steps=0,**assemble(rows),**api.FLAGS)


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('command',choices=('analyze','verify'))
    ap.add_argument('--output'); ap.add_argument('--analysis'); ap.add_argument('--analysis-sha'); args = ap.parse_args()
    a = build()
    if args.command == 'verify':
        require(rgb.read(args.analysis,args.analysis_sha,16*1024**2) == a,'whole deterministic semantic report rebuild')
        result = dict(verified=True,analysis_sha256=args.analysis_sha)
    elif args.output:
        path = Path(args.output).absolute()
        require(path.resolve() == path and path.parent == ROOT/'results' and path.name.startswith(NAME+'_') and path.suffix == '.json','exclusive own new report only')
        api.mid.write(path,a); result = dict(output=str(path),analysis_sha256=sha(path),per_route=a['per_route'])
    else: result = dict(status=a['status'],per_route=a['per_route'],fixed_slots=40,route_slots=80)
    print(canonical(result).decode())


if __name__ == '__main__': main()
