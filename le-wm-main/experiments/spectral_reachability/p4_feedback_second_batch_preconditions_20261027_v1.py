"""Saved-data preconditions for a genuinely root-disjoint second P4 batch.

No runtime/compiler import, neural execution, model loading or physics.
Audit the existing source closure; do not invent a runnable protocol on failure.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path

from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal, verify_seal

TAG='p4_feedback_second_batch_preconditions_20261027_v1'
REPO=Path(__file__).resolve().parents[2]
RUNTIME='p4_multiround_feedback_runtime_20261027_v1'
RESPONSE='pusht_geometry_feedback_response_20261027_v1'
CAPABILITY='pusht_geometry_capability_20261027_v1'
PLANNER='pusht_geometry_prefix_20261027_v3'
SMALL=16*1024*1024
LARGE=128*1024*1024


def require(ok,message):
    if not ok: raise ValueError(message)


def canonical(value):
    return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()


def decode(raw):
    def pairs(items):
        out={}
        for k,v in items:
            require(k not in out,'duplicate JSON key: '+k); out[k]=v
        return out
    value=json.loads(raw,object_pairs_hook=pairs); canonical(value); return value


class Inputs:
    def __init__(self):
        self.hashes={}; self.read_bytes=0

    def hash(self,path):
        path=Path(path).resolve()
        require(path.is_file() and path.stat().st_size<=1024**3,'bounded regular input required: '+str(path))
        h=hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''): h.update(chunk)
        value=h.hexdigest()
        require(str(path) not in self.hashes or self.hashes[str(path)]==value,'input changed: '+str(path))
        self.hashes[str(path)]=value; return value

    def raw(self,path,limit=SMALL):
        path=Path(path).resolve()
        require(path.stat().st_size<=limit,'oversize JSON/source: '+str(path))
        require(self.read_bytes+path.stat().st_size<=256*1024*1024,'aggregate parse budget')
        with path.open('rb') as stream: raw=stream.read(limit+1)
        require(len(raw)<=limit,'input grew')
        self.read_bytes+=len(raw)
        expected=self.hash(path)
        require(hashlib.sha256(raw).hexdigest()==expected,'read/hash race')
        return raw

    def json(self,path,limit=SMALL): return decode(self.raw(path,limit))

    def stable(self):
        for path,expected in list(self.hashes.items()): require(self.hash(path)==expected,'input changed')


def roots_of(value):
    """Metadata identity fields only; never interpret arbitrary 64-byte hashes as roots."""
    result=set()
    def walk(v):
        if isinstance(v,dict):
            for key,item in v.items():
                if key in ('root_sha256','root_id') and isinstance(item,str) and len(item)==64: result.add(item)
                if key in ('roots','qualified_roots','original_root_order') and isinstance(item,list):
                    result.update(s for s in item if isinstance(s,str) and len(s)==64)
                walk(item)
        elif isinstance(v,list):
            for item in v: walk(item)
    walk(value); return sorted(result)


def qualification(plans,root_order):
    require(len(root_order)==4 and len(set(root_order))==4 and len(plans)==4,'complete four-root planner census')
    require({p['root_sha256'] for p in plans}==set(root_order),'planner/root identity')
    result=[]
    for i,root in enumerate(root_order):
        p=next(v for v in plans if v['root_sha256']==root)
        require(type(p['root_index']) is int and p['root_index']==i,'planner root index')
        geometry=[a for a in p['arms'] if a['arm']=='geometry']
        require(len(geometry)==1,'unique geometry arm')
        arm=geometry[0]; candidates=arm['candidates']
        ids=[v['candidate_id'] for v in candidates]
        require(len(ids)==len(set(ids)) and arm['selected_candidate_id'] in ids and arm['greedy_candidate_id'] in ids,'selected/greedy candidate identity')
        different=arm['selected_candidate_id']!=arm['greedy_candidate_id']
        result.append(dict(root_index=i,root_sha256=root,selected_candidate_id=arm['selected_candidate_id'],
            greedy_candidate_id=arm['greedy_candidate_id'],old_model_only_qualification=different,
            qualification_semantics='saved selected!=greedy under old geometry contract; not P1 or physical success'))
    return result


def consumption_contract(runtime_text,feedback_text):
    """Locate and verify ACTUAL source statements through AST, without importing them."""
    tree=ast.parse(runtime_text); ftree=ast.parse(feedback_text)
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='arm_loop')
    statements=[n for n in ast.walk(fn) if isinstance(n,ast.Assign)]
    expressions={ast.unparse(n):n for n in statements}
    needed=[
        "observed = backend.encode(frames, between, f'{arm}_{boundary}')",
        "chosen_context = observed if arm == 'actual' else predicted",
        "plan, successor = backend.compile(deepcopy(chosen_context), deepcopy(p), arm, boundary)",
        "predicted = successor",
    ]
    for expr in needed: require(expr in expressions,'feedback consumer AST changed: '+expr)
    advance=[n for n in ast.walk(fn) if isinstance(n,ast.Call) and ast.unparse(n.func)=='feedback.advance_observed']
    require(len(advance)==2 and all(len(n.args)==5 and ast.unparse(n.args[2])=='terminal_rgb' for n in advance), 'actual RGB advance contract')
    helper=next(n for n in ftree.body if isinstance(n,ast.FunctionDef) and n.name=='advance_observed')
    helper_src=ast.unparse(helper)
    require("execution['status'] != 'completed'" in helper_src and "execution['completed_commands'] != 5" in helper_src and
        "execution['remaining_commands_not_executed'] != 0" in helper_src,'partial packet must never advance history')
    require('frames[1], frames[2], terminal_rgb' in helper_src and 'between[1], executed_packet' in helper_src,'exact own history rolling shift')
    result=[dict(role='actual_observation_encoding',line=expressions[needed[0]].lineno,
        legal_at_primitive_steps=[5,10,15],source_expression=needed[0],
        rule='own last three actual RGB boundaries and two completed raw packets, before selection'),
        dict(role='selection_feedback_consumer',line=expressions[needed[1]].lineno,
        legal_at_primitive_steps=[5,10,15],source_expression=needed[1],
        rule='actual uses observed; imagined uses predicted, never substitute other-arm histories'),
        dict(role='compiler_input',line=expressions[needed[2]].lineno,
        legal_at_primitive_steps=[5,10,15],source_expression=needed[2],
        rule='compile exact chosen context; seal selection before own next five commands'),
        dict(role='post_dispatch_history_update',line=advance[-1].lineno,
        legal_at_primitive_steps=[10,15,20],rule='only complete guarded packet; own terminal_rgb and executed_packet'),
        dict(role='terminal20_derived_latent',legal_at_primitive_steps=[20],
        rule='evaluation only after all three decisions; cannot retroactively condition step15 selection or make second batch')]
    return result


def exclude_roots(qualified,first_root,cap_roots,response_root,runtime_root,used_roots):
    result=[]
    for q in qualified:
        root=q['root_sha256']; reasons=[]
        if root==first_root: reasons.append('same_root_as_first_batch')
        if root in used_roots: reasons.append('previously_used_calibration_root_not_fresh_holdout')
        if not q['old_model_only_qualification']: reasons.append('old_geometry_selected_equals_greedy')
        if root not in cap_roots: reasons.append('no_root_specific_capability_pack_in_existing_chain')
        if root!=response_root: reasons.append('no_root_specific_feedback_response_pack_in_existing_chain')
        if root!=runtime_root: reasons.append('existing_runtime_hard_bound_to_other_root')
        result.append(dict(q,root_disjoint_from_first=root!=first_root,fresh_holdout_verified=False,
            source_chain_complete=root in cap_roots and root==response_root==runtime_root,
            engineering_prepare_eligible=root!=first_root and root in cap_roots and root==response_root==runtime_root,
            exclusion_reasons=reasons))
    return result


def inventory(results,inputs):
    paths=set()
    for pattern in ('p4*/protocol.json','p4*/report.json','p4*/receipt.json',
                    'pusht_geometry*/protocol.json'):
        paths.update(results.glob(pattern))
    rows=[]
    for path in sorted(paths):
        if path.parent.name.startswith(TAG): continue
        if path.stat().st_size>SMALL:
            rows.append(dict(path=str(path),sha256=inputs.hash(path),status='oversize_not_parsed',roots_unconfirmed=True)); continue
        obj=inputs.json(path); roots=roots_of(obj)
        c=obj.get('contract',{})
        runtime=c.get('planned_steps_per_arm')==20 and c.get('boundaries')==[5,10,15]
        if runtime: category='20step3round_protocol_not_an_extra_execution'
        elif obj.get('schema')==RUNTIME: category='first_batch_runtime_report'
        elif 'derived_receipt' in obj.get('schema',''): category='derived_encoding_not_new_batch'
        elif 'audit' in obj.get('schema','') or 'attribution' in obj.get('schema',''): category='saved_audit_not_new_batch'
        else: category='different_protocol_or_upstream_diagnostic_not_second_full20_batch'
        rows.append(dict(path=str(path),sha256=inputs.hashes[str(path.resolve())],schema=obj.get('schema'),
            status=obj.get('status'),root_sha256s=roots,category=category))
    return rows


def audit(project=REPO):
    project=Path(project).resolve(); results=project/'results'; inputs=Inputs()
    runtime_dir=results/(RUNTIME+'_prepare_v1')
    runtime_p=inputs.json(runtime_dir/'protocol.json'); runtime_r=inputs.json(runtime_dir/'report.json')
    require(runtime_r['status']=='calibration_recorded' and not runtime_r['failures'] and
        runtime_r['protocol_sha256']==inputs.hash(runtime_dir/'protocol.json'),'verified first runtime binding')
    root=runtime_p['root_sha256']; used=runtime_p['original_root_order']
    require(runtime_p['contract']['root_sha256']==root and roots_of(runtime_r)==[root] and
        len(runtime_r['rows'])==2 and {r['arm'] for r in runtime_r['rows']}=={'imagined','actual'},'first batch one-root two-arm census')
    response_dir=Path(runtime_p['source']).resolve()
    response=inputs.json(response_dir/'protocol.json')
    require(runtime_p['bindings'][str(response_dir/'protocol.json')]==inputs.hash(response_dir/'protocol.json'),'runtime/response SHA edge')
    cap_path=Path(response['capability']).resolve(); cap=inputs.json(cap_path); cap_p=inputs.json(cap_path.parent/'protocol.json')
    require(response['bindings'][str(cap_path)]==inputs.hash(cap_path) and
        cap['protocol_sha256']==inputs.hash(cap_path.parent/'protocol.json'),'response/capability SHA edges')
    planner_path=Path(cap_p['planning_report']).resolve()
    require(planner_path.name=='report.json' and planner_path.parent.name==PLANNER+'_cuda_v1','exact supported large planner identity')
    require(cap_p['bindings'][str(planner_path)]==inputs.hash(planner_path),'capability/planner SHA edge')
    # Sole large JSON exception: exact upstream path AND SHA already bound by capability.
    planner=inputs.json(planner_path,LARGE)
    require(planner['schema']==PLANNER and planner['status']=='plans_sealed' and planner['physical_outcomes_used'] is False and
        planner['protocol']['roots']==used and planner['gates'] and all(v is True for v in planner['gates'].values()),'original planner contract')
    qualified=qualification(planner['plans'],used)
    cap_roots=cap['census']['qualified_roots']
    require(cap_roots==[q['root_sha256'] for q in qualified if q['old_model_only_qualification']], 'complete qualification/exclusion root census')
    require(set(roots_of(cap_p['plans']))==set(cap_roots) and set(roots_of(cap['rows']))==set(cap_roots) and
        cap['census']['complete'] is True and len(cap_p['plans'])==cap['census']['branches']==len(cap['rows'])==123*len(cap_roots), 'capability full 3role41probe closure')
    require(root==response['root_sha256'] and [c['role'] for c in response['cases']]==['selected','greedy','zero'],'response role/root closure')
    sources={}
    for name in (RUNTIME,RESPONSE,CAPABILITY,'p4_feedback_matched_20261027_v2'):
        path=project/f'experiments/spectral_reachability/{name}.py'; sources[name]=inputs.raw(path).decode()
    tree=ast.parse(sources[RESPONSE]); literal=next(n.value.value for n in tree.body if isinstance(n,ast.Assign) and
        any(isinstance(t,ast.Name) and t.id=='ROOT' for t in n.targets) and isinstance(n.value,ast.Constant))
    require(literal==root and 'ROOT = response.ROOT' in sources[RUNTIME],'old consumer fixed root binding')
    # Verify relevant source bytes against ORIGINAL runtime seals, no runtime import.
    prepared=inputs.json(runtime_dir/'prepare_receipt.json'); original_sources={}
    for seal in prepared['source_seals']:
        path=Path(seal['output_dir'])/'manifest.json'
        require(inputs.hash(path)==seal['manifest_sha256'],'original source manifest SHA')
        for e in inputs.json(path)['files']: original_sources[e['source']]=e['sha256']
    for name in sources:
        path=str(project/f'experiments/spectral_reachability/{name}.py')
        require(path in original_sources and inputs.hash(path)==original_sources[path],'sealed source unchanged: '+name)
    legal=consumption_contract(sources[RUNTIME],sources['p4_feedback_matched_20261027_v2'])
    cap_src=sources[CAPABILITY]
    require("if arm['selected_candidate_id'] == arm['greedy_candidate_id']:" in cap_src and 'continue' in cap_src,'qualification implementation edge')
    v2_path=results/'p4_feedback_attribution_audit_20261027_v2_verify02/report.json'
    v2=inputs.json(v2_path)
    require(v2['status']=='saved_terminal20_supplement_verified' and v2['historical_runs']==1 and v2['inputs_unchanged'] is True and
        v2['P1_admission'] is False and v2['P4_admission'] is False,'prior v2 scope')
    s=v2['audit_source_seal']; verify_seal(output_dir=s['output_dir'],expected_manifest_sha256=s['manifest_sha256'])
    for path,h in v2['inputs'].items(): require(inputs.hash(path)==h,'v2 evidence changed: '+path)
    exclusions=exclude_roots(qualified,root,cap_roots,response['root_sha256'],root,set(used))
    require(not any(e['engineering_prepare_eligible'] for e in exclusions),'new candidate source requires dedicated admission implementation; do not auto-generate a protocol')
    inv=inventory(results,inputs)
    inputs.stable()
    chain=[dict(stage='planner',path=str(planner_path),roots=used,
        excludes='geometry selected==greedy rejected by capability producer; model-only criterion, not task success'),
        dict(stage='capability',path=str(cap_path),roots=cap_roots,branches=len(cap['rows']),
             excludes='only old model-qualified roots have 3role x41 actual probe source packs'),
        dict(stage='feedback_response',path=str(response_dir/'protocol.json'),roots=[response['root_sha256']],
             excludes='roles are same-root aliases, not independent roots'),
        dict(stage='multiround_runtime',path=str(runtime_dir/'protocol.json'),roots=[root],
             excludes='hardcoded response.ROOT, root_index1, source/census/root fingerprint gate'),
        dict(stage='terminal20_and_v2',path=str(v2_path),roots=[root],
             excludes='derived native192 encodings and saved audits do not create new physics batches')]
    missing=[
        dict(code='root_disjoint_source_pack',needed='one root different from first batch with its own bound root RGB3/raw history/native context/goalQ/model/reset provenance and capability/feedback prefix witnesses',
             evidence='no root in current capability->response->runtime intersection is disjoint from first batch'),
        dict(code='new_root_parameterized_consumer',needed='NEW reviewed runtime/prepare source contract supporting that root; old root constant/source/census must remain immutable',
             evidence='current sealed runtime pins response.ROOT and original root_index1'),
        dict(code='prospective_selection_and_exclusion_manifest',needed='pre-outcome finite root roster/seed/task contract and full prior-use exclusions; different calibration root is not a fresh holdout',
             evidence='all four available upstream roots are already used; no fresh root selection manifest in this audited closure'),
        dict(code='P1_scientific_admission',needed='independent valid P1 mechanism/task admission before P4 scientific pass',
             evidence='P1_admission remains false; this is a scientific gate, not permission to alter engineering sources')]
    return dict(schema=TAG,status='second_batch_preparation_blocked_missing_inputs',
        can_prepare_second_batch=False,new_protocol_written=False,engineering_candidate_count=0,
        independent_second_batch_verified=False,historical_full20_batches=1,first_batch_root_sha256=root,
        first_batch_arms=['imagined','actual'],root_exclusion_chain=exclusions,source_protocol_exclusion_chain=chain,
        legal_real_feedback_consumption=legal,source_paths={n:str(project/f'experiments/spectral_reachability/{n}.py') for n in sources},
        missing_inputs=missing,inventory=inv,inputs=inputs.hashes,inputs_unchanged=True,
        future_execution_outputs_not_pre_run_missing_inputs=['own real boundary RGB and encoded context at steps5/10/15 generated during future guarded execution',
            'own real terminal20 RGB/native192 evaluation after final dispatch'],
        preparation_semantics='no protocol.json on blocked branch; no empty/sentinel source pack, no reselection based on saved physical outcomes',
        scope='complete four-root old geometry->capability->response->runtime closure plus listed local P4 protocol/report inventory; not an all-repository fresh-data existence proof',
        P1_admission=False,P4_admission=False,new_neural_calls=0,new_physics_steps=0,new_AD_calls=0,
        limitations=['selected==greedy exclusion is an old finite model diagnostic, not proof a root cannot support feedback',
            'root-disjoint old calibration is weaker than prospective fresh heldout independence',
            'future boundary observations need not exist before a new runtime can be prepared; do not fabricate them'])


def write(path,value):
    raw=canonical(value); require(len(raw)<1024*1024,'small output cap')
    with Path(path).open('xb') as stream: stream.write(raw); stream.flush(); os.fsync(stream.fileno())


def run(output):
    output=Path(output).resolve()
    require(not output.exists() and output.name.startswith(TAG),'exclusive new output required')
    output.mkdir(exist_ok=False)
    files=[Path(__file__),REPO/f'tests/test_{TAG}.py',REPO/f'innerdocs/{TAG}.md',
           REPO/'experiments/spectral_reachability/d2p_prerun_source_seal.py']
    seal=create_seal(files=files,args_json=json.dumps(dict(scope='saved-only second-batch prerequisites',neural_calls=0,physics_steps=0)),output_dir=output/'source_seal')
    try:
        result=audit(); result['audit_source_seal']=seal
        result['source_verification']=verify_seal(output_dir=seal['output_dir'],expected_manifest_sha256=seal['manifest_sha256'])
        write(output/'report.json',result); return result
    except BaseException as exc:
        write(output/'failure.json',dict(status='audit_integrity_failed_not_a_scientific_negative',type=type(exc).__name__,message=str(exc),audit_source_seal=seal)); raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path,required=True)
    result=run(**vars(parser.parse_args()))
    print(json.dumps({k:result[k] for k in ('status','can_prepare_second_batch','new_protocol_written','P4_admission')}))
