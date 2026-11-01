"""Exploratory saved-only latest versus whole-history IO comparison, all164/80.

No torch/model import or forward. MAIN alone executes production analyze/verify.
"""
import argparse
import ast
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from experiments.spectral_reachability import agent_p5_latest_slot_factorial_cuda_20261029_v1 as latest
from experiments.spectral_reachability import agent_p5_history_factorial_cuda_20261029_v1 as whole

NAME='agent_p5_latest_whole_cellwise_20261029_v1'
ROOT=latest.ROOT
CELLS=('AA','AP','PA','PP')
require,sha,raw=latest.require,latest.sha,latest.raw
CAP,FLOOR,WALL=64*1024**2,200*1024**2,1800
OWNER_PINS={'latest':'f5436a9ae922e717920c7662087f8380960c2c6a343503e2b8a7bc750b090ead',
    'whole':'2b012657c44f2c469a79abec590b14f384a20b444f251772a6d7624247554b7d'}
REPORT_PINS={'latest':'a9479f81d87b1b70ebd1f64db0289d9d2c6a5ae57932cd021f85cc5ecfd47316',
    'whole':'47b12d991d0b4c1c99ecf30ee90afe7bb5550f0babc4d786d6fca2e1cc5d8b8f'}
SOURCE_PINS={
 'experiments/spectral_reachability/p5_jepa_native_action_response_20261027_v8.py':'fc915ddee2e53ed58366994684fe2190da5eaaeb8323ad608ffe3292534c936a',
 'external/jepa-wms/app/plan_common/models/AdaLN_vit.py':'77e20661849a33199c581b4ab47e2d0417ef2e81571e4998a30cc242b68abe97',
 'external/jepa-wms/src/models/utils/modules.py':'aabc4baa04036c05142a4e314d1712141a38e31611121d56d21674750f303fe8',
 'external/jepa-wms/app/plan_common/models/prop_embedding.py':'1768f9a63a742413bf2de3109844bba8b6ed1f0ddb0c7b4c84724a5893110360'}


def protocol():
    return dict(schema=NAME+'.protocol',branches=164,cells=list(CELLS),pairs=latest.protocol()['pairs'],
        input_frames=[5,10,15],output_successor_frames=[10,15,20],direction='latest minus whole',
        comparisons='all actual native visual/action/proprio inputs by slot; both visual/proprio full3 and final20 outputs; all80 exact endpoint comparison joins',
        selection='no subset, mask selection, tuning, significance, fitted metric or new cohort',
        mechanism='source mask permits earlier history; equal outputs alone never establish shielding or invariance',
        exploratory_post_outcome=True,new_neural_calls=0,new_physics_calls=0,P1_admission=False,P5_admission=False,
        cap_bytes=CAP,disk_floor_bytes=FLOOR,wall_seconds=WALL)


def read(path,expected,used):
    require(type(expected) is str and len(expected)==64 and all(x in '0123456789abcdef' for x in expected),'explicit external SHA')
    path=Path(path).absolute();v=latest.bridge.read(path,expected,CAP)
    latest.merge(used,{str(path):expected});return v


def load_ref(ref,used):
    directory=Path(ref['directory']);r=ref['array'];p=directory/r['file']
    require(directory.is_absolute() and p.resolve()==p and Path(r['file']).name==r['file'] and
        not any(x.is_symlink() for x in (p,*p.parents)) and p.stat().st_size==r['file_bytes']<=4*1024**2 and sha(p)==r['sha256'],
        'bounded exact original NPY path/bytes/SHA')
    body=p.read_bytes();buf=io.BytesIO(body);a=np.load(buf,allow_pickle=False)
    require(buf.tell()==len(body) and a.dtype==np.float32 and r['dtype']=='float32' and list(a.shape)==r['shape'] and
        a.nbytes==r['nbytes'] and np.isfinite(a).all(),'finite full float32 single NPY EOF')
    latest.merge(used,{str(p):r['sha256']});return a


def diff(a,b):
    """a minus b; bytes distinguish signed zeros, ULP counts numeric zeros alike."""
    require(isinstance(a,np.ndarray) and isinstance(b,np.ndarray) and a.shape==b.shape and a.size>0 and
        a.dtype==b.dtype==np.float32 and np.isfinite(a).all() and np.isfinite(b).all(),'exact finite nonempty float32 comparison domain')
    a=np.ascontiguousarray(a);b=np.ascontiguousarray(b)
    ua=a.view(np.uint32);ub=b.view(np.uint32)
    # Monotone float32 lattice, with +0/-0 sharing the same numeric rank.
    def ordered(u):
        magnitude=(u & np.uint32(0x7fffffff)).astype(np.int64)
        return np.where((u & np.uint32(0x80000000))!=0,-magnitude,magnitude)
    delta=a.astype(np.float64)-b.astype(np.float64)
    return dict(shape=list(a.shape),bitwise_equal=a.tobytes()==b.tobytes(),numeric_equal=bool(np.array_equal(a,b)),
        differing_float32_words=int(np.count_nonzero(ua!=ub)),numeric_nonzero=int(np.count_nonzero(delta)),
        maxabs=float(np.max(np.abs(delta))),L2=float(np.linalg.norm(delta.ravel())),
        max_ULP=int(np.max(np.abs(ordered(ua)-ordered(ub)))),
        left_bytes_sha256=hashlib.sha256(a.tobytes()).hexdigest(),right_bytes_sha256=hashlib.sha256(b.tobytes()).hexdigest())


def slots(a,b,output=False):
    require(a.shape==b.shape and a.ndim>=2 and a.shape[:2]==(1,3),'full3 time axis1, not patch axis')
    result=dict(full3=diff(a,b),per_slot=[diff(a[:,i:i+1],b[:,i:i+1]) for i in range(3)])
    result['final20' if output else 'latest15']=result['per_slot'][2]
    return result


def identity(a,b,i):
    keys=('index','root_slot','condition','task_id','selection_digest','aligned_history_global_frames',
        'raw_outgoing16_to20','actual_inputs','rolled_inputs','AA','common_label','H2_label','label_png')
    require(type(a['index']) is int and type(b['index']) is int and a['index']==b['index']==i and
        all(type(a[k]) is type(b[k]) and a[k]==b[k] for k in keys),'exact same task/root/condition/actions/model-inputs/labels, no filename identity')
    require(a['aligned_history_global_frames']==[5,10,15] and (a['root_slot'],a['condition'])==latest.bridge.factor.ORDER[i//41],
        'fixed164 identity and time grid')


def evidence(manifests,reports,used):
    """Read source as text/AST only; pin exact executed-source closure, never import it."""
    records=[]
    for relative,h in SOURCE_PINS.items():
        p=ROOT/relative
        require(sha(p)==h and all(m['bindings'].get(str(p))==h for m in manifests),'exact source bound by BOTH actual prepares')
        latest.merge(used,{str(p):h});text=p.read_text();tree=ast.parse(text);lines=text.splitlines()
        if relative.endswith('AdaLN_vit.py'):
            targets={'VisionTransformerAdaLN':{'__init__','forward'},'FWAdaLNBlock':{'forward'}}
        elif relative.endswith('modules.py'):targets={'RoPEAttention':{'forward'}}
        elif relative.endswith('prop_embedding.py'):targets={'ProprioceptiveEmbedding':{'__init__','forward'}}
        else:targets={}
        snippets=[]
        for node in ast.walk(tree):
            selected=isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in ('load_cuda','build_action_block_causal_attention_mask')
            if isinstance(node,ast.ClassDef) and node.name in targets:
                for f in node.body:
                    if isinstance(f,ast.FunctionDef) and f.name in targets[node.name]:
                        snippets.append(dict(symbol=node.name+'.'+f.name,line=f.lineno,end_line=f.end_lineno,text='\n'.join(lines[f.lineno-1:f.end_lineno])))
            if selected:snippets.append(dict(symbol=node.name,line=node.lineno,end_line=node.end_lineno,text='\n'.join(lines[node.lineno-1:node.end_lineno])))
        records.append(dict(path=str(p),sha256=h,snippets=snippets))
    for r in reports:
        require(r['imported_sources']['app.plan_common.models.AdaLN_vit']==str(ROOT/'external/jepa-wms/app/plan_common/models/AdaLN_vit.py') and
            r['imported_sources']['src.models.utils.modules']==str(ROOT/'external/jepa-wms/src/models/utils/modules.py'),
            'source proof concerns actual imported predictor and attention')
    return dict(sources=records,block_time_allow_matrix=[[True,False,False],[True,True,False],[True,True,True]],
        final20_can_attend_input_frames=[5,10,15],earlier_history_architecturally_masked=False,
        reason='loader local_window=(3,-1,-1),feature proprio,0 condition tokens; builder local_window_time=T, t2<=t1; sliced causal mask passed into RoPE SDPA',
        limit='Allowed edges are not proof of effective sensitivity; no weights/attention execution or new interventions here. Numeric equality is not causal shielding.')


def bundle(kind,directory,report_sha,strict_receipt,strict_sha,used):
    owner=latest if kind=='latest' else whole
    require(sha(owner.__file__)==OWNER_PINS[kind] and report_sha==REPORT_PINS[kind],'frozen owner and exact actual report, no replacements')
    latest.merge(used,{str(Path(owner.__file__).resolve()):OWNER_PINS[kind]})
    supplied=read(strict_receipt,strict_sha,used)
    rebuilt=owner.consume(directory,report_sha)
    require(supplied==rebuilt and rebuilt['status']=='all164_'+('latest_slot' if kind=='latest' else 'history')+'_factorial_and80_pairs_verified',
        'complete original strict independently rebuilt, not status-only trust')
    d=Path(directory).absolute();r=read(d/'report.json',report_sha,used);binding=r['execution_binding']
    m=read(binding['prepared'],binding['prepare_sha256'],used)
    require(len(m['requests'])==len(r['rows'])==164 and len(r['pairs'])==80,'entire164/80')
    latest.merge(used,m['bindings'])
    for f,rec in r['artifacts'].items():latest.merge(used,{str(d/f):rec['sha256']})
    return d,r,m


def native_cell(kind,i,cell,directory,report,req,originals,used):
    if cell=='AA' or kind=='whole' and cell=='PP':
        key='H1' if cell=='AA' else 'H2';source,d=originals[key]
        call=source['calls'][(3*i+2) if key=='H1' else (4*i+3)]
        visual_ref=req['AA'] if cell=='AA' else req['PP']
    else:
        owner=latest if kind=='latest' else whole
        j=('AP','PA','PP').index(cell);arm=owner.ARMS[j];call=report['calls'][len(owner.ARMS)*i+j];d=directory
        require(call['arm']==arm,'actual new corner arm')
        visual_ref=dict(directory=str(d),array=report['rows'][i]['predictions'][arm])
    require(call['branch']==i and call['module']=='predictor' and call['status']=='returned','actual native completed branch attribution')
    load=lambda rec:load_ref(dict(directory=str(d),array=rec),used)
    args,kwargs=latest.h2.restore(call['inputs'],load);out=latest.h2.restore(call['outputs'],load)
    require(kwargs=={} and type(args) is tuple and len(args)==3 and all(a.shape==s for a,s in zip(args,latest.SHAPES,strict=True)),
        'all three complete native inputs')
    actual=tuple(load_ref(x,used) for x in req['actual_inputs']);rolled=tuple(load_ref(x,used) for x in req['rolled_inputs'])
    require(latest.h2.exact_array(actual[1],rolled[1]),'actual/rolled action identity')
    if cell=='AA':expected=actual
    elif kind=='latest':expected=latest.splice_inputs(actual,rolled,latest.ARMS[('AP','PA','PP').index(cell)])
    else:expected=(actual[0] if cell[0]=='A' else rolled[0],actual[1],actual[2] if cell[1]=='A' else rolled[2])
    require(all(latest.h2.exact_array(x,y) for x,y in zip(args,expected,strict=True)),'actual full native corner input provenance, no same-name substitution')
    require(type(out) is tuple and len(out)==3 and out[1] is None and out[0].shape==(1,3,256,384) and
        out[2].shape==(1,3,256,16) and latest.h2.exact_array(out[0],load_ref(visual_ref,used)),
        'full native visual AND proprio, original row visual pointer')
    return args,(out[0],out[2])


def compare_cell(a,b):
    ai,ao=a;bi,bo=b
    ins={k:slots(x,y) for k,x,y in zip(('visual','action','proprio'),ai,bi,strict=True)}
    outs={k:slots(x,y,output=True) for k,x,y in zip(('visual','proprio'),ao,bo,strict=True)}
    samein=all(v['full3']['bitwise_equal'] for v in ins.values())
    sameout=all(v['full3']['bitwise_equal'] for v in outs.values())
    status='identical_inputs_and_outputs_no_distinct_contrast' if samein and sameout else (
        'identical_inputs_different_outputs_execution_numeric_seam' if samein else (
        'different_inputs_identical_outputs_mechanism_unresolved' if sameout else 'different_inputs_different_outputs'))
    return dict(inputs=ins,outputs=outs,classification=status,architectural_shielding_certified=False)


def analyze(latest_output,latest_report_sha,latest_strict_receipt,latest_strict_sha,
            whole_output,whole_report_sha,whole_strict_receipt,whole_strict_sha):
    started=time.monotonic();used={}
    arguments=dict(latest_output=str(Path(latest_output).absolute()),latest_report_sha=latest_report_sha,
        latest_strict_receipt=str(Path(latest_strict_receipt).absolute()),latest_strict_sha=latest_strict_sha,
        whole_output=str(Path(whole_output).absolute()),whole_report_sha=whole_report_sha,
        whole_strict_receipt=str(Path(whole_strict_receipt).absolute()),whole_strict_sha=whole_strict_sha)
    def check():require(time.monotonic()-started<WALL,'complete saved-only1800s cap')
    ld,lr,lm=bundle('latest',latest_output,latest_report_sha,latest_strict_receipt,latest_strict_sha,used);check()
    wd,wr,wm=bundle('whole',whole_output,whole_report_sha,whole_strict_receipt,whole_strict_sha,used);check()
    require(lm['expected_model']==wm['expected_model']==lr['model_before']==wr['model_before'] and
        lm['expected_strict_loads']==wm['expected_strict_loads'] and lr['protocol']['pairs']==wr['protocol']['pairs']==protocol()['pairs'],
        'exact same actual model/schema and all80 schedule')
    source_evidence=evidence([lm,wm],[lr,wr],used)
    originals={}
    for key in ('H1','H2'):
        p=ROOT/'results'/latest.PINS[key][0];originals[key]=(read(p,latest.PINS[key][1],used),p.parent)
    branches=[]
    for i,(lq,wq) in enumerate(zip(lm['requests'],wm['requests'],strict=True)):
        check();identity(lq,wq,i)
        actual=tuple(load_ref(x,used) for x in lq['actual_inputs']);rolled=tuple(load_ref(x,used) for x in lq['rolled_inputs'])
        source_inputs={k:slots(a,b) for k,a,b in zip(('visual','action','proprio'),actual,rolled,strict=True)}
        cells={c:compare_cell(native_cell('latest',i,c,ld,lr,lq,originals,used),native_cell('whole',i,c,wd,wr,wq,originals,used)) for c in CELLS}
        branches.append(dict(index=i,root_slot=lq['root_slot'],condition=lq['condition'],task_id=lq['task_id'],
            selection_digest=lq['selection_digest'],actual_minus_rolled_input=source_inputs,cells=cells))
    # Exact pair schedule membership, without choosing any signs/cohort or interpreting outcomes.
    pairs=[dict(s,cells={c:dict(minus_branch_comparison=s['minus_index'],plus_branch_comparison=s['plus_index'],
        endpoint_full3_visual_equal=all(branches[s[k+'_index']]['cells'][c]['outputs']['visual']['full3']['bitwise_equal'] for k in ('minus','plus')),
        endpoint_final20_visual_equal=all(branches[s[k+'_index']]['cells'][c]['outputs']['visual']['final20']['bitwise_equal'] for k in ('minus','plus')))
        for c in CELLS}) for s in protocol()['pairs']]
    counts={c:{status:sum(b['cells'][c]['classification']==status for b in branches) for status in (
        'identical_inputs_and_outputs_no_distinct_contrast','identical_inputs_different_outputs_execution_numeric_seam',
        'different_inputs_identical_outputs_mechanism_unresolved','different_inputs_different_outputs')} for c in CELLS}
    for p in (Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')):latest.merge(used,{str(p):sha(p)})
    for p,h in used.items():check();require(sha(p)==h,'entire read source/model/array closure unchanged')
    return dict(schema=NAME+'.analysis',status='all164_four_cell_native_IO_compared',protocol=protocol(),arguments=arguments,
        branches=branches,pairs=pairs,classification_counts=counts,model_mask_source_evidence=source_evidence,bindings=used,
        strict_receipts_rebuilt=True,new_neural_calls=0,new_physics_calls=0,P1_admission=False,P5_admission=False,
        limitations=['Post-outcome explanatory diagnosis, not preregistered confirmation.',
            'No neural execution. Allowed mask edges do not establish effective influence or explain every numerical equality.',
            'Extra agent position/velocity; neither RGB-only nor physical mixed-context counterfactual.',
            'Common H1 label identity only checked here; no outcome scores/cohorts/ranking or Nash summary duplicated.',
            'All original164/80 retained, not independent roots or new scientific admission.'])


def write_new(value,path):
    p=Path(path).absolute();body=raw(value)
    require(p.parent==ROOT/'results' and p.name.startswith(NAME+'_') and not p.exists() and
        not any(x.is_symlink() for x in (p,*p.parents)) and len(body)<=CAP and shutil.disk_usage(p.parent).free>=FLOOR+len(body),
        'exclusive own new bounded analysis/no overwrite/floor')
    with p.open('xb') as f:require(f.write(body)==len(body),'whole write');f.flush();os.fsync(f.fileno())
    latest.bridge.physical.paired.fsync_dir(p.parent)
    return dict(status=value['status'],path=str(p),sha256=sha(p))


def verify(path,expected):
    v=read(path,expected,{})
    require(v['schema']==NAME+'.analysis' and v['protocol']==protocol() and v==analyze(**v['arguments']),
        'deterministic full original-strict/native-IO/source-evidence reconstruction')
    return dict(status='saved_cellwise_comparison_recomputed',analysis_sha256=expected,new_neural_calls=0,new_physics_calls=0)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('analyze')
    for kind in ('latest','whole'):
        for key in ('output','report-sha','strict-receipt','strict-sha'):a.add_argument('--'+kind+'-'+key,required=True)
    a.add_argument('--output',required=True)
    a=sub.add_parser('verify');a.add_argument('--analysis',required=True);a.add_argument('--analysis-sha',required=True)
    q=vars(p.parse_args());command=q.pop('command')
    if command=='analyze':output=q.pop('output');result=write_new(analyze(**q),output)
    else:result=verify(q['analysis'],q['analysis_sha'])
    print(json.dumps(result,sort_keys=True,allow_nan=False))


if __name__=='__main__':main()
