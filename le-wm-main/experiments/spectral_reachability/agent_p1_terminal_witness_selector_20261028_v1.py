"""Saved-only, fixed-library terminal-witness selector; no P1 admission.

No backend, neural inference, renderer, physical labels, or hyperparameter search.
Execute as a module from le-wm-main. MAIN owns production analyze/verify.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
from io import BytesIO
import json
from pathlib import Path
import numpy as np
from experiments.spectral_reachability import agent_allmode_prefix_20261028_v2 as base
from experiments.spectral_reachability import agent_mixedmode_prefix_20261028_v1 as mixed

NAME='agent_p1_terminal_witness_selector_20261028_v1'
ROOT=base.ROOT
require=base.require
SOURCES=(
    ('allmode',42,base,'agent_allmode_prefix_20261028_v2',
     'c11543eaa251903ff6d0429d2447f03c7427e1fe65be3d0aabfb0d9c4ef79d0d',
     'c3e38899a12cd163d75918da79ef029adb14b2479f74f15bdfac0b6ffc57242d'),
    ('mixedmode',222,mixed,'agent_mixedmode_prefix_20261028_v1',
     '8f6d027116b76334e80f9a761688ffd33825bc25d825484012d37230f083ed6b',
     'd6166b98958af70cecb8529135396c77467b7c803bfe12a1a5de29c422c22ad2'))


def bits_equal(a,b,shape,dtype=np.float32):
    """Do not erase signed zeros or forgive a second float32 conversion."""
    x=np.asarray(a);y=np.asarray(b)
    require(x.shape==y.shape==shape and np.isfinite(x).all() and np.isfinite(y).all(),
            'finite exact shape')
    return x.astype(dtype).tobytes()==y.astype(dtype).tobytes() and np.array_equal(x,y)


def sha_object(value):
    return hashlib.sha256(base.canonical(value)).hexdigest()


def prefix_context(initial,c):
    after=deepcopy(initial)
    after['latent_history']=deepcopy(initial['latent_history'][1:])+[deepcopy(c['predicted_prefix'])]
    after['action_history']=np.concatenate([initial['action_history'],
        base.kernel.normalized_history(c['prefix_commands'])]).tolist()[-2:]
    return after


def select_panel(packet,q,result,expected_count):
    """Pure selector over model fields. Production additionally validates execution.

    Missing terminal evidence is never an implicit ridge approximation or a
    smaller comparison cohort. Request objects are literal data, not execution.
    """
    q=base.validate_q(q);initial=packet['root_context'];base.validate_context(initial)
    goal=base.compiler.finite(packet['goal'],(192,));cs=result['candidates']
    qsha=base.digest(q.tolist())
    require(result['q_sha256']==qsha and len(cs)==expected_count,'Q and complete fixed denominator')
    anchor=base.cost(initial['latent_history'][-1],goal,q)
    records=[];requests=[]
    for i,c in enumerate(cs):
        require(type(c['index']) is int and c['index']==i and c['completed'] is True and
                c['q_sha256']==qsha,'stable complete candidate identity')
        prefix=np.tanh(np.asarray(c['prefix_params'],np.float32)).reshape(5,2)
        suffix=np.tanh(np.asarray(c['suffix_params'],np.float32)).reshape(5,2)
        require(bits_equal(c['prefix_commands'],prefix,(5,2)) and
                bits_equal(c['suffix_commands'],suffix,(5,2)) and
                bits_equal(c['commands'],np.concatenate([prefix,suffix]),(10,2)),
                'literal raw float32 bits; once tanh, same action witness')
        stage=base.cost(c['predicted_prefix'],goal,q)
        reach=float(c['own_reach_value']);objective=stage+reach
        require(np.isfinite(reach) and reach>=0 and c['stage_cost']==stage and
                c['objective']==objective,'exact unchanged beta1 stage plus ridge score')
        present=[c.get(k) is not None for k in ('predicted_terminal','nonlinear_terminal_cost','context')]
        require(all(present) or not any(present),'inconsistent partial terminal evidence')
        terminal=None
        if all(present):
            terminal=base.cost(c['predicted_terminal'],goal,q)
            end=deepcopy(initial)
            end['latent_history']=[deepcopy(initial['latent_history'][-1]),
                deepcopy(c['predicted_prefix']),deepcopy(c['predicted_terminal'])]
            end['action_history']=base.kernel.normalized_history(c['commands']).tolist()
            require(c['nonlinear_terminal_cost']==terminal and c['independent_model_replay_exact'] is True and
                    base.canonical(c['context'])==base.canonical(end),
                    'nonlinear terminal Q and full chronological replay context')
        else:
            requests.append(dict(index=i,goal_sha256=sha_object(packet['goal']),q_sha256=qsha,
                suffix_H1=dict(context=prefix_context(initial,c),raw=deepcopy(c['suffix_commands'])),
                full_H2=dict(context=deepcopy(initial),commands=deepcopy(c['commands'])),
                expected_prefix_sha256=sha_object(c['predicted_prefix']),
                required='suffix return equals independent H2 full context; no surrogate substitution'))
        records.append(dict(index=i,proposal_kind=c['proposal_kind'],stage_cost=stage,
            own_reach_value=reach,original_objective=objective,nonlinear_terminal_Q=terminal,
            raw_float32_sha256=hashlib.sha256(np.asarray(c['commands'],np.float32).tobytes()).hexdigest(),
            predicted_prefix_sha256=sha_object(c['predicted_prefix']),
            predicted_terminal_sha256=None if terminal is None else sha_object(c['predicted_terminal']),
            model_prefix_uphill_from_anchor=stage>anchor))
    old=min(records,key=lambda c:(c['original_objective'],c['index']))
    require(result['winner']['index']==old['index'] and result['winner']['objective']==old['original_objective'] and
            bits_equal(result['winner']['commands'],cs[old['index']]['commands'],(10,2)),
            'reproduce original stable surrogate winner')
    greedy=min(records,key=lambda c:(c['stage_cost'],c['index']))
    for c in records:
        c['model_prefix_worse_than_zero']=c['stage_cost']>records[0]['stage_cost']
        c['model_prefix_worse_than_stage_greedy']=c['stage_cost']>greedy['stage_cost']
    def selected(r,field):
        i=r['index']
        return dict(index=i,score=r[field],commands=deepcopy(cs[i]['commands']),
                    raw_float32_sha256=r['raw_float32_sha256'],
                    model_prefix_uphill_from_anchor=r['model_prefix_uphill_from_anchor'])
    terminal=None if requests else min(records,key=lambda c:(c['nonlinear_terminal_Q'],c['index']))
    return dict(candidate_denominator=expected_count,terminal_available=expected_count-len(requests),
        initial_Q=anchor,stage_greedy_index=greedy['index'],zero_prefix_index=0,
        context_sha256=sha_object(initial),goal_sha256=sha_object(packet['goal']),q_sha256=qsha,
        original_stage_plus_ridge=selected(old,'original_objective'),
        nonlinear_terminal_Q=None if terminal is None else selected(terminal,'nonlinear_terminal_Q'),
        candidates=records,literal_forward_requests=requests,
        status='complete_model_only' if not requests else 'blocked_missing_terminal_full_denominator_retained',
        P1_admission=False)


def array_loader(folder,report,meter):
    def load(a):
        name=a['file'];path=folder/name
        require(Path(name).name==name and not path.is_symlink() and
                a['sha256']==report['artifact_sha256'][name],'sealed flat model array')
        raw=path.read_bytes();meter['array_reads']+=1;meter['array_bytes_read']+=len(raw)
        require(hashlib.sha256(raw).hexdigest()==a['sha256'],'actual model array bytes hash')
        stream=BytesIO(raw);v=np.load(stream,allow_pickle=False)
        require(stream.tell()==len(raw) and str(v.dtype)==a['dtype'] and list(v.shape)==a['shape'] and
                v.dtype==np.float32 and np.isfinite(v).all(),'finite native float32 array schema')
        return v
    return load


def validate_raw_bits(slot,row,load):
    """Strengthens upstream numeric equality with exact signed bits/context."""
    want=base.expected_forward_inputs(slot['packet'],row['result'])
    for f,(ctx,raw) in zip(row['forwards'],want,strict=True):
        require(base.canonical(f['input'])==base.canonical(ctx) and
                bits_equal(f['raw'],raw,(5,2)) and
                bits_equal(load(f['raw_array']),raw,(5,2)),
                'exact raw/context bits for actual full schedule')


def build():
    """Read only the two pinned model reports/prepares and sealed model arrays.

    This is not a replacement upstream whole-run consume or freshness verifier.
    No recursive parent build, model load, image load, or physical outcome read.
    """
    panels=[];inputs=[];common=None;resources=[]
    meter=dict(array_reads=0,array_bytes_read=0)
    for family,n,module,stem,prep_sha,report_sha in SOURCES:
        prepared=ROOT/'results'/f'{stem}_prepare_MAIN01.json'
        folder=ROOT/'results'/f'{stem}_cuda_MAIN01';report_path=folder/'report.json'
        require(base.sha(prepared)==prep_sha and base.sha(report_path)==report_sha,'pinned original inputs')
        p=base.overlay.read(prepared,module.PREP_CAP);r=base.overlay.read(report_path,module.CAP)
        require(p['schema']==stem+'.prepare' and r['schema']==stem+'.report' and
                p['contract']==r['contract']==module.contract() and r['errors']==[] and
                r['first_error'] is None and r['commands_released'] is r['source_verified'] is r['model_unchanged'] is True and
                r['prepare_sha256']==prep_sha and Path(r['prepared']).resolve()==prepared.resolve() and
                r['model_hash_before']==r['model_hash_after']==p['expected_model_hash'],
                'complete frozen model/contract/prepare identity')
        require(len(p['slots'])==len(r['rows'])==4,'all four root-condition slots')
        # Verify direct numerical helpers against the original prepare, without
        # rebuilding historical pipelines or importing any physical runner.
        for m in (base,mixed,base.kernel,base.compiler):
            path=Path(m.__file__).resolve()
            if str(path) in p['bindings']:
                require(base.sha(path)==p['bindings'][str(path)],'frozen direct numerical helper')
        slots_signature=sha_object([dict(index=s['index'],root_slot=s['root_slot'],condition=s['condition'],
            packet=s['packet'],q=s['q'],task_sha256=s['task_sha256'],blind_sha256=s['blind_sha256']) for s in p['slots']])
        require(common is None or common==slots_signature,'same four exact encoded tasks across libraries')
        common=slots_signature
        load=array_loader(folder,r,meter)
        for i,(slot,row) in enumerate(zip(p['slots'],r['rows'],strict=True)):
            require(slot['index']==row['index']==i and
                    (slot['root_slot'],slot['condition'])==(row['root_slot'],row['condition'])==base.ORDER[i] and
                    row['status']=='plan_sealed','fixed ordered task identity')
            # This original validator recomputes every original bank/ridge solve
            # and checks all saved native module bridges, replays, forward/AD fees.
            module.validate_execution(slot,row,load)
            validate_raw_bits(slot,row,load)
            panel=select_panel(slot['packet'],slot['q'],row['result'],n)
            panels.append(dict(family=family,index=i,root_slot=slot['root_slot'],condition=slot['condition'],
                               task_sha256=slot['task_sha256'],**panel))
        require(r['math_resources']==base.math_resources(r['rows']),'original mathematical ledger')
        expected={k:dict(attempted=4*(6*n+6),completed=4*(6*n+6)) for k in base.Modules.ORDER}
        require(r['resources']==expected,'original full module/API fees')
        resources.append(dict(family=family,original_native_resources=r['resources'],
            original_mathematical_resources=r['math_resources'],
            new_saved_validation_root_banks=4,new_saved_validation_suffix_solves=4*n,
            new_saved_validation_successful_helper_SVD=4*(n+2)))
        require(base.sha(prepared)==prep_sha and base.sha(report_path)==report_sha,'inputs unchanged after read')
        inputs.append(dict(family=family,prepared=str(prepared),prepare_sha256=prep_sha,
                           report=str(report_path),report_sha256=report_sha))
    source_paths=[Path(__file__),ROOT/'tests'/f'test_{NAME}.py',ROOT/'innerdocs'/f'{NAME}.md',
                  Path(base.__file__),Path(mixed.__file__),Path(base.kernel.__file__),Path(base.compiler.__file__)]
    return dict(schema=NAME+'.analysis',status='new_exploratory_model_only_selector_not_admitted',
        inputs=inputs,source_sha256={str(p.resolve()):base.sha(p) for p in source_paths},
        root_condition_denominator=8,candidate_denominator=1056,
        unique_encoded_root_conditions=4,independent_physical_roots=2,
        overlapping_first42_retained_per_library=True,panels=panels,
        resources=dict(per_library=resources,saved_io=meter,new_neural_calls=0,new_physics_calls=0,
            new_training=0,new_Q_fits=0,new_saved_validation_successful_helper_SVD=1072,
            selector_cost_evaluations=8+2*1056,internal_solve_iterations=None,total_FLOPs=None,
            inherited_costs='historical16 encodes, prior planning and failed allmode v1 paid separately; not erased',
            equal_total_compute=False),
        definitions=dict(original='stable argmin(stage_cost + own_reach_value, index); unchanged beta=1',
            terminal='stable argmin(mean((Q@(predicted_terminal-goal))**2), index)',
            scope='same fixed prefix and own ridge suffix witness; NOT nonlinear suffix optimization',
            uphill='strict model stage_cost > anchor Q; zero/greedy comparisons separately labeled',
            restriction='no physical/native-state/RGB/outcome input to selection; no root filtering',
            authority='pinned model evidence + original per-row execution validator; not full upstream consume',
            missing='no terminal argmin over subset; literal suffix H1 plus independent H2 requests'),
        new_algorithm=True,retrospective_preregistration=False,independent_confirmation=False,P1_admission=False)


def write_new(path,value):
    path=Path(path)
    require(path.name.startswith(NAME+'_') and path.suffix=='.json' and not path.exists(),
            'new exclusive-stem JSON only')
    raw=base.canonical(value)
    require(len(raw)<8*1024**2,'compact derivative cap')
    with path.open('xb') as f:f.write(raw)
    return dict(path=str(path.resolve()),sha256=base.sha(path))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('analyze','verify'))
    parser.add_argument('--output');parser.add_argument('--result');parser.add_argument('--sha')
    a=parser.parse_args(argv)
    if a.command=='analyze':out=write_new(a.output,build())
    else:
        require(a.result is not None and a.sha is not None and base.sha(a.result)==a.sha,'external derivative SHA')
        require(base.overlay.read(a.result,8*1024**2)==build(),'full deterministic saved reconstruction')
        out=dict(status='verified_model_only_not_admitted',sha256=a.sha,P1_admission=False)
    print(json.dumps(out,ensure_ascii=False,sort_keys=True));return 0


if __name__=='__main__':raise SystemExit(main())
