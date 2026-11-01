"""Exploratory evaluation of sealed model choices on exact existing executions only."""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import numpy as np
from experiments.spectral_reachability import agent_p1_model_library_selector_20261028_v1 as selector
from experiments.spectral_reachability import agent_p1_task_metric_mechanism_panel_20261028_v1 as panel

NAME = 'agent_p1_model_library_selector_evaluation_20261028_v1'
ROOT, teacher, rgb = selector.ROOT, selector.teacher, panel.rgb
require, sha, canonical, digest = selector.require, selector.sha, selector.canonical, selector.digest
SELECTOR = ROOT/'results/agent_p1_model_library_selector_20261028_v1_MAIN01.json'
SELECTOR_SHA = 'f807d2ffe2d7ffb4d012852302f199b9edb64afc5a23cc36ac264c6adea6c1bb'
PANEL = ROOT/'results/agent_p1_task_metric_mechanism_panel_20261028_v1_analysis_NASH02.json'
PANEL_SHA = '7eda1333a3bf2c8ef1d46deaf67d74c6d464498c018b52cf35e9802204501e5d'
CAP = 32*1024**2


def array_sha(value):
    return hashlib.sha256(value.tobytes()).hexdigest()


def join_slot(candidate, row, physical, job, anchor, blind, returns, histories):
    """Evaluation-only exact join, including all unselected candidates and failed RGB12."""
    p = candidate['original_plan']; i = p['index']
    selector.validate_candidate(candidate,i)
    require(all(row[k] == physical[k] == p[k] for k in ('index','root_slot','condition','role','library_index','library_label',
        'anchor_id','blind_sha256','task_id','plan_digest')) and row['job_digest'] == physical['job_digest'] == job['job_digest'],
        'full model/panel/native/job identity, never name/index alone')
    require(all(physical[k] == v for k,v in job.items()), 'original physical prepared job immutable')
    require(digest({k:v for k,v in job.items() if k != 'job_digest'}) == job['job_digest'], 'original complete physical job digest')
    full, prefix, suffix = selector.raw(p['commands'],10), selector.raw(p['prefix_commands'],5), selector.raw(p['suffix_commands'],5)
    require(all(selector.raw(physical[k],n).tobytes() == selector.raw(p[k],n).tobytes()
        for k,n in (('commands',10),('prefix_commands',5),('suffix_commands',5))) and
        candidate['full10_raw_bits'] == full.tobytes().hex() and row['suffix_raw_bits'] == suffix.tobytes().hex(),
        'same complete original prefix and suffix actions, no new dispatch or word-only reuse')
    require(physical['initial_exact'] is physical['history_exact'] is physical['start_verified'] is physical['teacher_history_verified'] is True,
        'actual original placement/history/start authenticated, not nominal root metadata')
    context_digest = digest(blind['context'])
    require(context_digest == p['teacher_context_sha256'] == physical['teacher_context_sha256'] and
        anchor['anchor_id'] == p['anchor_id'] and anchor['task_id'] == p['task_id'] and
        anchor['condition'] == p['condition'] and anchor['role'] == p['role'] and
        anchor['goal'] == blind['goal'] == row['goal192'] and np.array_equal(blind['q'],np.eye(192)),
        'same compiler context and exact original latent goal/Q, not posthoc residual')
    require(candidate['prediction_witness']['goal_sha256'] == array_sha(np.asarray(row['goal192'],np.float32)) and
        candidate['prediction_witness']['shifted_context'] == row['array_receipts']['prediction'] and
        p['model_Q'] == row['predicted_Q'], 'prediction/goal array binding, endpoint semantics inherited')
    require(physical['goal'] == row['physical_goal'] == row['goal_binding']['state'] and
        physical['goal_pose'] == row['outline_pose'] == row['goal_binding']['pose'] and
        physical['goal_pixel_sha256'] == row['goal_binding']['pixel_sha256'], 'same native goal7/outline/pixel contract')
    root, start, end = (row['boundary_trace'][k]['state'] for k in ('10','15','20'))
    require(physical['root_state'] == root and physical['start_state'] == start and physical['last_state'] == end == row['returned_state'],
        'actual native root10/start15/return20, no approximate saved state restore')
    checkpoints = [cp for cp in physical['checkpoints'] if cp['step'] == 5]
    require(len(checkpoints) == 1, 'one actual suffix-start checkpoint')
    frame = checkpoints[0]['frame']
    require(checkpoints[0]['absolute_step'] == 15 and frame['state'] == start and
        frame['sha256'] == anchor['png_sha256'] and frame['pixel_sha256'] == anchor['pixel_sha256'] and
        frame['pre'] == frame['post'], 'same actual teacher15 RGB and immutable native snapshot')
    warm = panel.base.physics.moving.warmup_commands()
    expected = np.concatenate((warm,full))
    phases = [('warmup',j) for j in range(1,11)]+[('prefix',j) for j in range(11,16)]+[('suffix',j) for j in range(1,6)]
    require(len(returns) == 20, 'all20 actual returns including failed final-render12, no failed denominator loss')
    for e,(phase,step),raw in zip(returns,phases,expected,strict=True):
        require(e['phase'] == phase and e['step'] == step and e['raw_bits'] == e['actual_bits'] == raw.tobytes().hex() and
            e['actual_dtype'] == '<f4' and e['terminated'] is e['truncated'] is False,
            'every actually returned action/raw dtype/phase exact, no padding')
    require(returns[9]['state'] == root and returns[14]['state'] == start and returns[19]['state'] == end and
        returns[19]['event_sha256'] == row['actual_return_event_sha256'], 'actual journal state and terminal return binding')
    incoming = np.concatenate((warm[5:],prefix))
    require(len(histories) == 1 and histories[0]['context_sha256'] == context_digest and
        histories[0]['teacher_png_sha256'] == anchor['png_sha256'] and
        selector.raw(histories[0]['raw_history'],10).tobytes() == incoming.tobytes() == selector.raw(blind['raw_history'],10).tobytes() and
        histories[0]['tokens'] == blind['context']['action_history'] == teacher.kernel.normalized_history(incoming).tolist(),
        'actual warm/prefix incoming history and original model tokens, not evaluator state fed to compiler')
    return dict(index=i, plan_digest=p['plan_digest'], prediction_digest=candidate['prediction_digest'],
        job_digest=job['job_digest'], teacher_context_sha256=context_digest, prefix_sha256=array_sha(prefix),
        suffix_sha256=array_sha(suffix), full10_sha256=array_sha(full), incoming_history_sha256=array_sha(incoming),
        native_goal=deepcopy(row['physical_goal']), latent_goal_sha256=candidate['prediction_witness']['goal_sha256'],
        actual_root10=deepcopy(root), actual_start15=deepcopy(start), teacher15_png_sha256=anchor['png_sha256'],
        teacher15_pixel_sha256=anchor['pixel_sha256'], start_snapshot_digest=digest(frame['pre']),
        actual_return_event_sha256=row['actual_return_event_sha256'], physical_source_status=physical['status'],
        actual20_verified=True, exact_existing_execution_reuse=True, newly_executed=False)


def benefit(a,b):
    return dict(physical_metrics_higher_is_better={k:None if (d:=panel.delta(b['physical_metrics'][k],a['physical_metrics'][k])) is None
        else orientation*d for k,orientation in panel.base.METRICS.items()},
        native_reward_gain=panel.delta(a['returned_reward'],b['returned_reward']),
        predicted_Q_gain=panel.delta(b['predicted_Q'],a['predicted_Q']), actual_latent_Q_gain=panel.delta(b['actual_Q'],a['actual_Q']),
        reference_index=b['index'], evaluated_index=a['index'])


def outcome(row, zero):
    return dict(original_index=row['index'], word=row['library_label'], plan_digest=row['plan_digest'],job_digest=row['job_digest'],
        predicted_Q=row['predicted_Q'],actual_Q=row['actual_Q'],physical_metrics=deepcopy(row['physical_metrics']),
        native_reward=row['returned_reward'], native_success=row['native_success'],
        own_zero_comparison=benefit(row,zero), unknown_actual_latent=row['actual_Q'] is None)


def assemble(sealed, rows, joins):
    require(sealed['contract'] == selector.contract() and len(sealed['candidates']) == len(rows) == len(joins) == 40 and
        len(sealed['selections']) == 8 and [r['index'] for r in rows] == list(range(40)), 'full sealed8/all40 evaluation denominator')
    groups=[]
    for i,(c,r,j) in enumerate(zip(sealed['candidates'],rows,joins,strict=True)):
        p=c['original_plan']
        require(p['index'] == r['index'] == j['index'] == i and p['plan_digest'] == r['plan_digest'] == j['plan_digest'] and
            j['prediction_digest'] == c['prediction_digest'] and j['job_digest'] == r['job_digest'] and j['actual20_verified'] is True,
            'all40 independently joined existing physical outcomes, no drop/replacement')
        require((r['actual_Q'] is None) == (i==12), 'original index12 unknown preserved, not failure-imputed')
    for anchor,s in enumerate(sealed['selections']):
        require(s['anchor_index'] == anchor and s['selection_digest'] == digest({k:v for k,v in s.items() if k != 'selection_digest'}),
            'immutable original sealed selection witness, no physical reranking')
        i=s['selected_original_index']; d=s['original_generated_diagonal']['original_index']; z=anchor*5
        require(z <= i < z+5 and d == z+(1 if anchor%2==0 else 3) and
            s['selected_original_plan_digest'] == rows[i]['plan_digest'] and
            s['selected_prediction_digest'] == joins[i]['prediction_digest'] and
            s['original_generated_diagonal']['original_plan_digest'] == rows[d]['plan_digest'] and
            s['commands_raw_bits'] == rows[i]['suffix_raw_bits'] and s['teacher_context_sha256'] == joins[i]['teacher_context_sha256'],
            'exact selected/diagonal original plans, suffix bits and context')
        require(all(joins[k]['actual_start15'] == joins[z]['actual_start15'] and
            joins[k]['start_snapshot_digest'] == joins[z]['start_snapshot_digest'] for k in range(z,z+5)),
            'all five independent existing executions have identical suffix start, not approximate same-state claim')
        groups.append(dict(anchor_index=anchor,root_slot=s['root_slot'],condition=s['condition'],role=s['role'],
            selection_digest=s['selection_digest'], fixed_slot_indices=list(range(z,z+5)),
            exploratory_model_selector=outcome(rows[i],rows[z]), original_generated_diagonal=outcome(rows[d],rows[z]),
            own_zero=outcome(rows[z],rows[z]), selector_minus_diagonal=benefit(rows[i],rows[d]),
            same_original_candidate=i==d, same_action_as_original_diagonal=rows[i]['suffix_raw_bits']==rows[d]['suffix_raw_bits'],
            new_policy_actually_dispatched=False, causal_gain_inferred=False))
    return dict(fixed_slots=40,fixed_policy_pairs=8,rows=deepcopy(rows),exact_reuse_witnesses=deepcopy(joins),groups=groups,
        known_roots=2,fresh_roots=0,missing_actual_latent=[12],actual_latent_available=39,
        policy_identities=dict(selector='new retrospective exploratory model-Q stable selector',
            diagonal='original model-generated diagonal, not five-word argmin policy'))


def build():
    # Seal and verify model-only choice BEFORE opening evaluator outcomes. Never give labels to selector.
    sealed=teacher.read(SELECTOR,SELECTOR_SHA,selector.CAP)
    require(sealed == selector.build(), 'sealed MAIN selector independently rebuilt without evaluator outcomes')
    authority=teacher.read(PANEL,PANEL_SHA,CAP)
    require(authority == panel.build(), 'entire strict40/180 native panel/array/source closure')
    path=ROOT/rgb.INPUTS['physical'][0]; physical=teacher.read(path,rgb.INPUTS['physical'][1],CAP)
    ep=physical['execution_binding']; prep=teacher.read(ep['prepared'],ep['prepare_sha256'],teacher.PREP_CAP)
    require(ep['compiler_report_sha256'] == selector.REPORT_SHA and ep['compiler_consume_sha256'] == selector.RECEIPT_SHA,
        'same compiler/source executed by physical producer')
    model=teacher.read(selector.REPORT,selector.REPORT_SHA,selector.CAP)
    events=panel.base.physics.read_events(path.parent/physical['journal']['file'],physical['journal_tail'])
    require(len(physical['rows']) == len(prep['jobs']) == 40 and len(prep['anchors']) == len(prep['teacher_inputs']) == 8,
        'real physical prepared full40/eight-context ABI')
    joins=[]
    for i,c in enumerate(sealed['candidates']):
        a=i//5; scoped=[e for e in events if e.get('job_digest') == physical['rows'][i]['job_digest']]
        require(prep['teacher_inputs'][a] == model['rows'][a]['blind_input'], 'exact original model input vs executed prepare')
        joins.append(join_slot(c,authority['rows'][i],physical['rows'][i],prep['jobs'][i],prep['anchors'][a],prep['teacher_inputs'][a],
            [e for e in scoped if e['kind']=='raw_return'],[e for e in scoped if e['kind']=='teacher_history_verified']))
    bindings=deepcopy(sealed['bindings'])
    for path,h in authority['bindings'].items():
        require(path not in bindings or bindings[path]==h, 'no conflicting model/evaluator source closure')
        bindings[path]=h
    bindings.update({str(SELECTOR):SELECTOR_SHA,str(PANEL):PANEL_SHA})
    own=[Path(__file__).resolve(),ROOT/f'tests/test_{NAME}.py',ROOT/f'innerdocs/{NAME}.md']
    for f in own:bindings[str(f)]=sha(f)
    rgb.verify_bindings(bindings)
    return dict(schema=NAME+'.evaluation',status='exact_seen_execution_reuse_exploratory_not_admitted',bindings=bindings,
        source_files={str(f):bindings[str(f)] for f in own},selector_sha256=SELECTOR_SHA,panel_sha256=PANEL_SHA,
        boundary_array_receipts=deepcopy(authority['boundary_array_receipts']),
        P1_admission=False,fresh_confirmation=False,C2_claim=False,physical_release=False,
        selector_received_physical_labels=False,new_neural_calls=0,new_physics_calls=0,
        interpretation='retrospective policy evaluation on previously seen known-root executions, not new deployment/fresh validation',
        **assemble(sealed,authority['rows'],joins))


def write(path,value):
    p=Path(path).absolute();data=canonical(value)
    require(p.resolve()==p and p.parent==ROOT/'results' and p.name.startswith(NAME+'_') and p.suffix=='.json' and len(data)<=CAP,
        'exclusive bounded own-stem new output')
    with p.open('xb') as f:
        require(f.write(data)==len(data),'whole evaluation write');f.flush();os.fsync(f.fileno())
    teacher.sync_dir(p.parent);return dict(output=str(p),evaluation_sha256=sha(p))


def verify(path,expected,rebuilt):
    require(teacher.read(path,expected,CAP)==rebuilt,'entire evaluation rebuilt, not hash-only receipt')
    return dict(verified=True,evaluation_sha256=expected)


def consume(path,evaluation_sha):
    value=build();verify(path,evaluation_sha,value)
    return dict(schema=NAME+'.consume',evaluation_sha256=evaluation_sha,fixed_slots=40,fixed_policy_pairs=8,
        P1_admission=False,fresh_confirmation=False,C2_claim=False,physical_release=False)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=('analyze','verify','consume'))
    ap.add_argument('--output');ap.add_argument('--evaluation');ap.add_argument('--evaluation-sha');args=ap.parse_args()
    if args.command=='consume':out=consume(args.evaluation,args.evaluation_sha)
    else:
        value=build();out=write(args.output,value) if args.command=='analyze' else verify(args.evaluation,args.evaluation_sha,value)
    print(canonical(out).decode())


if __name__=='__main__':main()
