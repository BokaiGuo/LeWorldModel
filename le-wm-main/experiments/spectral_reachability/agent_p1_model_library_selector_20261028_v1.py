"""Exploratory saved model-only stable five-word selector; no experiment dispatch."""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import math
import os
from pathlib import Path
import numpy as np
from experiments.spectral_reachability import agent_p1_teacher_cross_library_20261028_v1 as teacher

NAME = 'agent_p1_model_library_selector_20261028_v1'
ROOT = teacher.ROOT
require, sha, canonical, digest = teacher.require, teacher.sha, teacher.canonical, teacher.digest
REPORT = ROOT/'results/agent_p1_teacher_cross_library_20261028_v1_cuda_MAIN01/report.json'
REPORT_SHA = 'd5ab99d55bacd73c8d9cf0fb774819bb46af63100053198858e639d8fa746738'
RECEIPT = ROOT/'results/consume_teacher_cross_MAIN_20261028_v1_MAIN01.json'
RECEIPT_SHA = 'd82109eb8de582e59a4f0485307c01c9b91f9f84551c218c61cca4da24335213'
CAP = 16*1024**2
PLAN_KEYS = {'schema','index','task','root_slot','condition','role','library_index','library_label','anchor_id',
    'blind_sha256','task_id','prefix_commands','suffix_commands','commands','raw_authority',
    'teacher_context_sha256','model_Q','selection_uses_probe_response','selection_uses_physics','P1_admission','plan_digest'}
CANDIDATE_KEYS = {'original_plan','prediction_witness','prediction_digest','full10_raw_bits','suffix_raw_bits'}
WITNESS_KEYS = {'source_report_sha256','source_receipt_sha256','anchor_index','library_index','native_prediction',
    'shifted_context','endpoint_sha256','goal_sha256','Q','semantics'}


def contract():
    return dict(schema=NAME+'.contract', algorithm='stable_argmin(predicted_Q,library_index)',
        library=list(teacher.LIBRARY), fixed_candidates=40, fixed_selections=8,
        order='task(root0off,root0on,root1off,root1on), role(Z,S), original library index',
        cost='original identity192 mean((predicted_endpoint-goal)^2), no fitted score or threshold',
        missing_cost='fail closed, no subset/fallback', ties='exact numerical equality then smallest original index; signed zero is a tie',
        selected_commands='suffix5 float32 raw2 ONLY, dispatch after same actual prefix5/context15; not full10',
        original_diagonal='Z:uZ, S:uS; original generator identity, NOT prior argmin deployment',
        exploratory=True, retrospective_known_inputs=True, original_policy_used_this_selector=False,
        candidate_costs_complete_before_selection=True, selection_uses_physics=False,
        physical_release=False, P1_admission=False, fresh_confirmation=False,
        known_roots=2, fresh_roots=0, new_neural_calls=0, new_physics_calls=0,
        new_model_compute=0, total_FLOPs=None, inherited_compilation_cost_not_free=True)


def raw(commands, n):
    a = np.asarray(commands, dtype='<f4'); original = np.asarray(commands, dtype=np.float64)
    require(a.shape == original.shape == (n,2) and np.isfinite(a).all() and
        np.array_equal(a.astype(np.float64), original) and np.max(np.abs(a)) <= 1,
        'literal rawfloat32 commands, no silent rounding/scaling/clipping')
    return a


def validate_candidate(candidate, index):
    require(set(candidate) == CANDIDATE_KEYS, 'model-only candidate whitelist; no outcome labels')
    p, w = candidate['original_plan'], candidate['prediction_witness']
    require(set(p) == PLAN_KEYS and set(w) == WITNESS_KEYS, 'exact frozen model plan/prediction ABI')
    task, rem = divmod(index,10); role, word = divmod(rem,5)
    require((p['schema'],p['index'],p['task'],p['root_slot'],p['condition'],p['role'],p['library_index'],p['library_label']) ==
        (teacher.NAME+'.model_plan',index,task,task//2,('off','on')[task%2],('Z','S')[role],word,teacher.LIBRARY[word]),
        'fixed original all40 identities/order, no filtered candidates')
    cost = p['model_Q']
    require(type(cost) in (int,float) and math.isfinite(cost) and cost >= 0, 'complete finite original predicted Q, no unknown/boolean cost')
    require(p['plan_digest'] == digest({k:v for k,v in p.items() if k != 'plan_digest'}), 'original complete model plan digest')
    require(p['selection_uses_probe_response'] is p['selection_uses_physics'] is p['P1_admission'] is False and
        p['raw_authority'] == 'literal_float32', 'original blind action authority, no physics-dependent selector')
    full, suffix, prefix = raw(p['commands'],10), raw(p['suffix_commands'],5), raw(p['prefix_commands'],5)
    require(full.tobytes() == np.concatenate((prefix,suffix)).tobytes() and
        full.tobytes().hex() == candidate['full10_raw_bits'] and suffix.tobytes().hex() == candidate['suffix_raw_bits'],
        'exact full10 and suffix5 original raw bytes')
    require(w['source_report_sha256'] == REPORT_SHA and w['source_receipt_sha256'] == RECEIPT_SHA and
        w['anchor_index'] == index//5 and w['library_index'] == word and w['Q'] == cost and
        candidate['prediction_digest'] == digest(w), 'prediction/source identity and Q binding')
    return p


def select40(candidates):
    """Pure model-only ABI. No physical report, labels, root state or evaluator input."""
    require(len(candidates) == 40, 'full40 prediction denominator, never subset selection')
    plans = [validate_candidate(c,i) for i,c in enumerate(candidates)]
    selections = []
    for anchor in range(8):
        bank = candidates[anchor*5:anchor*5+5]; pp = plans[anchor*5:anchor*5+5]
        first = pp[0]
        for p in pp:
            require(all(p[k] == first[k] for k in ('task_id','anchor_id','blind_sha256','teacher_context_sha256','prefix_commands')),
                'same original teacher context/history/goal action input for all five candidates')
        require(bank[0]['suffix_raw_bits'] == np.zeros((5,2),'<f4').tobytes().hex(), 'fixed zero candidate retained')
        for plus, minus in ((1,2),(3,4)):
            require(raw(pp[minus]['suffix_commands'],5).tobytes() == (-raw(pp[plus]['suffix_commands'],5)).tobytes(),
                'original symmetric raw library, not time reversal')
        chosen = min(range(5), key=lambda j:(pp[j]['model_Q'],j))
        diagonal = 1 if first['role'] == 'Z' else 3
        c, p = bank[chosen], pp[chosen]
        row = dict(schema=NAME+'.selection', selector_contract_digest=digest(contract()),
            anchor_index=anchor, root_slot=p['root_slot'],condition=p['condition'],role=p['role'],
            task_id=p['task_id'], anchor_id=p['anchor_id'], blind_sha256=p['blind_sha256'],
            teacher_context_sha256=p['teacher_context_sha256'], source_report_sha256=REPORT_SHA,
            source_receipt_sha256=RECEIPT_SHA, selected_original_index=p['index'], library_index=chosen,
            library_label=p['library_label'], cost=p['model_Q'], selected_original_plan_digest=p['plan_digest'],
            selected_prediction_digest=c['prediction_digest'], prediction_witness=deepcopy(c['prediction_witness']),
            commands=deepcopy(p['suffix_commands']), commands_shape=[5,2], commands_dtype='<f4',
            commands_raw_bits=c['suffix_raw_bits'], commands_sha256=hashlib.sha256(raw(p['suffix_commands'],5).tobytes()).hexdigest(),
            original_full10_commands=deepcopy(p['commands']), original_full10_raw_bits=c['full10_raw_bits'],
            required_original_prefix_commands=deepcopy(p['prefix_commands']),
            fixed_candidate_costs=[dict(original_index=x['index'],library_index=j,library_label=x['library_label'],
                cost=x['model_Q'],original_plan_digest=x['plan_digest'],prediction_digest=bank[j]['prediction_digest']) for j,x in enumerate(pp)],
            exact_minimum_tie_indices=[j for j,x in enumerate(pp) if x['model_Q'] == p['model_Q']],
            original_generated_diagonal=dict(original_index=pp[diagonal]['index'], library_index=diagonal,
                library_label=pp[diagonal]['library_label'], original_plan_digest=pp[diagonal]['plan_digest'],
                prediction_digest=bank[diagonal]['prediction_digest'], cost=pp[diagonal]['model_Q'],
                was_five_word_argmin_policy=False),
            same_as_original_diagonal=chosen == diagonal, physical_release=False, P1_admission=False)
        row['selection_digest'] = digest(row)
        selections.append(row)
    for task in range(4):
        z,s = plans[task*10:task*10+5], plans[task*10+5:task*10+10]
        require(z[0]['task_id'] == s[0]['task_id'] and all(raw(a['suffix_commands'],5).tobytes() == raw(b['suffix_commands'],5).tobytes()
            for a,b in zip(z,s,strict=True)), 'same original common library after both roles, no duplicate removal')
    return dict(contract=contract(), fixed_candidates=40, fixed_selections=8,
        candidates=deepcopy(candidates), selections=selections)


def make_candidate(plan, row, directory):
    """Only producer model outputs enter the pure selector, never physical evidence."""
    word = plan['library_index']; e = row['evaluations'][word]; f = row['forwards'][word+2]
    require(e['stage'] == f['stage'] == f'evaluate_{word}' and e['status'] == 'completed', 'original evaluation position')
    native_receipt = f['modules'][-1]['output']
    native = teacher.load_array(directory,native_receipt); shifted = teacher.load_array(directory,e['latents'])
    require(native.dtype == shifted.dtype == np.float32 and native.shape == shifted.shape == (3,192) and
        np.array_equal(native[-1],shifted[-1]), 'endpoint to endpoint only, full native3 is not shifted history3')
    b = row['blind_input']; goal = np.asarray(b['goal'],np.float64); q = np.asarray(b['q'],np.float64)
    require(goal.shape == (192,) and np.array_equal(q,np.eye(192)), 'original fixed goal and identity Q')
    cost = float(np.mean((q@(native[-1].astype(np.float64)-goal))**2))
    require(cost == e['cost'] == plan['model_Q'] and
        digest(b['context']) == plan['teacher_context_sha256'] and e['raw_commands'] == plan['suffix_commands'],
        'independent saved endpoint Q and same context/raw commands')
    witness = dict(source_report_sha256=REPORT_SHA,source_receipt_sha256=RECEIPT_SHA,
        anchor_index=row['index'], library_index=word, native_prediction=deepcopy(native_receipt),
        shifted_context=deepcopy(e['latents']), endpoint_sha256=hashlib.sha256(native[-1].tobytes()).hexdigest(),
        goal_sha256=hashlib.sha256(np.asarray(b['goal'],np.float32).tobytes()).hexdigest(), Q=cost,
        semantics='last row native prediction == last row shifted context; first two rows not compared')
    return dict(original_plan=deepcopy(plan),prediction_witness=witness,prediction_digest=digest(witness),
        full10_raw_bits=raw(plan['commands'],10).tobytes().hex(), suffix_raw_bits=raw(plan['suffix_commands'],5).tobytes().hex())


def build():
    r = teacher.read(REPORT,REPORT_SHA,CAP); receipt = teacher.read(RECEIPT,RECEIPT_SHA,CAP)
    require(receipt == teacher.consume(REPORT.parent,REPORT_SHA), 'original full native saved consumer, no physics consumer or model execution')
    require(len(r['rows']) == 8 and len(receipt['plans']) == 40, 'full source8x5')
    candidates = [make_candidate(p,r['rows'][p['index']//5],REPORT.parent) for p in receipt['plans']]
    p = teacher.read(r['prepared'],r['prepare_sha256'])
    bindings = {**p['bindings'], str(REPORT):REPORT_SHA, str(RECEIPT):RECEIPT_SHA, r['prepared']:r['prepare_sha256']}
    for name,h in r['artifact_sha256'].items():
        bindings[str(teacher.flat(REPORT.parent,name))] = h
    own = [Path(__file__).resolve(),ROOT/f'tests/test_{NAME}.py',ROOT/f'innerdocs/{NAME}.md']
    for f in own:
        bindings[str(f)] = sha(f)
    for path,h in bindings.items():
        require(sha(path) == h, 'complete original source/model/prepare/array closure')
    return dict(schema=NAME+'.selector', status='exploratory_model_selection_witness_not_physical_release',
        bindings=bindings, source_files={str(f):bindings[str(f)] for f in own},
        original_resources=deepcopy(receipt['resources']), physical_labels_consumed_by_selector=False,
        P1_admission=False, fresh_confirmation=False, legacy_policy_used_selector=False, **select40(candidates))


def write(path, value):
    p = Path(path).absolute(); data = canonical(value)
    require(p.resolve() == p and p.parent == ROOT/'results' and p.name.startswith(NAME+'_') and p.suffix == '.json' and
        len(data) <= CAP, 'exclusive bounded own-stem new file')
    with p.open('xb') as stream:
        require(stream.write(data) == len(data), 'whole selector write')
        stream.flush(); os.fsync(stream.fileno())
    teacher.sync_dir(p.parent)
    return dict(output=str(p),selector_sha256=sha(p))


def verify(path, expected, rebuilt):
    require(teacher.read(path,expected,CAP) == rebuilt, 'whole selector independently rebuilt, not digest-only verification')
    return dict(verified=True,selector_sha256=expected)


def consume(path, selector_sha):
    value = build(); verify(path,selector_sha,value)
    return dict(schema=NAME+'.consume', selector_sha256=selector_sha, fixed_candidates=40, fixed_selections=8,
        selected_plans=deepcopy(value['selections']), physical_release=False, P1_admission=False,
        legacy_policy_used_selector=False, fresh_confirmation=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('analyze','verify','consume'))
    parser.add_argument('--output'); parser.add_argument('--selector'); parser.add_argument('--selector-sha')
    args = parser.parse_args()
    if args.command == 'consume':
        out = consume(args.selector,args.selector_sha)
    else:
        value = build()
        out = write(args.output,value) if args.command == 'analyze' else verify(args.selector,args.selector_sha,value)
    print(canonical(out).decode())


if __name__ == '__main__':
    main()
