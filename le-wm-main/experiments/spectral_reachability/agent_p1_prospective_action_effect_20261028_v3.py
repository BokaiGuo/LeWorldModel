"""Saved-only strict wrapper for frozen v2 output; deliberately no run/prepare.

Reuses v2 consumer code/native validators with one exact, independently validated
terminal-event envelope projection in a local function namespace (no mutation).
Additional checks cover the UNPROJECTED eventset and pixels -> preprocessing.
"""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
from types import FunctionType
from experiments.spectral_reachability import agent_p1_prospective_action_effect_20261028_v2 as v2
from experiments.spectral_reachability import agent_p1_teacher_library40_encode_20261028_v2 as pixels

NAME='agent_p1_prospective_action_effect_20261028_v3'
ROOT=v2.ROOT
require=v2.require
FROZEN={
    'experiments/spectral_reachability/agent_p1_prospective_action_effect_20261028_v2.py':
        '72d1d12cd03e33072795b42ae35b5c1e70ade8040f3e40ce3b70cda807baae40',
    'tests/test_agent_p1_prospective_action_effect_20261028_v2.py':
        'bd460557fce2f4d56a796fd33075ff1083ccd1d3deba86fcd6b7cbd5f376ee73',
    'innerdocs/agent_p1_prospective_action_effect_20261028_v2.md':
        'c8da14999a4c473cd8b63a5b88b59f14548e2969aa4b175f9e52035498ef3899',
    'experiments/spectral_reachability/agent_p1_teacher_library40_encode_20261028_v2.py':
        'f7e3abb842cc185c77fe217922cd83ce709f4a6bf8fce0d9337c87bf4f66f75b',
}
# Reachable emitters only: v2 Meter/NativeTrace/CudaModel/orchestrator;
# phys.dispatch/observe/episode, helper.observe, base.capture, paired.place/bind_goal.
# In particular, other producer APIs' operation_completed/returned_transition/
# probe_returned etc. are NOT released event kinds for this runtime.
ALLOWED_EVENTS=frozenset({
    'run_started','run_finished',
    'operation_attempted','operation_returned','operation_failed',
    'load_intent','consumed_buffer_frozen',
    'tensor_module_attempt','tensor_module_completed',
    'image_attempt','image_return','forward_attempt','forward_return',
    'placement_recipe','goal_attached','fresh_observation','invalid_observation',
    'rgb_guard_observed','reference_frame_verified','raw_return','reference_raw_verified',
    'checkpoint_recorded','teacher_history_verified',
    'replay20_ready','model_selection_sealed','extension_started',
    'alternate_encode_failed','terminal25_encoded','branch_error',
})


def authority():
    for name,expected in FROZEN.items():
        path=ROOT/name
        require(path.is_file() and not path.is_symlink() and v2.sha(path)==expected,'frozen v2/original preprocessing authority')
    require(Path(v2.__file__).resolve()==ROOT/next(iter(FROZEN)) and
        Path(pixels.__file__).resolve()==ROOT/'experiments/spectral_reachability/agent_p1_teacher_library40_encode_20261028_v2.py',
        'actual imported frozen modules')
    return {str(ROOT/name):h for name,h in FROZEN.items()}


def validate_eventset(events):
    counts=Counter()
    for e in events:
        require(isinstance(e,dict) and type(e.get('kind')) is str and e['kind'] in ALLOWED_EVENTS,
                'unreleased event in full unprojected journal')
        counts[e['kind']]+=1
    return dict(sorted(counts.items()))


def validate_preprocessing(output,images):
    checked=missing=0
    for image in images:
        require(type(image.get('completed')) is bool,'literal image completion')
        if 'preprocessed' not in image:
            require(not image['completed'],'completed image must contain original preprocessing')
            missing+=1;continue
        rgb=v2.load_array(output,image['pixels'])
        expected=pixels.preprocess_pixels(rgb)  # Frozen original224 NumPy helper, no torch/neural execution.
        actual=v2.load_array(output,image['preprocessed'])
        require(v2.exact_array(actual,expected),'pixels -> released preprocessing exact dtype/shape/bytes')
        checked+=1
    return dict(checked=checked,partial_without_preprocessed=missing)


ENVELOPE=frozenset({'index','previous_sha256','event_sha256'})
OBSERVATION=frozenset({'kind','anchor_index','word','job_digest','pre','post','receipt'})


def terminal_projection(report,events):
    """Allow only the diagnosed v2 Store.emit envelope discrepancy.

    Each mapping is keyed by the EXACT canonical terminal journal triple, after
    validating its signed envelope and every payload byte against its own row.
    No generic recursive stripping; receipt/state/snapshot/action/order stay intact.
    """
    mappings={}
    for row in report.get('rows',[]):
        if row['status'] not in ('terminal25_encoded','terminal25_original_encoded'):continue
        frames=[e for e in events if e['kind']=='fresh_observation' and e.get('job_digest')==row['job_digest']]
        require(len(frames)>=3 and len(row['terminal_triplet'])==3,'terminal actual triplet available')
        full=frames[-3:];body=row['terminal_triplet'];projected=[]
        for e,b in zip(full,body,strict=True):
            require(set(e)==OBSERVATION|ENVELOPE and set(b)==OBSERVATION,'exact terminal event/envelope fields only')
            require(type(e['index']) is int and e['index']>=0 and
                all(type(e[k]) is str and len(e[k])==64 and all(c in '0123456789abcdef' for c in e[k])
                    for k in ('previous_sha256','event_sha256')) and
                e['event_sha256']==v2.digest({k:v for k,v in e.items() if k!='event_sha256'}),'valid terminal journal envelope, not ignored metadata')
            require(all(e[k]==b[k]==row[k] for k in ('anchor_index','word','job_digest')) and
                e['receipt']['phase']=='suffix' and e['receipt']['step']==10,'exact terminal row/time identity')
            projected.append({k:v for k,v in e.items() if k not in ENVELOPE})
        require([e['index'] for e in full]==sorted({e['index'] for e in full}),'terminal envelope strict chronological order')
        require(v2.canonical(projected)==v2.canonical(body),'exact terminal payload; no state/raw/renderer tolerance')
        v2.validate_triplet(body)
        key=v2.canonical(full);require(key not in mappings,'unique terminal triple')
        mappings[key]=(v2.canonical(body),row['index'])
    return mappings


def consume_with_projection(output,report_sha,mappings):
    """Reuse pinned v2 code without module/global monkeypatch or copied runner.

    Only exact validated terminal canonical byte strings have an alternative
    representation. All other canonical values and all helper globals retain
    original semantics. The original journal/hashchain itself is never projected.
    """
    usage=Counter()
    def canonical(value):
        raw=v2.canonical(value)
        if raw not in mappings:return raw
        projected,row_index=mappings[raw];usage[row_index]+=1;return projected
    original=v2.consume
    namespace=dict(original.__globals__,canonical=canonical)
    local=FunctionType(original.__code__,namespace,original.__name__,original.__defaults__,original.__closure__)
    local.__kwdefaults__=original.__kwdefaults__
    result=local(output,report_sha)
    require(dict(usage)=={row_index:1 for _,row_index in mappings.values()},'exactly one validated terminal join projection per known row')
    return result,dict(usage)


def consume(output,report_sha):
    bindings=authority();output=Path(output).absolute()
    r=v2.teacher.read(output/'report.json',report_sha,v2.phys.REPORT_CAP)
    events=v2.read_events(output/'journal.jsonl')
    require(v2.sha(output/'journal.jsonl')==r['artifacts']['journal.jsonl'],'same v2-bound full journal')
    event_counts=validate_eventset(events)  # Never validate only native20_projection(filtered).
    mappings=terminal_projection(r,events)
    # Mandatory pinned v2 code and all native/source/array validators, with ONLY
    # the exact prevalidated envelope join adapted. Original v2 failure retained.
    base,usage=consume_with_projection(output,report_sha,mappings)
    preprocessing=validate_preprocessing(output,r['images'])
    # Preserve external v2 binding after the extra reads; no writes to producer directory.
    require(v2.sha(output/'report.json')==report_sha,'v2 report unchanged during stricter consume')
    for name,h in r['artifacts'].items():
        path=output/name
        require(Path(name).name==name and path.is_file() and not path.is_symlink() and v2.sha(path)==h,
                'v2 artifact unchanged during stricter consume')
    require(base['status'] in ('saved_adapter_audited_not_P1_admission','partial_saved_audit'),'known frozen v2 result')
    own=[Path(__file__),ROOT/'tests'/f'test_{NAME}.py',ROOT/'innerdocs'/f'{NAME}.md']
    return dict(schema=NAME+'.consume',report_sha256=report_sha,
        status='strict_saved_adapter_audited_not_P1_admission' if base['status']=='saved_adapter_audited_not_P1_admission' else 'partial_saved_audit',
        v2_consume=base,full_event_counts=event_counts,preprocessing=preprocessing,
        v2_consume_execution='pinned code with isolated exact terminal-envelope canonical adapter; original v2 consume is NOT claimed to pass',
        terminal_join_compatibility=dict(envelope_fields=sorted(ENVELOPE),validated_rows=len(mappings),
            substitutions=usage,payload_tolerance=0,original_v2_failure='line630 enriched journal event vs Store.observations body'),
        frozen_authority=bindings,consumer_files={str(p.resolve()):v2.sha(p) for p in own},
        new_neural_calls=0,new_physics_calls=0,P1_admission=False,
        interpretation='saved audit hardening only; not an explanation of original P1 reversals or a gate change')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('consume',))
    parser.add_argument('--output',required=True);parser.add_argument('--report-sha',required=True)
    parser.add_argument('--receipt')
    args=parser.parse_args();result=consume(args.output,args.report_sha)
    if args.receipt:
        path=Path(args.receipt).absolute()
        require(path.name.startswith(NAME+'_') and path.parent!=Path(args.output).absolute(),'exclusive new v3 receipt outside frozen v2 output')
        # Existing v2 writer accepts only its own prefix: do not monkeypatch it.
        raw=v2.canonical(result);require(path.resolve()==path and path.suffix=='.json' and len(raw)<=v2.phys.PREP_CAP,'bounded own receipt')
        import os
        with path.open('xb') as f:
            require(f.write(raw)==len(raw),'complete receipt write');f.flush();os.fsync(f.fileno())
        v2.phys.adapter.paired.fsync_dir(path.parent)
        result=dict(receipt=str(path),sha256=v2.sha(path),status=result['status'])
    print(v2.canonical(result).decode())


if __name__=='__main__':main()
