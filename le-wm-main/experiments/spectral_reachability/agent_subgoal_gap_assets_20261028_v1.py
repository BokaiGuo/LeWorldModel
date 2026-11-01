"""Saved-only E1 asset packaging. No execution adapter, neural or physics imports.

Uses MAIN's authenticated saved-consumer receipt, not a repeat scientific audit.
Always blocked: legacy configuration is evidence, not a unified E1 controller.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from experiments.spectral_reachability import agent_subgoal_gap_runner_20261028_v1 as r

NAME='agent_subgoal_gap_assets_20261028_v1'
ROOT=r.ROOT
RUNNER_SHA='0a301db423416d2930264b0fd3f80c2356806c93791c128faddd6ee02ddb2154'
RECEIPT=ROOT/'results/agent_p2_guard_saved_consume_20261028_v1_MAIN02.json'
RECEIPT_SHA='c7d183f5b18d1bfacd81f855025efdbbb403f97e5442650970080766db17088b'
REPORT=ROOT/'results/agent_p2_runtime_guard_20261028_v1_known_MAIN01/report.json'
PROTOCOL=ROOT/'results/p2_composition_predict_frontier_cuda_20261027_v2_prepared01/protocol.json'
NEXT=ROOT/'results/agent_p2_resume_20261027_next_prepared01.json'
LEARNED=ROOT/'results/p1_hilewm_bounded_object_fetch_20261027_v2_probe_main01/report.json'
LEARNED_SHA='669bf41043dfae01a9bca183fe230d61baad29234c5cc73038c80409844e0b90'
c=r.c
require=c.require


def ref(path):
    path=Path(path).absolute()
    require(path.is_file() and not path.is_symlink(),'regular local asset')
    return dict(path=str(path),sha256=r.file_sha(path))


def authenticate(receipt,receipt_sha):
    """Hash only. Does NOT rerun old array scoring or import native executors."""
    require(r.file_sha(r.__file__)==RUNNER_SHA,'frozen runner drift')
    saved=c.bound(receipt,receipt_sha)
    require(saved['status']=='terminal_saved_evidence_verified_NOT_physical_release'
        and [saved[k] for k in ('root_denominator','arm_denominator','candidate_denominator')]==[4,8,480]
        and saved['physical_execution_authorized'] is saved['P1_admission'] is saved['P2_admission'] is False,
        'terminal complete known-root receipt, not physical release')
    bindings=dict(saved['bindings'])
    bindings.update(saved['consumer_sources'])
    bindings[str(Path(receipt).absolute())]=receipt_sha
    for path,h in bindings.items():
        c.sha(h);p=Path(path)
        require(p.is_absolute() and p.is_file() and not p.is_symlink() and r.file_sha(p)==h,
            'saved receipt binding drift: '+path)
    return saved,bindings


def assemble(protocol,report,rows,compiled,native_contract,*,bindings,put):
    """Pure metadata seam; put(name, JSON) returns its real persisted byte ref.

    No winner chosen: candidate index zero of the conditional arm for every root.
    Missing common identities are REAL written unknown descriptors, not future
    checkpoint hashes. Null assets keep runner dispatch closed.
    """
    cases=protocol['context']['cases']
    require(len(cases)==len(rows)==4 and report['status']=='known_root_completed'
        and report['error'] is None and report['model_unchanged'] is report['source_unchanged'] is True,
        'complete four-root source')
    require([x['root_index'] for x in cases]==list(range(4)),'original root order')
    require(len({x['root_sha256'] for x in cases})==4,'unique fixed roots')
    assets=dict.fromkeys(r.ASSETS,None);trials=[];inventory=[]
    # These are intentionally dispatch-disabled limits, NOT measured costs or
    # a proposed equal-compute scientific budget. No budget asset is fabricated.
    caps={phase:r.vector() for phase in r.PHASES}
    for cap in caps.values():cap['callback_calls']=1
    for slot,(case,row) in enumerate(zip(cases,rows,strict=True)):
        require(row['root_slot']==slot and row['root_sha256']==case['root_sha256']
            and row['complete'] is True and row['faults']==dict(primary=None,secondary=[]), 'complete ordered root row')
        results=row['results']
        require(len(results)==2 and [len(x['attempts']) for x in results]==[80,40]
            and all(x['complete'] is True and x['error'] is None for x in results),'preserve full source denominator')
        attempt=results[1]['attempts'][0]
        require(attempt['index']==0 and attempt['completed'] is True,'fixed index zero, no outcome selection')
        witness=attempt['witness'];prov=witness['receipt']['provenance'];root=case['root']
        require(root==witness['receipt']['root'],'actual witness root/history')
        normal=protocol['context']['normalization'];metric=compiled[case['root_sha256']]['scoring']['metric']
        # Metric identity excludes its self-digest in the original protocol.
        require(metric['sha256']==case['metric_sha256']==prov['metric_sha256'],'original Q identity')
        for key in ('model','encoder'):
            h=prov['checkpoint_sha256' if key=='model' else 'encoder_sha256']
            paths=[p for p,v in bindings.items() if v==h]
            require(paths,'actual checkpoint bytes bound by MAIN receipt')
            asset=dict(path=sorted(paths)[0],sha256=h)
            if assets[key] is not None:require(assets[key]==asset,'common frozen checkpoint')
            assets[key]=asset
        if slot==0:
            assets['normalizer']=put('normalizer',normal)
            assets['native_success_evaluator']=native_contract['source']
        require(c.digest(normal)==case['action_domain']['normalization_sha256'],'original mean/std identity')
        qref=put(f'q_{slot}',metric)
        goal=put(f'goal_{slot}',dict(schema=NAME+'.goal',goal_rgb_sha256=root['goal_rgb_sha256'],
            latent=compiled[case['root_sha256']]['goal_vectors'][0],native_goal7=None,
            native_identity_certified=False,renderer_goal_pose_is_task_goal=False))
        shared=dict(root=c.digest(root),history=c.digest(root['context']),task_goal=goal['sha256'])
        for key in r.COMMON:
            if assets[key] is not None:shared[key]=assets[key]['sha256']
            else:shared[key]=put('unknown_'+key,dict(schema=NAME+'.unknown_identity',asset=key,
                supported=False,reason='no unified E1 '+key+' asset'))['sha256']
        restore=dict(schema=NAME+'.restore_input',root=root,source_action_domain=case['action_domain'],
            goal=goal,native_contract=native_contract,native_start_snapshot=None,
            exact_restore_certified=False,original_full_context_preserved=True)
        inp=put(f'input_{slot}',dict(root=root,task_goal_sha256=shared['task_goal'],restore_binding=restore))
        query=put(f'compiled_{slot}',dict(schema=NAME+'.saved_compiled_query',source_slot=slot,
            candidate_index=0,witness=witness,source_actions=attempt['actions'],
            target=[e['output_context']['latent_history'][-1] for e in witness['edges']],
            reused_saved_evidence_only=True,new_generator_call=False))
        trials.append(dict(trial_id=case['root_sha256'],shared=shared,input=inp,
            restore_binding_sha256=c.digest(restore),proposal_inputs={c.ARMS[0]:None,c.ARMS[1]:None,c.ARMS[2]:query}))
        inventory.append(dict(root_slot=slot,root_id=case['root_sha256'],root_content_sha256=c.digest(root),
            context_sha256=shared['history'],goal=goal,source_Q_sha256=case['metric_sha256'],
            packaged_Q=qref,native_success=None,compiled_candidate_index=0))
    spec=dict(schema=r.NAME+'.input',latent_dim=192,checkpoints=[5,10],assets=assets,phase_caps=caps,trials=trials)
    return spec,dict(schema=NAME+'.inventory',roots=inventory,root_denominator=4,arm_denominator=12,
        source_candidate_denominator=480,source_arm_denominator=8,status='blocked',
        missing_assets=[k for k,v in assets.items() if v is None],
        missing_per_root=[c.ARMS[0],c.ARMS[1],'native_goal7','exact_native_start_snapshot'],
        source_native_contract=native_contract,goal_identity_scope='bound descriptor; NOT certified native task goal',
        cost_scope='disabled guard caps; actual new model/physics work absent; inherited cost unknown',
        inherited_compute=None,scientific_pass=False,physical_execution_authorized=False)


def build(output):
    saved,bindings=authenticate(RECEIPT,RECEIPT_SHA)
    def bound(path):
        path=str(Path(path).absolute());require(path in bindings,'not in MAIN sealed dependency closure: '+path)
        return c.bound(path,bindings[path])
    report=bound(REPORT);require(saved['report_sha256']==bindings[str(REPORT)],'actual report linkage')
    protocol=bound(PROTOCOL);next_plan=bound(NEXT)
    native_source=next_plan['plan']['native_contract']['source']
    require(bindings[native_source['path']]==native_source['sha256'],'native evaluator source seal')
    compilation=bound(protocol['context']['compilation'])
    compiled={x['root_sha256']:x['compiled'] for x in compilation['rows']}
    rows=[]
    for i,entry in enumerate(report['rows']):
        require(entry['root_slot']==i and entry['complete'] is True and entry['file']==f'root_{i}.json','source rows')
        path=REPORT.parent/entry['file'];require(bindings[str(path)]==entry['sha256'],'source row seal')
        rows.append(bound(path))
    learned=c.bound(LEARNED,LEARNED_SHA)
    require(learned['status']=='failed_preserved' and learned['unpickled'] is False,'failed learned fetch is not usable prior')
    bindings[str(LEARNED)]=LEARNED_SHA
    output=Path(output).absolute()
    require(output.name.startswith(NAME) and not output.exists(),'new exclusive asset directory')
    output.mkdir()
    generated={}
    def put(name,value):
        path=output/(name+'.json');data=c.encode(value)
        if path.exists():require(path.read_bytes()==data,'conflicting generated identity')
        else:
            with path.open('xb') as f:f.write(data)
        item=ref(path);generated[item['path']]=item['sha256'];return item
    # Failure leaves existing files intact; no rollback, retry or deletion.
    spec,inventory=assemble(protocol,report,rows,compiled,next_plan['plan']['native_contract'],bindings=bindings,put=put)
    # Frozen prepare preserves iteration order in its blocker LIST. Normalize
    # before the first prepare so stdout/file reload yields the same list.
    spec=c.parse(c.encode(spec))
    spec_ref=put('spec',spec);prepared=r.prepare(spec)
    require(prepared['status']=='blocked' and prepared['arm_denominator']==12,'must not grant metadata readiness')
    inventory.update(runner_blockers=prepared['blockers'],learned_probe=ref(LEARNED),
        learned_probe_status=learned['status'],spec=spec_ref,prepared=put('prepared',prepared),
        dependencies=bindings,sources={str(p):r.file_sha(p) for p in
            (Path(__file__).resolve(),ROOT/'tests'/(NAME+'_tests.py'),ROOT/'innerdocs'/(NAME+'.md'))})
    inventory['generated_files']=dict(generated)
    manifest=put('inventory',inventory)
    verify(manifest['path'],manifest['sha256'])
    return dict(status='blocked',inventory=manifest,spec=spec_ref,prepared=inventory['prepared'],
        root_denominator=4,arm_denominator=12,missing_assets=inventory['missing_assets'],
        physical_execution_authorized=False,scientific_pass=False)


def verify(path,sha):
    value=c.bound(path,sha)
    for p,h in (value['dependencies']|value['sources']|value['generated_files']).items():
        require(ref(p)['sha256']==h,'asset pack closure drift: '+p)
    prepared=c.bound(**dict(path=value['prepared']['path'],expected=value['prepared']['sha256']))
    spec=c.bound(value['spec']['path'],value['spec']['sha256'])
    require(prepared['spec']==spec and r.prepare(spec)==prepared,'runner exact prepare seam')
    r.verify(prepared,c.digest(prepared))
    for root in value['roots']:
        for key in ('goal','packaged_Q'):c.bound(root[key]['path'],root[key]['sha256'])
    require(value['status']==prepared['status']=='blocked','no engineering/scientific promotion')
    return value


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','verify'))
    p.add_argument('--output');p.add_argument('--input');p.add_argument('--sha256');a=p.parse_args(argv)
    if a.command=='prepare':require(a.output is not None,'new output path');result=build(a.output)
    else:require(a.input is not None and a.sha256 is not None,'external inventory byte SHA');result=verify(a.input,a.sha256)
    print(c.encode(result).decode());return 0


if __name__=='__main__':raise SystemExit(main())
