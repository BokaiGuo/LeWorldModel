"""Moving RGB-feedback execution inputs for MAIN, reusing the sealed driver.

No neural/physics/network work in prepare/verify. NOT a multiround P4 driver:
the existing moving runner has one feedback boundary (5), three O/I/R modes.
The known-root dispatch seam invokes its REAL run(), only on explicit authority.
"""
from __future__ import annotations
import argparse
import ast
import importlib
import os
from pathlib import Path
from experiments.spectral_reachability import agent_subgoal_gap_contract_20261028_v1 as c
import hashlib

NAME='agent_p4_runtime_assets_20261028_v1'
ROOT=Path(__file__).resolve().parents[2]
NATIVE='p1_moving_live_feedback_physical_20261027_v1'
DRIVER=ROOT/'experiments/spectral_reachability'/(NATIVE+'.py')
DRIVER_SHA='6742a1b35102e433d7772e4a208b4f97b37b76f815b991cfc0e472763754ace9'
PREPARED=ROOT/'results'/(NATIVE+'_prepare_main01.json')
PREPARED_SHA='bc3fbc91d7a19ae256b24db22ad05b2526fcbe454e3b55d904136eba2f8fdff5'
REPORT=ROOT/'results'/(NATIVE+'_main01/report.json')
REPORT_SHA='990843e8bba15955db5163b9b5d77c1f8eecaa1870d3745b49d46cc22ca895e3'
AUTH='MAIN_MOVING_LIVE_FEEDBACK_CUDA_AND_PHYSICS_ONLY'
LAYERS=[(slot,arm) for slot in (0,1) for arm in ('line_search_GN','full10_conditional_D2P')]
CALLBACKS={'encode_observation':'Handler.encode_observation','compile_suffix':'Handler.compile_suffix',
    'verify_suffix':'Handler.verify_suffix_seal','native_factory':'native_factory',
    'native_metrics':'native_transition','dispatch':'run','episode':'episode',
    'resource_meter':'Meter','durability':'Store','single_use_dispatch':'DispatchGate'}
require=c.require


def sha(path):
    p=Path(path);require(p.is_file() and not p.is_symlink(),'regular local input: '+str(p))
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def sources():
    return {str(p):sha(p) for p in (Path(__file__).resolve(),ROOT/'tests'/(NAME+'_tests.py'),ROOT/'innerdocs'/(NAME+'.md'))}


def source_callbacks(path):
    """Finite source-identity inventory, NOT an algorithm/code-audit verdict."""
    tree=ast.parse(Path(path).read_text());symbols={}
    for node in tree.body:
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)):
            symbols[node.name]=node.lineno
            if isinstance(node,ast.ClassDef):
                for child in node.body:
                    if isinstance(child,ast.FunctionDef):symbols[node.name+'.'+child.name]=child.lineno
    require(set(CALLBACKS.values())<=symbols.keys(),'existing concrete callback entries required')
    return {k:dict(path=str(Path(path).absolute()),symbol=v,line=symbols[v]) for k,v in CALLBACKS.items()}


def inspect_saved(p,report):
    require(p['schema']==NATIVE+'.prepare','original prepare schema')
    contract=p['contract']
    require([(x['source_slot'],x['arm']) for x in p['layers']]==LAYERS and
        contract['layers']==[list(x) for x in LAYERS] and contract['modes']==['O','I','R']
        and contract['episodes']==12 and contract['warmup']==10 and contract['horizon']==10
        and contract['feedback_step']==5 and contract['with_target'] is False,'fixed moving O/I/R contract, not multiround')
    require(contract['P1_admission'] is contract['P4_admission'] is contract['independent_confirmation'] is False,
        'no scientific admission inherited')
    require(report['schema']==NATIVE+'.report' and report['status']=='live_feedback_calibration_recorded'
        and report['fixed_denominator']==12 and report['independent_tasks']==2 and not report['errors'],
        'completed sealed calibration evidence')
    require(report['contract']==contract and len(report['rows'])==12 and
        [(x['source_slot'],x['prefix_arm'],x['mode']) for x in report['rows']]==
        [(s,a,m) for s,a in LAYERS for m in ('O','I','R')],'all historical slots retained')
    return dict(schema=NAME+'.execution_inputs',native_prepared=str(PREPARED),native_prepare_sha256=PREPARED_SHA,
        native_module='experiments.spectral_reachability.'+NATIVE,
        native_entry='run(prepared, expected, output, authorization=AUTH)',
        callbacks=source_callbacks(DRIVER),contract=contract,
        budget=dict(planned_upper=dict(physical=contract['physical_caps'],neural=contract['neural_caps']),
            actual_new_work=None,flops=None,device_time_ns=None,inherited_compute=None,
            total_compute_matched=False,attempt_return_accounting='original Meter + Sink event mapping; no refunds',
            wall_seconds=contract['wall_seconds'],output_cap_bytes=contract['output_cap_bytes'],
            disk_floor_bytes=contract['disk_floor_bytes'],final_reserve_bytes=contract['final_reserve_bytes'],
            cuda_allocated_cap=contract['cuda_allocated_cap'],cuda_reserved_cap=contract['cuda_reserved_cap']),
        native_metrics=dict(producer='native_transition -> body.eval_state + kernel.metric',
            fields=['native_distance','native_angle_term','negative_native_angle','object_pose',
                'object_position_error','object_success','wrapped_allxy_success',
                'wrapped_object_angle_error','native_success'],
            success_positions='goal[:4] includes agent AND object',
            native_goal='original task goal7, not renderer outline',missing=None),
        denominator=dict(independent_tasks=2,layers=4,episodes=12,mode_slots=4),
        known_root_engineering_only=True,execution_authorized=False,
        formal_efficacy_blockers=['independent P1 admission missing','fresh prospective root certification missing',
            'moving multi-boundary driver not present in this binding (only step5)',
            'full-resource equivalence not certified'],
        P1_admission=False,P4_admission=False,scientific_pass=False,
        status='single_feedback_inputs_bound_NOT_P4_ready')


def prepare():
    require(sha(DRIVER)==DRIVER_SHA,'frozen moving driver drift')
    p=c.bound(PREPARED,PREPARED_SHA);report=c.bound(REPORT,REPORT_SHA)
    bindings=dict(p['bindings'])|p['modelbinding']['source_closure']|p['modelbinding']['model_files']
    bindings.update({str(PREPARED):PREPARED_SHA,str(REPORT):REPORT_SHA,str(DRIVER):DRIVER_SHA})
    for path,h in bindings.items():c.sha(h);require(sha(path)==h,'original dependency drift: '+path)
    require(bindings[p['manifest']]==p['manifest_sha256'] and bindings[p['backend_source']]==p['backend_sha256'],
        'real original selection and suffix binding')
    inputs=inspect_saved(p,report)
    return dict(inputs,bindings=bindings,sources=sources(),
        native_modelbinding=p['modelbinding'],historical_report_sha256=REPORT_SHA,
        historical_outcomes_scope='identity/calibration provenance only; never a planner request')


def verify(value,expected):
    c.sha(expected);require(c.digest(value)==expected,'external canonical execution-input SHA')
    require(prepare()==value,'execution inputs/source/assets drift');return value


def dispatch_known(value,expected,output,authority_path,authority_sha):
    """MAIN-only actual dispatch. Worker tests replace module with RAM callback.

    No arbitrary module/entry allowed. Native verify and native run both stay
    unchanged, including frozen-model/CUDA/Store/Meter/12-slot failure handling.
    Confirmatory/multiround requests are unconditionally unsupported here.
    """
    verify(value,expected)
    output=Path(output).absolute()
    require(output.parent==ROOT/'results' and output.name.startswith(NAME) and not output.exists(),
        'NEW exclusive results path')
    a=c.bound(authority_path,authority_sha)
    require(a==dict(schema=NAME+'.MAIN_authority',reviewer='MAIN',input_sha256=expected,
        output=str(output),scope='known_root_single_feedback_engineering',
        host_cuda_and_physics_authorized=True,confirmatory=False)
        and a['host_cuda_and_physics_authorized'] is True and a['confirmatory'] is False,
        'explicit MAIN known-root CUDA+physics authority; no confirmatory release')
    # Exclusive launch intent prevents retry after an import/native-verify/startup
    # failure. It is never represented as actual GPU/physics completion.
    launch=output.with_name(output.name+'.launch.json')
    payload=c.encode(dict(schema=NAME+'.launch_intent',input_sha256=expected,
        authority_sha256=authority_sha,output=str(output),actual_computation=None,
        root_denominator=2,episode_denominator=12,P4_admission=False))
    require(len(payload)<65536,'small launch record')
    with launch.open('xb') as f:
        require(f.write(payload)==len(payload),'complete launch intent write')
        f.flush();os.fsync(f.fileno())
    fd=os.open(launch.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    native=importlib.import_module('experiments.spectral_reachability.'+NATIVE)
    require(Path(native.__file__).resolve()==DRIVER and sha(native.__file__)==DRIVER_SHA and native.AUTH==AUTH,
        'actual imported sealed driver')
    native.verify(value['native_prepared'],value['native_prepare_sha256'])
    return native.run(value['native_prepared'],value['native_prepare_sha256'],str(output),authorization=AUTH)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','verify','run-known'))
    for key in ('input','sha256','output','authority','authority-sha256'):p.add_argument('--'+key)
    a=p.parse_args(argv)
    if a.command=='prepare':out=prepare()
    else:
        value=c.bound(a.input,a.sha256);expected=c.digest(value)
        if a.command=='verify':out=verify(value,expected)
        else:out=dispatch_known(value,expected,a.output,a.authority,a.authority_sha256)
    print(c.encode(out).decode());return 0


if __name__=='__main__':raise SystemExit(main())
