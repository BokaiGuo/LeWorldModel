"""Thin gradient-MPC selection adapter to the sealed generic physical executor.

No new model/optimizer/physics implementation. MAIN alone prepares/executes.
Native source certificates are evaluator-only inside generic.load_source, never
read by the selection builder or returned to the compiler.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import os
from pathlib import Path
import shutil
from experiments.spectral_reachability import p1_constructive_gradient_mpc_cuda_20261027_v1 as compiler
from experiments.spectral_reachability import p1_constructive_plan_physical_20261027_v1 as physical

NAME='p1_constructive_gradient_mpc_physical_20261027_v1'
ROOT=compiler.ROOT
REPORT=ROOT/'results/p1_constructive_gradient_mpc_cuda_20261027_v1_cuda_main01/report.json'
REPORT_SHA='8eadbc6068ee4949f280892442eda45298934ece87dd7ae38a44b4f93d78d728'
PREPARE_SHA='83371ef03c1b18727d8632b6a69085f04638189d7991ecd7831fa3affb62ca8f'
COMPILER_SHA='2c874aa56cc0b7727dbc20fbcbe60c3c0265bd368a6f5e1d050301f095be5be6'
PHYSICAL_SHA='9dea58dfc6418dd01bf9ab28de6d712189ca1154a5cf29b165efa4b8ef65ea96'
SOURCEPACK=ROOT/'results/p1_constructive_reachable_source_20261027_v1_main01/sourcepack.json'
SOURCEPACK_SHA='9ad3a414fd2148258524d4a7e9d4f070f59532ec2e5943cfa61f9145b3a41bd9'
SOURCE_PREPARED=ROOT/'results/p1_constructive_reachable_source_20261027_v1_prepare01.json'
SOURCE_PREPARE_SHA='98ffb806018a270f5ef27ffa76be929393a74355c213bc274bf158c72d80b086'
ROOT_SHAS=('f7aec52327fa6a6c8fb444919ba426a467ec2382b57da6d8a4f3701f82c65962',
           'f47e9958781f7eac4c810f9dfe24232b635f41d5b4c56423d8c2623f9fd5a328')
BLIND_SHAS=('d12ef7b66b69e53444ae6a4bde51c3830bacd6ad449d79845030ea70e2908da3',
            '9c0f6451de986bbb48c8251a8cfaab2b3fed60ce15698a9c6454c6206e228957')
TASK_IDS=('ac82e87557962f3db47b39634410362fddc455cd85c50ca3e8f923fc29264ee8',
          '05ad3266de1b02051c132bae16d24e2ab41b0938d979bf363f75f0937ce9cdd7')
ARM='gradient_MPC'
WALL=600
require,sha,read=compiler.require,compiler.sha,compiler.read


def files():
    return [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]


def selection_from(consumed):
    """Pure mapping, not candidate selection; never use costs or physical states."""
    require(consumed['schema']==compiler.NAME+'.consume' and consumed['status']=='own_gradient_model_commands_verified_not_admitted' and
        consumed['new_physics']==0 and consumed['P1_admission'] is False,'only own complete model consume receipt')
    rows=consumed['rows'];require(len(rows)==2,'all two gradient branches, even equal actions')
    plans=[]
    for s,row in enumerate(rows):
        require(type(row['source_slot']) is int and row['source_slot']==s and row['arm']==ARM and
            row['schema']==compiler.NAME+'.model_plan' and row['task_id']==TASK_IDS[s] and
            row['blind_sha256']==BLIND_SHAS[s] and row['whole_run_report_sha256']==REPORT_SHA and
            row['task_export']==compiler.EXPORTS[s] and row['task_sha256']==compiler.EXPORT_SHAS[s] and
            row['with_target'] is False and row['physical_execution'] is False and
            row['selection_uses_physics'] is False and row['P1_admission'] is False,'own approved source/task/arm/no-target selection')
        physical.commands(row['commands'])  # reject, never clip/round/pad
        body={k:v for k,v in row.items() if k not in ('selection_sha256','whole_run_report_sha256')}
        require(compiler.core.digest(body)==row['selection_sha256'],'own model-plan selection seal')
        plan=dict(root_slot=s,arm=ARM,commands=deepcopy(row['commands']),compiler_report=str(REPORT),compiler_report_SHA=REPORT_SHA,
            compiler_schema=compiler.NAME+'.report',compiler_status='gradient_mpc_model_plans_sealed_not_admitted',
            commands_pointer=f'/rows/{s}/commands',source_slot_pointer=f'/rows/{s}/source_slot',
            blind_sha_pointer=f'/rows/{s}/blind_sha256',arm_pointer=f'/rows/{s}/arm')
        plan['selection_digest']=physical.source.digest(plan);plans.append(plan)
    return dict(schema=physical.NAME+'.selection',sourcepack=str(SOURCEPACK),sourcepack_sha256=SOURCEPACK_SHA,
        source_prepared=str(SOURCE_PREPARED),source_prepare_sha256=SOURCE_PREPARE_SHA,
        roots=[dict(root_slot=s,root_sha256=h) for s,h in enumerate(ROOT_SHAS)],arms=[ARM],plans=plans,
        selection_fixed_before_physics=True,physical_outcome_selection=False,wall_seconds=WALL)


def build():
    require(sha(compiler.__file__)==COMPILER_SHA and sha(physical.__file__)==PHYSICAL_SHA,'frozen compiler and generic executor sources')
    require(sha(REPORT)==REPORT_SHA,'MAIN external actual gradient report SHA')
    report=read(REPORT)
    require(report['prepare_sha256']==PREPARE_SHA,'MAIN frozen gradient prepare SHA')
    # The sole command authority is the already reviewed own consume entry.
    # No direct selected/candidate indexing or alternative planner fallback.
    consumed=compiler.consume(REPORT.parent,REPORT_SHA)
    manifest=selection_from(consumed)
    reference=dict(references=[dict(root_sha256=ROOT_SHAS[s],blind_sha256=BLIND_SHAS[s]) for s in (0,1)])
    physical.selection(manifest,reference)  # raw report JSON-pointer binding, no sourcepack/certificate access
    prepared=compiler.verify(report['prepared'],PREPARE_SHA)
    bindings=deepcopy(prepared['bindings_sha256'])
    bindings.update({str(REPORT):REPORT_SHA,str(Path(report['prepared']).absolute()):PREPARE_SHA,
        str(Path(physical.__file__).resolve()):PHYSICAL_SHA})
    for f in files():bindings[str(f)]=sha(f)
    for path,h in bindings.items():require(sha(path)==h,'adapter/source/model/export seal changed')
    return manifest,bindings


def write_new(path,value):
    path=Path(path).absolute();raw=physical.raw(value)
    require(not path.exists() and len(raw)<256*1024 and shutil.disk_usage(path.parent).free>=physical.FLOOR+len(raw),'NEW small metadata +200MiB reserve')
    with path.open('xb') as f:
        require(f.write(raw)==len(raw),'short metadata write');f.flush();os.fsync(f.fileno())
    compiler.io.fsync_directory(path.parent)
    return dict(output=str(path),sha256=sha(path),bytes=len(raw))


def selection(output):
    manifest,_=build();return write_new(output,manifest)


def material(manifest_path,manifest_sha):
    compiler.core.hash_value(manifest_sha)
    require(sha(manifest_path)==manifest_sha,'external new selection SHA')
    wanted,bindings=build()
    require(read(manifest_path)==wanted,'exact own gradient selection only, no overridden source/plans/wall')
    # Evaluator-only source reconstruction is delegated unchanged. It may read
    # approved observed state/goal certificate fields, NEVER witness commands.
    p=physical.material(manifest_path,manifest_sha)
    require(p['sourcepack_sha256']==SOURCEPACK_SHA and p['source_slots']==wanted['roots'] and
        p['plans']==wanted['plans'] and p['contract']==physical.contract(2,WALL),'original generic full10 safety/budget/source contract')
    for path,h in bindings.items():
        require(path not in p['bindings'] or p['bindings'][path]==h,'non-conflicting inherited source seal')
    p=deepcopy(p);p['bindings'].update(bindings)
    p.update(schema=NAME+'.prepare',adapter=dict(source_sha256=sha(__file__),compiler_report=str(REPORT),
        compiler_report_sha256=REPORT_SHA,compiler_prepare_sha256=PREPARE_SHA,generic_executor_sha256=PHYSICAL_SHA,
        command_authority='frozen gradient own consume only; no physical outcome selection'))
    for path,h in p['bindings'].items():require(sha(path)==h,'final prepare closure TOCTOU')
    return p


def prepare(manifest,manifest_sha,output):
    return write_new(output,material(manifest,manifest_sha))


def verify(prepared,expected):
    compiler.core.hash_value(expected)
    require(sha(prepared)==expected,'external adapter physical prepare SHA')
    p=read(prepared)
    require(p==material(p['manifest'],p['manifest_sha256']),'entire own adapter/generic/source/command reconstruction')
    return p


def run(prepared,expected,output,factory,*,authorization):
    require(authorization==physical.AUTH,'MAIN-only generic physics authorization')
    p=verify(prepared,expected);manifest=read(p['manifest'])
    data=physical.load_source(manifest['sourcepack'],manifest['sourcepack_sha256'],
        manifest['source_prepared'],manifest['source_prepare_sha256'])
    store=physical.Store(output)
    store.put('execution_binding.json',dict(prepared=str(Path(prepared).resolve()),prepare_sha256=expected))
    # Original generic executor, same guards/caps/partial/journal/report schema.
    # No duplicate physics loop, monkey patch, or old endpoint substitution.
    return physical.execute(p,data,store,factory,authorization=authorization)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('selection','prepare','verify','run'))
    for key in ('manifest','manifest-sha','prepared','prepare-sha','output'):p.add_argument('--'+key)
    p.add_argument('--authorize-physics',action='store_true');a=p.parse_args()
    if a.command=='selection':result=selection(a.output)
    elif a.command=='prepare':result=prepare(a.manifest,a.manifest_sha,a.output)
    elif a.command=='verify':result=dict(status='verified_no_execution',plans=len(verify(a.prepared,a.prepare_sha)['plans']))
    else:
        require(a.authorize_physics,'MAIN CLI physical authorization')
        def factory():
            import gymnasium as gym
            import stable_worldmodel  # noqa: F401
            return gym.make('swm/PushT-v1',render_mode='rgb_array',resolution=224,with_target=False,render_action=False)
        result=run(a.prepared,a.prepare_sha,a.output,factory,authorization=physical.AUTH)
    print(physical.source.canonical({k:v for k,v in result.items() if k!='rows'}))
    if a.command=='run' and result['status']!='physical_calibration_recorded':raise SystemExit(2)


if __name__=='__main__':main()
