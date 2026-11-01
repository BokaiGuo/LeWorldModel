"""Finite frozen P5 runtime inventory. Hashes/JSON only; never launch any command."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from experiments.spectral_reachability import agent_baseline_launch_readiness_20261028_v1 as saved

NAME = 'agent_p5_frozen_replication_assets_20261028_v1'
HELPER_SHA = '6056d61e1463a13db6a20d28467196558d1ad92de4790c25c2f8f08d979bb042'
SPECS = {
    'assets': ('results/p5_local_asset_binding_20261027_v1_assets/manifest.json', 'bb6e1b895bef75c917c19879596603d05136fb7c8dc28df1269960917fe2f915'),
    'tworoom': ('results/p5_local_asset_binding_20261027_v1_tworoom_calibration1_cuda/report.json', '8e035871f205c734d96936d5588ec902f8ef4a72a98555f6f2f6cf4a6ac72ada'),
    'architecture': ('results/p5_architecture_readiness_20261027_v2/report.json', '4b9d8e612222f11b80ac42123a3af3562add6a7942dffba1c311e2b96fdfcdaf'),
    'native_manifest': ('results/p5_jepa_fresh_fidelity_main_20261027_v12/manifest_frozen.json', '41ef9bf8f5c5ae5c4ff8890a25e6a83137b7e637058a8033281a5eabc9cea07f'),
    'native': ('results/p5_jepa_fresh_fidelity_main_20261027_v12/cuda_main_v1/report.json', 'd8667a4e8537a5e005d13507d4d05b8d7907996549889decc8a00e1db873eda6'),
    'closure': ('results/p5_closure_readiness_20261027_v1.json', '07cd48053f31457ec390f99b03a728ece2c32376cd770c25886161b4b8381980'),
    'native_C1': ('results/p5_native_c1_outcome_audit_20261027_v2_main_v1/report.json', '3d822b29738f6240a68ed4a104e62dcf984b8d2e10f802b19c369eb7519ec811'),
}
SOURCES = {
    'tworoom': ('p5_local_asset_binding_20261027_v1', '32f019577e8c1b4c4b86110b736cd262729d2f5ec1902c0092ed21661e1ab0d5'),
    'native_jepa': ('p5_jepa_fresh_fidelity_main_20261027_v12', '1478f108f35b541ecb73cb149befc0f445471ea20302d43cd842a5864a79b115'),
    'saved_admission': ('p5_native_stage_admission_20261027_v1', 'c3c6936cf26ba88829865599a4d09476b59c8b0e506fbd726d80b59a16d897ee'),
}
CACHE = Path('/home/gbk/.stable_worldmodel')
OFFICIAL_SOURCE_TAR = Path('/tmp/lewm-dinov2-source-DSuJiP7Y/official.tar.gz')
OFFICIAL_SOURCE_SHA = 'c27dcdaf50e9fb5bbdf2bb529da357716372e19c6afab17d5350f3f0094aed4b'


class Files(saved.Files):
    """Fixed authenticated lists only, <=1100 unique files /1.5GiB hashed bytes."""
    def allowed(self, path):
        return (path.is_relative_to(self.root) or path.is_relative_to(self.root.parent/'docs')
                or path.is_relative_to(CACHE/'checkpoints') or path == OFFICIAL_SOURCE_TAR)

    def digest(self, path):
        path = Path(path)
        if path not in self.hashes:
            if len(self.hashes) >= 1100 or path.stat().st_size + self.bytes_hashed > 1536*1024**2:
                raise ValueError('bounded P5 inventory hash cap')
        return super().digest(path)

    def size(self, path):
        return Path(path).stat().st_size


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def source_check(files, key, errors):
    module, expected = SOURCES[key]
    return saved.check(files, files.path('experiments/spectral_reachability/'+module+'.py'), expected, errors)


def check_list(files, entries, errors):
    count = 0
    for path, expected in entries.items():
        if not Path(path).is_absolute():
            errors.append(saved.issue('nonabsolute_binding', path))
        else:
            count += int(saved.check(files,Path(path),expected,errors))
    return count


def build(files):
    common = []
    saved.check(files,Path(saved.__file__),HELPER_SHA,common)
    refs, ref_errors = {}, {}
    for key,(path,expected) in SPECS.items():
        errors=[]
        refs[key]=saved.read_pinned(files,files.path(path),expected,errors)
        ref_errors[key]=errors
    two_errors = list(common)+ref_errors['assets']+ref_errors['tworoom']
    native_errors = list(common)+ref_errors['native_manifest']+ref_errors['native']+ref_errors['architecture']
    source_check(files,'tworoom',two_errors)
    source_check(files,'native_jepa',native_errors)
    assets=refs['assets']
    two_count=0
    dataset=None
    if assets is not None:
        if (assets.get('schema')!='p5_local_asset_binding_20261027_v1.v1'
                or assets.get('P5_admission') is not False or assets.get('P1_admission') is not False
                or assets.get('calibration_roots')!=list(range(202610360,202610368))):
            two_errors.append(saved.issue('tworoom_frozen_protocol',SPECS['assets'][0]))
        entries=assets.get('files',[])
        if len(entries)!=41:
            two_errors.append(saved.issue('asset_census',SPECS['assets'][0],41,len(entries)))
        else:
            # Follow exactly this old allowlist. No recursive asset discovery or dataset scan.
            for row in entries:
                if row.get('exists') is True:
                    two_count += int(saved.check(files,Path(row['path']),row.get('sha256'),two_errors))
                elif files.is_file(Path(row['path'])):
                    two_errors.append(saved.issue('old_absence_changed',row['path'],False,True))
        dataset=assets.get('dataset')
        if dataset is None:
            two_errors.append(saved.issue('dataset_metadata_missing','TwoRoom HDF5'))
        else:
            path=Path(dataset['path'])
            try:
                if str(path)!=str(CACHE/'datasets/tworoom.h5') or files.size(path)!=dataset['bytes']:
                    two_errors.append(saved.issue('dataset_metadata_mismatch',path,dataset['bytes']))
            except Exception as exc:
                two_errors.append(saved.issue('dataset_missing',path,detail=str(exc)))
    two=refs['tworoom']
    if two is not None and (two.get('status')!='replay_completed' or two.get('P5_admission') is not False
            or not two.get('gates') or not all(v is True for v in two['gates'].values())):
        two_errors.append(saved.issue('tworoom_actual_replay_unverified',SPECS['tworoom'][0]))
    native_manifest,native=refs['native_manifest'],refs['native']
    native_count=0
    if native_manifest is not None:
        bindings=native_manifest.get('bindings',{})
        if not isinstance(bindings,dict) or len(bindings)!=988:
            native_errors.append(saved.issue('native_binding_census',SPECS['native_manifest'][0],988))
        else:
            if bindings.get(str(OFFICIAL_SOURCE_TAR)) != OFFICIAL_SOURCE_SHA:
                native_errors.append(saved.issue('official_source_binding',OFFICIAL_SOURCE_TAR,OFFICIAL_SOURCE_SHA))
            native_count=check_list(files,bindings,native_errors)
        if native_manifest.get('P5_admission') is not False:
            native_errors.append(saved.issue('native_no_admission',SPECS['native_manifest'][0]))
    if native is not None:
        strict=native.get('strict_loads',{})
        if (native.get('status')!='fresh_native_panel_fidelity_recorded'
                or native.get('P5_admission') is not False or native.get('model_unchanged') is not True
                or native.get('sources_unchanged') is not True or native.get('new_physical_steps')!=0
                or [native.get('protocol',{}).get(k) for k in ('training','cpu_neural_fallback','root_selection_changed')]!=[False]*3
                or native.get('protocol',{}).get('seeds')!=[2026102702,2026102703,2026102704]
                or set(strict)!= {'predictor','proprio_encoder'}
                or not all(r.get('exact_match') is True for r in strict.values())):
            native_errors.append(saved.issue('native_factual_runtime_protocol',SPECS['native'][0]))
        if native_manifest is not None and native.get('manifest_sha256')!=canonical_digest(native_manifest):
            native_errors.append(saved.issue('native_manifest_canonical_binding',SPECS['native_manifest'][0],
                                             native.get('manifest_sha256'),canonical_digest(native_manifest)))
    admission_errors=[]
    okay=source_check(files,'saved_admission',admission_errors)
    audit_commands=[saved.command(files,SOURCES['saved_admission'][0],'--require-admission')] if okay else []
    if audit_commands:
        audit_commands[0]['expected_exit_code']=3
    def row(key,family,environment,errors,count):
        return dict(candidate=key,architecture_family=family,environment=environment,
                    status='ready_for_MAIN_frozen_runtime_preflight' if not errors else 'blocked',
                    errors=errors,bindings_verified=count,training_repair_teacher_readout_used=False,
                    existing_runtime_completed=(refs['tworoom' if key=='tworoom' else 'native'] is not None),
                    rerun_recommended=False,actual_run_command=None,
                    independent_architecture_replication_established=False,
                    checkpoint_license_remote_revision_verified=None,
                    launch_authorized=False,P5_admission=False)
    rows=[row('tworoom','LeWM_CLS192','TwoRoom',two_errors,two_count),
          row('native_jepa','JEPA_WM_AdaLN_patch384_with_proprio','PushT',native_errors,native_count)]
    cube_errors=[]
    if assets is not None:
        for r in assets['files']:
            if '/checkpoints/cube/lewm/' in r['path']:
                saved.check(files,Path(r['path']),r.get('sha256'),cube_errors)
    cube_errors.extend([saved.issue('source_native_normalizer_unbound','Cube frozen action statistics',
                                    detail='Need existing 5D raw action stats, data/revision/units binding; no fit or replacement.'),
                        saved.issue('no_training_free_cube_runner','experiments/run_live_cube_action_adapter_planning.py',
                                    detail='Known CLI requires --adapter-artifact. Excluded from frozen native replication; do not run it.'),
                        saved.issue('cube_packet_width','Cube action_encoder.input_dim',10,25,
                                    detail='5D raw x5=25; TwoRoom 2D x5=10 cannot be substituted.')])
    rows.append(dict(candidate='cube',architecture_family='LeWM_CLS192',environment='OGBCube',
                     status='blocked',errors=cube_errors,actual_run_command=None,
                     independent_architecture_replication_established=False,P5_admission=False))
    return dict(schema=NAME+'.report',candidates=rows,
                latest_saved_reports={k:dict(path=str(files.path(SPECS[k][0])),sha256=SPECS[k][1],
                                            status=v.get('status') if v else None,errors=ref_errors[k]) for k,v in refs.items()},
                saved_admission_commands=audit_commands,saved_admission_errors=admission_errors,
                family_boundary='TwoRoom/Cube/PushT/Reacher LeWM are SAME architecture; three JEPA tasks are SAME AdaLN family.',
                independent_architecture_replications_established=0,
                known_independent_dynamics_candidate='JEPA_WM_AdaLN, native strict-load/factual evidence, not P5 efficacy',
                full_dataset_sha_reverified=False,dataset_metadata=dataset,
                licensing_and_remote_revision='local model cards/source claims only, independent attestation unknown',
                missing_science=['P1/core/used-component freeze and per-model P0/task premises',
                                 'predeclared independent roots and mechanism plus real task-benefit replication',
                                 'matched strong controls and separate same-model multitask matrix'],
                network_calls=0,neural_calls=0,physics_calls=0,unpickled=False,
                scientific_admission=False,P5_admission=False)


def propose(files,report,key,output,purpose):
    r=next(x for x in report['candidates'] if x['candidate']==key)
    if r['status']!='ready_for_MAIN_frozen_runtime_preflight' or not purpose.strip():
        raise ValueError('blocked inputs or missing MAIN new experiment purpose')
    output=Path(output)
    if not output.is_absolute() or output.parent!=files.root/'results' or output.exists():
        raise ValueError('need NEW direct absolute results child')
    if shutil.disk_usage(output.parent).free<200*1024**2:
        raise ValueError('200MiB floor unavailable; MAIN must additionally budget actual output')
    module=SOURCES[key][0]
    if key=='tworoom':
        if not output.name.startswith('p5_local_asset_binding_20261027_v1'):
            raise ValueError('original TwoRoom runner requires its p5_local_asset_binding_20261027_v1 output prefix')
        args=['--replay-cuda','--binding-manifest',files.path(SPECS['assets'][0]),'--output',output]
        physical='15 original warmup physics steps; MAIN approval required, not new candidate rollout'
    else:
        args=['cuda','--manifest',files.path(SPECS['native_manifest'][0]),'--output',output]
        physical='0 new physics, saved factual inputs only'
    r['actual_run_command']=saved.command(files,module,*args,role='MAIN_only_after_code_review_and_HOST_CUDA_authorization')
    r['actual_run_command']['physics_scope']=physical
    r['main_new_purpose']=purpose
    r['remaining_MAIN_gates']=['Review existing strict verifier/input identities and source modifications',
                               'Confirm applicable license and immutable asset provenance; currently unknown',
                               'Exclusive HOST CUDA and actual output/memory/wall budget approval; no CPU fallback',
                               'No scientific/P1/P5 admission and no result-based root or threshold changes']


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,default=saved.ROOT)
    p.add_argument('--new-run-candidate',choices=['tworoom','native_jepa'])
    p.add_argument('--new-run-output',type=Path)
    p.add_argument('--main-purpose',default='')
    a=p.parse_args(argv)
    try:
        f=Files(a.project);r=build(f)
        if a.new_run_candidate or a.new_run_output:
            if not(a.new_run_candidate and a.new_run_output):
                raise ValueError('need actual candidate and NEW output')
            propose(f,r,a.new_run_candidate,a.new_run_output,a.main_purpose)
        r.update(hash_reads_unique=len(f.hashes),bytes_hashed=f.bytes_hashed)
        print(json.dumps(r,indent=2,sort_keys=True,allow_nan=False))
        return int(any(x['status']=='blocked' for x in r['candidates']))
    except Exception as exc:
        print(json.dumps(dict(schema=NAME+'.error',error=type(exc).__name__,detail=str(exc))))
        return 2


if __name__=='__main__':
    raise SystemExit(main())
