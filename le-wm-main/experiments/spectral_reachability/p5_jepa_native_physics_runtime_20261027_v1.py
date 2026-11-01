"""Fresh-native physics runtime preparation/probe; no physics or neural runner.

bootstrap prints a main-only installer plan; it NEVER installs anything.
prepare and probe never construct/reset/step a physics environment.
"""
import argparse
import hashlib
import importlib.metadata as md
import importlib.util
import json
import os
from pathlib import Path
import sys

NAME = "p5_jepa_native_physics_runtime_20261027_v1"
OLD = "p5_jepa_physical_bridge_20261027_v1"
OLD_SHA = "10760e277e91926d45086c640b0e64eb3e6c1f72d369a7559811c424b6bdd94d"
ROOT = Path(__file__).resolve().parents[2]
ENV_REL = "external/jepa-wms/evals/simu_env_planning/envs/pusht_env/pusht_env.py"
LOCK = {"wheel":"0.45.1", "gym":"0.23.1", "pymunk":"6.8.0",
        "scikit-image":"0.25.2", "lazy-loader":"0.4", "tifffile":"2025.6.11"}
INHERITED = ["numpy","scipy","pygame","opencv-python","shapely","matplotlib",
             "cffi","pycparser","cloudpickle","gym-notices","networkx","pillow",
             "imageio","packaging","setuptools","pip"]
MAX_NEW_BYTES = 150_000_000
SEEDS = [2026102701,2026102702,2026102703,2026102704]


def sha(path):
    p=Path(path)
    if p.is_symlink() or not p.is_file():
        raise ValueError("regular non-symlink source/artifact required: "+str(p))
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def paths(root):
    root=Path(root).resolve()
    return root/".jepa-physics-site",root/f"results/{NAME}/bootstrap_artifacts"


def lock_path():
    return Path(__file__).with_suffix('.requirements.txt')


def verify_lock():
    lines=[l.strip() for l in lock_path().read_text().splitlines() if l.strip() and not l.lstrip().startswith('#')]
    if lines!=[k+'=='+v for k,v in LOCK.items()]:raise ValueError('exact six-package lock required')


def source_paths(root):
    base=root/'external/jepa-wms'
    return [Path(__file__).resolve(), lock_path(),root/'tests'/f'test_{NAME}.py',
        root.parent/'docs'/f'{NAME}.md',root/f'results/{OLD}/manifest_frozen.json',
        base/'pyproject.toml',base/'LICENSE',
        base/'app/plan_common/datasets/pusht_dset.py',
        base/'evals/simu_env_planning/envs/pusht_gym_wrap.py',
        *sorted((base/'evals/simu_env_planning/envs/pusht_env').rglob('*.py'))]


def fresh_proposal():
    # Proposal only. No running or importing a simulator here.
    return dict(status='proposed_not_executed',seeds=SEEDS,selection='all four seeds, fixed order, no retry/replacement/contact-outcome filtering',
        env_constructor=dict(legacy=False,block_cog=None,damping=None,render_action=False,render_size=224,
            reset_to_state=None,relative=True,action_scale=100,with_velocity=True,with_target=True,shape='T',color='LightSlateGray'),
        root='fresh native constructor then env.seed(seed), native reset ONCE; authoritative root is POST reset hidden .01s step',
        constructor_rng='constructor calls seed(None); record that nondeterministic seed but override it BEFORE reset; no setup/physics occurs in constructor',
        factual_stage=dict(initial_pilot_seed=SEEDS[0],root_reset_calls=1,controls=15,substeps_per_control=10,reset_substeps=1,
            space_step_calls=151,simulation_seconds=1.51,frames_to_save=list(range(16)),
            action='explicit np.zeros(2) NATIVE RAW RELATIVE env arguments, NOT None and NOT normalized zero',
            normalized_representation='env argument u is dimensionless relative displacement/100: z=(u-ACTION_MEAN)/ACTION_STD; dataset raw_pixel=100*u; raw zero != normalized zero',
            repeat='independent env with same seed/config, repeat full factual prefix; exact state and RGB agreement receipts before any branch use',
            zero_limit='raw zero holds current positional target with PD velocity damping; NOT physics pause; reset contacts can still evolve'),
        capture=dict(before_reset='source/lock/runtime versions and hashes; explicit config/seed; RNG generator and RandomState state encoded numeric/JSON, never pickle',
            reset='full precision post-reset bodies, shapes, space settings and lossless RGB; native float32 state7 and proprio4 separately; do not use state7 as full snapshot',
            each_tick='raw xy relative argument, actual current agentxy and absolute target=agentxy+100*u, before/after body state, contact count/coverage, RGB and hashes',
            bodies=['position','velocity','angle','angular_velocity','force','torque','center_of_gravity','mass','moment','body_type','is_sleeping'],
            space=['gravity','damping','iterations','collision_slop','collision_bias','collision_persistence','idle_speed_threshold','sleep_time_threshold','current_time_step'],
            shapes='geometry/radius/vertices/offset, friction, elasticity, surface_velocity, collision_type, filter, sensor, colors; all bodies including static walls',
            numeric='float64 arrays plus dtype/shape/SHA; lossless uint8 .npy/PNG RGB, no MP4 as identity evidence; encode infinite thresholds explicitly, no invalid JSON NaN/Infinity',
            hidden_solver='public property receipts are NOT a complete contact-cache serialization; retain live native state. For branching use independent deterministic seed+prefix replays and exact full-public-state/RGB identity gates; do not claim arbitrary snapshot restore'),
        next_action_fidelity=dict(status='separate future frozen protocol, no model or physical runner in this delivery',
            contexts=[0,5,10],target=15,preserve='two history packets and proprio; official native encoder, not HF equivalence',
            requires='first pass runtime import + factual fresh-root repeat identity; freeze a deterministic nonzero env-argument u schedule independent of outcomes BEFORE running that schedule; z=(u-mean)/std, dataset pixel delta=100*u',
            no_efficacy_from_zero='all-zero actual packet/reverse are identical; zero-only traces cannot establish nontrivial action fidelity',
            controls='actual vs native-mean(normalized zero), negative normalized, reversed within five steps; compare factual successor features, not counterfactual target labels',
            future_control='only after separately assessed action fidelity; no observed-target ridge fitting relabelled as prospective planning'),
        not_archived_episode1=True,P5_admission=False)


def versions(names):
    out={}
    for n in names:
        try:
            d=md.distribution(n)
            out[n]=dict(version=d.version,site=str(Path(d.locate_file('')).resolve()))
        except md.PackageNotFoundError:out[n]=None
    return out


def dependency_constraints(names):
    """Read active installed metadata recursively; no package code execution."""
    from packaging.requirements import Requirement
    seen=set();queue=list(names);records={};problems=[]
    while queue:
        name=queue.pop(0)
        canonical=name.lower().replace('_','-')
        if canonical in seen:continue
        seen.add(canonical)
        try:d=md.distribution(name)
        except md.PackageNotFoundError:
            problems.append('missing:'+name);continue
        required=[]
        for raw in d.requires or []:
            r=Requirement(raw)
            if r.marker is not None and not r.marker.evaluate({'extra':''}):continue
            if r.url:problems.append('unreviewed_direct_url:'+raw);continue
            try:v=md.version(r.name)
            except md.PackageNotFoundError:v=None
            ok=v is not None and (not r.specifier or r.specifier.contains(v,prereleases=True))
            required.append(dict(requirement=raw,installed_version=v,satisfied=ok))
            if not ok:problems.append('unsatisfied:'+name+':'+raw)
            queue.append(r.name)
        records[canonical]=dict(version=d.version,site=str(Path(d.locate_file('')).resolve()),requires=required)
    return dict(records=records,problems=problems)


def prepare(root):
    root=Path(root).resolve();verify_lock()
    old=root/f'results/{OLD}/manifest_frozen.json'
    if sha(old)!=OLD_SHA:raise ValueError('old blocked report drift')
    inherited=versions(INHERITED)
    if any(v is None for v in inherited.values()):raise ValueError('missing inherited dependency; review lock before bootstrap')
    base=root/'.venv/lib/python3.12/site-packages'
    if not base.is_dir() or Path(sys.prefix).resolve()!=root/'.venv':
        raise ValueError('prepare must run in existing project Python3.12 .venv')
    if sys.version_info[:2]!=(3,12):raise ValueError('Python3.12 lock only')
    seals={str(p):sha(p) for p in source_paths(root)}
    closure=dependency_constraints(INHERITED)
    if closure['problems']:raise ValueError('inherited metadata constraints rejected: '+str(closure['problems']))
    return dict(schema=NAME,status='prepared_runtime_only_no_physics',root=str(root),seals=seals,
        lock=LOCK,inherited=inherited,inherited_dependency_closure=closure,base_site=str(base),base_python=str(root/'.venv/bin/python'),
        base_python_version=sys.version,bootstrap_policy=dict(new_bytes_limit=MAX_NEW_BYTES,
            no_deps=True,no_torch_download=True,target_overlay=True,
            staging='main-only install temporary/cache files confined to exclusive bootstrap_artifacts directory',
            lock_boundary='version lock now; NOT a preauthenticated artifact hash lock. Probe binds installed target files; no remote artifact authentication claim.',
            source_build='gym0.23.1 only; other five packages binary wheels only; no build isolation or dependency downloads; no source patch',
            failure='main must bound overlay+staging to150MB and retain partial outputs on failure; no uninstall/delete/downgrade of project .venv'),
        fresh_calibration_proposal=fresh_proposal(),physics_executed=False,neural_executed=False,P5_admission=False)


def verify(manifest,root):
    if (manifest.get('schema')!=NAME or manifest.get('status')!='prepared_runtime_only_no_physics'
            or manifest.get('root')!=str(Path(root).resolve()) or manifest.get('lock')!=LOCK
            or manifest.get('fresh_calibration_proposal')!=fresh_proposal()):
        raise ValueError('manifest protocol rejected')
    expected={str(p):sha(p) for p in source_paths(Path(root).resolve())}
    if manifest.get('seals')!=expected:raise ValueError('frozen source/lock changed')
    verify_lock()


def new_bytes(directories):
    total=0
    for directory in directories:
        for base,_,files in os.walk(directory,followlinks=False):
            for f in files:
                p=Path(base)/f
                if not p.is_symlink():total+=p.stat().st_size
    return total


def budget(directories):
    value=new_bytes(directories)
    if value>MAX_NEW_BYTES:raise RuntimeError(f'150MB new-file budget exceeded: {value}; retain partial outputs, no cleanup')
    return value


def installed_records(directory):
    out={}
    for p in sorted(Path(directory).rglob('*')):
        if p.is_symlink():raise ValueError('target symlink rejected')
        if p.is_file():out[str(p.relative_to(directory))]=dict(bytes=p.stat().st_size,sha256=sha(p))
    return out


def bootstrap(root,manifest,manifest_path):
    """Print-only exact installer requirements. No writes or subprocesses."""
    verify(manifest,root)
    if Path(sys.prefix).resolve()!=root/'.venv':raise ValueError('plan must use base project interpreter')
    if versions(INHERITED)!=manifest['inherited']:raise ValueError('base inherited versions/sites changed')
    overlay,staging=paths(root)
    uv=['uv','pip','install','--python',manifest['base_python'],'--target',str(overlay),'--no-deps','--no-cache']
    return dict(schema=NAME,status='installer_plan_only_not_executed',manifest_sha256=sha(manifest_path),
        fresh_target_required=True,target_already_exists=overlay.exists(),target=str(overlay),staging=str(staging),
        environment=dict(PYTHONPATH=str(overlay),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',
            TMPDIR=str(staging),UV_CACHE_DIR=str(staging/'uv-cache'),UV_NO_CACHE='1'),
        setup='main creates fresh target/staging only; do not reuse an unreviewed partial install',
        stage1_argv=uv+['--only-binary',':all:',*[n+'=='+v for n,v in LOCK.items() if n!='gym']],
        stage2_argv=uv+['--no-build-isolation','gym=='+LOCK['gym']],
        rationale='five binary wheels first including wheel build tool; gym source build uses inherited setuptools and overlay wheel. --no-deps prevents torch or other dependency resolution.',
        limits=dict(max_new_bytes=MAX_NEW_BYTES,measure='new_bytes([target,staging]) before/after stages; main must monitor installers and stop above cap; no automatic download/space guarantee from a plan'),
        next='same base .venv/bin/python, explicit PYTHONPATH=target only, run probe; GPU main omits PYTHONPATH overlay',
        installation_executed=False,physics_executed=False,neural_executed=False)


def probe(root,manifest,manifest_path):
    verify(manifest,root)
    overlay,staging=paths(root)
    installed=versions(list(LOCK)+INHERITED)
    problems=[]
    if Path(sys.prefix).resolve()!=root/'.venv':problems.append('not_original_project_interpreter')
    if not sys.dont_write_bytecode:problems.append('PYTHONDONTWRITEBYTECODE_required')
    search=[str(Path(p or os.getcwd()).resolve()) for p in sys.path]
    if str(overlay) not in search or (manifest['base_site'] in search and search.index(str(overlay))>search.index(manifest['base_site'])):
        problems.append('target_not_before_base_site')
    for n,v in LOCK.items():
        found=installed[n]
        if found is None or found['version']!=v:problems.append('version:'+n)
        elif Path(found['site'])!=overlay:problems.append('not_target_distribution:'+n)
    for n,expected in manifest['inherited'].items():
        if installed[n]!=expected:problems.append('inherited_drift:'+n)
    closure=dependency_constraints(INHERITED)
    if closure!=manifest['inherited_dependency_closure']:problems.append('inherited_transitive_metadata_drift')
    active=dependency_constraints(list(LOCK))
    problems.extend(active['problems'])
    result=dict(schema=NAME,status='runtime_probe_failed',manifest_file_sha256=sha(manifest_path),
        problems=problems,versions=installed,active_dependency_constraints=active,physics_executed=False,neural_executed=False,
        env_construct_calls=0,reset_calls=0,physics_step_calls=0,P5_admission=False)
    if problems:return result  # No native import in an unapproved runtime.
    before=installed_records(overlay)
    if any(n.lower().startswith(('torch','transformers','nvidia')) for n in os.listdir(overlay)):
        raise ValueError('neural package unexpectedly installed in physics target')
    budget([overlay,staging])
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT']='1'
    os.environ['MPLCONFIGDIR']=str(staging/'.mpl-cache')
    os.environ['SDL_VIDEODRIVER']='dummy';os.environ['SDL_AUDIODRIVER']='dummy'
    try:
        import gym
        import pymunk
        import skimage.transform
        if not hasattr(pymunk.Space,'add_collision_handler'):raise ValueError('native collision API absent')
        for module,n in ((gym,'gym'),(pymunk,'pymunk'),(skimage,'scikit-image')):
            if not Path(module.__file__).resolve().is_relative_to(overlay):raise ValueError('import origin rejected: '+n)
        path=root/ENV_REL
        spec=importlib.util.spec_from_file_location(NAME+'_official_source_import',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        cls=module.PushTEnv
        if cls.__module__!=spec.name:raise ValueError('native class provenance rejected')
        imported={}
        for key,m in list(sys.modules.items()):
            if key.split('.')[0] in ('gym','pymunk','skimage','pygame','cv2','shapely','matplotlib','numpy','scipy'):
                p=getattr(m,'__file__',None)
                if p and Path(p).is_file() and not Path(p).is_symlink():imported[str(Path(p).resolve())]=sha(p)
        verify(manifest,root)
        result.update(status='native_import_ready_physics_not_executed',native_source_sha256=sha(path),
            native_class=cls.__module__+'.'+cls.__name__,imported_dependency_hashes=imported,
            source_hashes_unchanged=True,installed_target_records=before,new_bytes=budget([overlay,staging]),
            ready_scope='main may review next fresh factual physics protocol; no physics/render/ABI execution validation or neural fidelity claim')
        if installed_records(overlay)!=before:raise ValueError('target files changed during import probe')
    except Exception as exc:
        result['status']='runtime_probe_failed'
        result['problems'].append(type(exc).__name__+': '+str(exc))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','bootstrap','probe'],nargs='?',default='prepare')
    p.add_argument('--manifest',type=Path,default=ROOT/f'results/{NAME}/manifest_frozen.json')
    args=p.parse_args()
    if args.command=='prepare':result=prepare(ROOT)
    else:
        manifest=json.loads(args.manifest.read_text())
        if args.command=='bootstrap':result=bootstrap(ROOT,manifest,args.manifest)
        else:result=probe(ROOT,manifest,args.manifest)
    print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))
    if result['status']=='runtime_probe_failed':raise SystemExit(2)


if __name__=='__main__':main()
