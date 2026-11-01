"""Result-blind raw15 input preparation; official GRASP same-input dispatch CLOSED.

No model/physical executor. Never import/execute the cached foreign solver.
Existing producer task digests and normalization are reused, not redefined.
"""
from __future__ import annotations
import argparse
import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import zlib
import numpy as np
from experiments.spectral_reachability import p1_conditional_task_admission_20261027_v1 as core
from experiments.spectral_reachability import p2_grasp_official_contract_20261027_v1 as official

NAME = 'agent_p2_grasp_raw15_prepare_20261029_v1'
ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT/'results/agent_p2_three_program_20261028_v1_prepare_MAIN01.json'
UPSTREAM_SHA = '86af2d24f73e769fb3b91fa8a571f8ca265df95c93af780d3924ea644585d461'
OFFICIAL = ROOT/'results/p2_grasp_official_contract_20261027_v1'
MODEL = '48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607'
CONFIG = '2564086e961e7b5c7c04dffc451091115b389a590645ff19653c64fd0bc16e09'
TENSORS = '4b12c01f5859f4b46bea478f3a8d5e2d3f516df456a622a3feec31bde7e33546'
NORMALIZER = '0a4eb4592951139e596b35becede17d175df7b48a2fde16679ab0aac7c4dce34'
SOURCE = '85ca1df3de910628bfd6c5bd1a0a482f61880a2ee76bfd404732a77dd8c7666a'
PINS = {
    str(UPSTREAM.relative_to(ROOT)): UPSTREAM_SHA,
    'experiments/spectral_reachability/p1_conditional_task_admission_20261027_v1.py':
        'd28d48c183791bfe1cae03e8a78ba27e28039cfd355993a19b32992cd7f6e211',
    'experiments/spectral_reachability/p2_grasp_official_contract_20261027_v1.py':
        '0af7b3204ca9ca03450447b4b7b399d1f72ab54e9441da71d5d0bf41c4b86951',
    'experiments/spectral_reachability/agent_p2_three_program_20261028_v1.py':
        '3111312477ff3cd9a764dc7d063861c1becaaf58885dc0fb25c2f3ec73d15fd5',
    'experiments/spectral_reachability/p1_moving_grasp_baseline_cuda_20261027_v1.py':
        '52d3157574e4e2726f6d88b43bd7e5dc804ce79730b65512888e63e0dd7c7106',
    'experiments/spectral_reachability/p1_constructive_grasp_baseline_20261027_v1.py':
        'ae9d2dd59bf613bc10c81858cb767dad1d92dcf9b7b7d103bedb018e0bd1fa28',
    'experiments/run_pusht_counterfactual.py': NORMALIZER,
}
TASK_SHAS = ('80f0f4ed14498509e596322b830f0f8d1bedc4d38f9d3b7d64b57da375f9a132',
             '009ed5769fb55513100c1223c5659bd5b34911b5cfe13e9b121117c14a0db2d7')
LIMIT, FLOOR = 2*1024**2, 200*1024**2
need = core.require


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def parse(raw):
    def pairs(xs):
        d = {}
        for k, v in xs:
            need(k not in d, 'duplicate JSON key'); d[k] = v
        return d
    value = json.loads(raw, object_pairs_hook=pairs)
    encoded(value)  # Also rejects nonfinite values anywhere in the document.
    return value


def read_bound(path, expected, *, compressed=False):
    core.hash_value(expected); path = Path(path)
    need(path.is_file() and not path.is_symlink() and path.stat().st_size <= LIMIT,
         'bounded regular pinned input')
    raw = path.read_bytes(); need(len(raw) <= LIMIT and sha(raw) == expected, 'input byte SHA')
    if compressed:
        d = zlib.decompressobj(31); raw = d.decompress(raw, LIMIT+1)
        need(len(raw) <= LIMIT and d.eof and not d.unused_data and not d.unconsumed_tail,
             'single complete bounded task gzip')
    return raw


def native(value, shape):
    a = core.array(value, shape)
    need(np.array_equal(a, a.astype(np.float32)), 'literal original float32 values')
    return a.astype(np.float32)


def normalization():
    return dict(mean=np.asarray([-.0087, .0068], np.float32).tolist(),
                std=np.asarray([.2019, .2002], np.float32).tolist(), source_sha256=NORMALIZER,
                training_provenance_verified=False,
                raw_semantics='relative raw already action_scale/100; no second division')


def initial_coordinates(task):
    """Actual arithmetic ABI, not proposed actions or a neural experiment."""
    raw = np.zeros((15, 2), np.float32)
    # Reuse the producer packet primitive three times; never relax its H2 task schema.
    normalized = np.concatenate([core.normalized_tokens(task, raw[k:k+5]) for k in (0,5,10)])
    official_zero_raw = np.tile(np.asarray(task['action_mean'], np.float32), (15,1))
    return dict(p2_zero_logit_raw15=raw.tolist(), p2_zero_logit_normalized3=normalized.tolist(),
                official_zero_normalized3=np.zeros((3,10), np.float32).tolist(),
                official_zero_normalized_inverse_raw15=official_zero_raw.tolist(),
                initializations_equal=bool(np.array_equal(raw, official_zero_raw)))


def raw15_inputs(request, logits):
    """Validate real P2 action coordinates and build the FIRST model input only.

    Later model outputs are unknown: do not manufacture future contexts. This
    function performs only saved float32 arithmetic and has no backend callback.
    """
    need(set(request) == {'root_context','goal','q','normalization'}, 'model input whitelist')
    core.context(request['root_context']); native(request['root_context']['latent_history'], (3,192))
    native(request['root_context']['action_history'], (2,10))
    need(request['normalization'] == normalization() and request['q'] == np.eye(192).tolist(),
         'original normalizer and identity192 Q')
    native(request['goal'], (192,)); u = core.array(logits, (30,))
    need(np.linalg.norm(u) <= .75*np.sqrt(15), 'literal joint30 logit ball, no projection/epsilon')
    raw = np.tanh(u).astype(np.float32).reshape(15,2)
    tokens = np.concatenate([core.normalized_tokens(dict(action_mean=request['normalization']['mean'],
                         action_std=request['normalization']['std']),raw[k:k+5]) for k in (0,5,10)])
    return dict(raw15=raw.tolist(), raw15_sha256=sha(raw.tobytes()), normalized_packets=tokens.tolist(),
                first_latent_input=deepcopy(request['root_context']['latent_history']),
                first_normalized_actions=np.concatenate((np.asarray(request['root_context']['action_history'],
                    np.float32),tokens[:1])).tolist(), future_contexts=None,
                is_official_GRASP_input=False)


def official_facts():
    """Read pinned text/AST, never import the foreign solver or parse a checkpoint."""
    specs = [('grasp_solver.py','lewm'), ('config/eval/solver/grasp.yaml','lewm'),
             ('planning/grasp.py',None), ('preprocessor.py',None), ('LICENSE',None)]
    bodies = {(path, adapter): official.read_source(OFFICIAL,path,adapter) for path,adapter in specs}
    solver = bodies['grasp_solver.py','lewm'].decode()
    cls = next(n for n in ast.parse(solver).body if isinstance(n,ast.ClassDef) and n.name=='GRASPSolver')
    init = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    defaults = dict(zip([a.arg for a in init.args.args][-len(init.args.defaults):],
                       [ast.literal_eval(n) for n in init.args.defaults], strict=True))
    cfg = {}
    for ln in bodies['config/eval/solver/grasp.yaml','lewm'].decode().splitlines():
        if not ln.strip(): continue
        k,v = ln.split(':',1)
        need(k not in cfg, 'unique official YAML field')
        cfg[k] = v.strip() if k in ('_target_','model','seed') else ast.literal_eval(v.strip())
    expected = dict(n_steps=750,batch_size=1,num_samples=1,lr_s=.1,lr_a=.005,
                    state_noise_scale=.005,gd_interval=50,gd_opt_steps=10,gd_lr=.05,device='cuda')
    need({k:cfg[k] for k in expected} == expected and cfg['seed']=='${seed}' and
         defaults['sync_mode']=='gd' and defaults['schedule_decay'] is False,
         'original fixed official config/defaults, not result-selected settings')
    anchors = [official.anchor(OFFICIAL,'grasp_solver.py',needle,'lewm') for needle in (
        'return emb[:, -1, :]', 'def _jepa_predict_step(', 'def _jepa_full_rollout(',
        'a_init = torch.zeros(current_bs, T, self.action_dim', 's_t = s_full[:, t].detach()',
        's_virtual.data += cur_noise * torch.randn_like(s_virtual)', 'all_top_actions.append(a_opt.detach().cpu())')]
    return dict(repositories=dict(DINO_GRASP=dict(repo=official.REPO,commit=official.COMMIT),
                LeWM_GRASP=dict(repo=official.ADAPTERS['lewm'][0],commit=official.ADAPTERS['lewm'][1])),
                configuration=expected,seed_binding=cfg['seed'],sync_mode='gd',anchors=anchors,
                official_noise_RNG='global torch.randn_like; constructor generator does not govern this call',
                paper_table_code_revision=None,official_frozen_process=None)


def budget():
    syncs = [k+1 for k in range(750) if (k+1)%50==0 and k>0]
    native_predict = 3*(750+len(syncs)*10)
    return dict(basis='prospective batch1 H3=3future packets; NOT actual attempted/returned',
                joint_iterations=750,sync_iterations=len(syncs)*10,sync_after=syncs,
                official_solver_predict_per_root=native_predict,
                official_solver_backward_per_root=750+len(syncs)*10,
                official_solve_RGB_encoder_per_root=2,official_solve_projector_per_root=2,
                optional_cached_input_adapter_encoder=0,
                optional_independent_full15_plus_segmented15_replay=6,
                prospective_with_dual_replay_predict_per_root=native_predict+6,
                prospective_with_dual_replay_predict_two_roots=2*(native_predict+6),
                P2_joint_standalone_predict_per_root=138,P2_composed_standalone_predict_per_root=936,
                P2_standalone_scope='OLDER pinned prepare protocol only; NOT current common-cap comparator',
                current_common_cap_MAIN_notice=dict(provenance='MAIN user message; not independently consumed here',
                    joint_actual_predict=1902,composition_actual_predict=1062,J_each=85,
                    common_predict_cap=1902,wall_cap_seconds=450,
                    use_as_certified_saved_authority=False),
                AE_predictor_pred_proj='one each per transition in LeWM; nested not added to API',
                actual_attempts=None,actual_returns=None,FLOPs=None,actual_wall=None,
                backward_engine_work=None,equal_API=False,equal_total_compute=False)


def make_request(p, exports):
    need(p['schema']=='agent_p2_three_program_20261028_v1.prepare' and len(exports)==2,
         'original raw15 prepared authority, not old H2 plan/physical report')
    c = p['contract']
    for k,v in dict(roots=[0,1],condition='off',known_roots=2,horizon=3,raw_steps=15,
                   steps_per_program=5,logit_cap=float(.75*np.sqrt(15))).items():
        need(type(c[k]) is type(v) and c[k]==v, 'original raw15 contract '+k)
    need(p['expected_model_hash']==TENSORS and p['source_report_sha256']==SOURCE and
         c['P1_admission'] is False and c['P2_admission'] is False and c['fresh'] is False and
         [(s['root_slot'],s['condition'],s['index']) for s in p['slots']]==[(0,'off',0),(1,'off',2)],
         'both known off tasks, no relabel/subset')
    slots = []
    for s,e in zip(p['slots'],exports,strict=True):
        core.fields(s,'blind_path blind_sha256 condition index packet q root_slot task_file task_id task_sha256 warm_raw10', 'original slot')
        core.fields(e,'schema task task_contract task_id P1_admission','original task export')
        t,tc=e['task'],e['task_contract']; core.validate_task(t)
        core.fields(tc,'blind_path blind_sha256 condition goal_pose_edited metric normalization pixel_sha256 prepare_sha256 raw_sha256 renderer_only root_slot schema source_report_sha256 with_target',
                    'original task contract')
        core.fields(s['packet'],'root_context goal','original model-visible packet')
        need(type(s['root_slot']) is int and type(s['index']) is int and
             type(tc['root_slot']) is int and tc['with_target'] is False, 'literal slot identities/flags')
        need(e['schema']=='p1_moving_overlay_recompile_cuda_20261027_v1.task_export' and
             tc['schema']=='p1_moving_overlay_recompile_cuda_20261027_v1.task_contract' and
             e['P1_admission'] is False and s['task_sha256']==TASK_SHAS[s['root_slot']] and
             s['task_id']==e['task_id']==core.task_id(t) and t['task_contract_sha256']==core.digest(tc),
             'producer-native task ID / task contract digest, not own compact digest')
        need((tc['root_slot'],tc['condition'],tc['with_target'])==(s['root_slot'],'off',False) and
             tc['renderer_only'] is True and tc['goal_pose_edited'] is False and
             tc['source_report_sha256']==SOURCE and tc['normalization']==normalization() and
             tc['blind_path']==s['blind_path'] and tc['blind_sha256']==s['blind_sha256'],
             'original off renderer/source/blind/normalizer')
        need(t['model_sha256']==MODEL and t['model_config_sha256']==CONFIG and
             t['root_context']==s['packet']['root_context'] and t['goal']==s['packet']['goal'] and
             t['projection']==s['q']==np.eye(192).tolist() and tc['metric']=='identity_native192_no_fit',
             'same complete model-visible inputs/goal/Q')
        warm=native(t['root_raw_action_history'],(10,2))
        need(np.array_equal(warm,np.asarray([[.1,0]]*5+[[0,.1]]*5,np.float32)) and
             s['warm_raw10']==warm.tolist() and tc['raw_sha256']==sha(warm.tobytes()) and
             t['action_mean']==normalization()['mean'] and t['action_std']==normalization()['std'],
             'literal nonzero raw warmup10, one original normalization')
        native(t['root_context']['latent_history'],(3,192));native(t['goal'],(192,))
        need(len(tc['pixel_sha256'])==4 and t['goal_source_sha256']==tc['pixel_sha256'][-1],
             'original history3 + goal image identities')
        for h in (*tc['pixel_sha256'],s['blind_sha256']):core.hash_value(h)
        request=dict(root_context=deepcopy(t['root_context']),goal=deepcopy(t['goal']),
                     q=deepcopy(t['projection']),normalization=normalization())
        slots.append(dict(root_slot=s['root_slot'],condition='off',task_id=e['task_id'],
            task_file=s['task_file'],task_sha256=s['task_sha256'],blind_pointer=dict(path=s['blind_path'],
            sha256=s['blind_sha256'],opened=False),observed_warmup_steps=[0,5,10],pixel_sha256=tc['pixel_sha256'],
            source_export_horizon=2,target_future_packets=3,target_raw_steps=15,warm_raw10=warm.tolist(),
            model_input=request,initialization=initial_coordinates(t),first_call_preview=raw15_inputs(request,np.zeros(30))))
    return slots


def differences():
    return [dict(id='history',official='last image H1; lifted single-state edges; sync starts H1',
                 P2='3 actual latents plus 2 between-state action packets throughout recurrence',
                 required='history-aware collocation AND serial sync adapter; explicitly a variant, not source-identical'),
            dict(id='coordinates',official='unbounded normalized action optimized directly; no tanh/logit ball',
                 P2='joint30 logit L2 .75sqrt15; tanh(float64).astype(float32) raw; normalize once',
                 required='explicit bounded raw-coordinate variant with the same global ball, not per-packet H2 clipping'),
            dict(id='initialization',official='normalized action zero; inverse raw equals mean',
                 P2='raw zero at logit zero; normalized action is not zero',
                 required='prospective coordinate/initialization choice; cannot silently equate the two zeros'),
            dict(id='budget',official='750 joint + 15x10 serial sync; 2700 predict/root before verification',
                 P2='OLDER pinned protocol only: joint138/composition936; MAIN current notice: joint1902/composition1062, 85J each, cap1902/wall450; current receipt not consumed',
                 required='MAIN predeclared comparison resource/stopping rule; no truncation/padding/equalFLOPs claim'),
            dict(id='official_process',official='paper trial/checkpoint/process/seed pairing unknown',
                 P2='local frozen public mean/std bound; training provenance not verified',
                 required='original frozen process only for official-paper replication; NOT a reason to fit new stats for local adaptation')]


def build():
    bindings={}
    for name,h in PINS.items():
        read_bound(ROOT/name,h);bindings[str(ROOT/name)]=h
    p=parse(read_bound(UPSTREAM,UPSTREAM_SHA));exports=[]
    for s in p['slots']:
        path=Path(s['task_file']);i=s['root_slot']
        need(type(i) is int and i in (0,1) and path==ROOT/f'results/p1_moving_overlay_recompile_cuda_20261027_v1_cuda_MAIN01/task_{2*i:02d}.json.gz'
             and p['bindings_sha256'][str(path)]==TASK_SHAS[i], 'only two known task exports, no discovery')
        exports.append(parse(read_bound(path,TASK_SHAS[i],compressed=True)));bindings[str(path)]=TASK_SHAS[i]
    slots=make_request(p,exports);facts=official_facts()
    for a in facts['anchors']:bindings[str(OFFICIAL/'sources'/('lewm__'+a['path'].replace('/','__')+'.source.txt'))]=a['source_sha256']
    for name in ('lewm__config__eval__solver__grasp.yaml.source.txt','planning__grasp.py.source.txt',
                 'preprocessor.py.source.txt','LICENSE.source.txt'):
        bindings[str(OFFICIAL/'sources'/name)]=official.PINS[name]
    config=Path(p['policy'])/'config.json';metadata=parse(read_bound(config,CONFIG));bindings[str(config)]=CONFIG
    need(metadata['encoder']['image_size']==224 and metadata['encoder']['patch_size']==14 and
         metadata['projector']['input_dim']==192,'real224/patch14/257hidden->CLS192 config')
    for path in (Path(__file__),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')):
        need(path.is_file() and not path.is_symlink(), 'own triple present');bindings[str(path)]=sha(path.read_bytes())
    for path,h in bindings.items():read_bound(path,h)
    return dict(schema=NAME+'.prepare',status='inputs_prepared_official_same_contract_refused',
        input_authority=dict(path=str(UPSTREAM),sha256=UPSTREAM_SHA,scope='pinned model-visible prepared inputs only; not outcome consume'),
        slots=slots,official_sources=facts,differences=differences(),prospective_budget=budget(),
        existing_reuse=dict(moving_H3_bounded='p1_moving_grasp_baseline_cuda_20261027_v1',
            existing_horizon=2,warning='cannot relabel raw10 as raw15; per-packet domain differs',
            H1_reference='p2_grasp_reference_core_20261027_v1; H3 input loss still unresolved'),
        declared_checkpoint=dict(weights_sha256=MODEL,config_sha256=CONFIG,tensor_sha256=TENSORS,
            model_weights_opened=False,model_tensor_values_verified_this_prepare=False),
        bindings_sha256=bindings,unknown=dict(official_process=None,official_seed_ids=None,total_FLOPs=None,
            actual_runtime=None),official_same_input_ready=False,production_dispatch=False,physical_release=False,
        equal_total_compute=False,official_reproduction=False,P1_admission=False,P2_admission=False,
        new_neural_calls=0,new_physics=0)


def write_new(path, value):
    path=Path(path).absolute();raw=encoded(value)
    need(path.name.startswith(NAME+'_') and not path.exists() and not path.is_symlink() and
         len(raw)<=LIMIT and shutil.disk_usage(path.parent).free>=FLOOR+len(raw),'exclusive own <=2MiB +200MiB floor')
    with path.open('xb') as f:
        need(f.write(raw)==len(raw),'full metadata write');f.flush();os.fsync(f.fileno())
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return dict(path=str(path),sha256=sha(raw),status=value['status'])


def verify(path, expected):
    value=parse(read_bound(path,expected));need(value==build(),'entire outcome-blind request rebuild');return value


def official_execution_request(value):
    # Fail even if a caller forges ready=True or deletes the diagnostics.
    raise ValueError('no official same-H3/joint30-bounded raw15 adapter in this version; no dispatch/release')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('inspect','prepare','verify'))
    p.add_argument('--output');p.add_argument('--prepared');p.add_argument('--prepare-sha');a=p.parse_args(argv)
    if a.command=='prepare':out=write_new(a.output,build())
    elif a.command=='verify':
        v=verify(a.prepared,a.prepare_sha);out=dict(status=v['status'],verified_saved_inputs=True,production_dispatch=False)
    else:
        v=build();out={k:v[k] for k in ('status','differences','prospective_budget','official_same_input_ready','production_dispatch')}
    print(encoded(out).decode(),end='');return 0


if __name__=='__main__':raise SystemExit(main())
