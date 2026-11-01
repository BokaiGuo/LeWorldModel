"""Local JEPA architecture contract; stdout only; no model imports/unpickling.

0=local contract consistent, not official provenance/scientific admission.
2=invalid input/reserve/contract; 3=--require-complete with remote/human gaps.
Large existing weights are STREAM HASHED, not loaded, copied or downloaded.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import stat
import struct
import subprocess

NAME = 'p5_architecture_provenance_20261027_v1'
ROOT = Path(__file__).resolve().parents[2]
JEPA = 'external/jepa-wms'
OFFICIAL = 'external/dinov2_official_7764ea0f912e53c92e82eb78a2a1631e92725fc8'
ENC = '.cache/dinov2/dinov2-small'
ACQUISITION = 'external/jepa_acquisition_manifest.json'
CARD = 'external/jepa_wms_hf/README.md'
META = 'results/p5_jepa_strict_metadata_20261027_v3/report_native_interfaces.json'
PREP = 'results/p5_prospective_c1_cuda_compiler_20261027_v1_ready_main_v2/manifest_frozen.json'
C1 = 'results/p5_prospective_c1_cuda_compiler_20261027_v1_cuda_main_v2/compiler_receipt.json'
EXEC = 'results/p5_native_c1_executor_20261027_v2/manifest_ready_frozen.json'
STATS = JEPA + '/app/plan_common/datasets/pusht_dset.py'
PINS = {
    ACQUISITION: 'fb315951e612567e188558a05ef809f24378397d41d62b79fcd4792ae1dc6597',
    META: '2cb1307e001670c9d89e395a0007926e50fd585a45189bb5a9de284d3be6e911',
    PREP: '68e87734f196d5b1bdd69fa9eb43a12a6a4e02b36b61f2b5e0499a3c18e88dc8',
    C1: '4c3529de98d55d736df5bd7140dbfd9948e05d4cadb0d4699f6d69a352866b0a',
    EXEC: '1099ee3b001ddb9b88fad4d9c31e1fac39fef07b1120501ccee44a907e6c95b3',
}
EDITS = {
    'app/plan_common/datasets/pusht_dset.py': 'decord/OpenCV decoder fallback; NOT certified equivalent by this static audit',
    'app/plan_common/models/dino.py': 'local TorchHub source preference with remote fallback; not invoked here or assumed used by C1',
    'app/plan_common/models/prop_embedding.py': 'getattr defaults for use_mlp/tokens_per_step; this file WAS imported by C1',
    'app/plan_common/models/vit.py': 'getattr use_sdpa fallback; binding alone is not execution evidence',
    'app/vjepa_wm/modelcustom/simu_env_planning/vit_enc_preds.py': 'normalized-zero action history padding; C1 direct predictor must not be conflated with this wrapper',
}
CONFIG_FIELDS = (
    'nodes', 'tasks_per_node', 'data.img_size', 'data.loader.batch_size', 'data.custom.frameskip',
    'data.custom.num_hist', 'data.custom.num_pred', 'data.custom.normalize_action', 'data.custom.split_ratio',
    'meta.freeze_encoder', 'meta.dtype', 'model.grid_size', 'model.tubelet_size_enc',
    'model.visual_encoder.enc_version', 'model.visual_encoder.embed_dim',
    'model.action_encoder.action_encoder_inpred', 'model.proprio_encoder.proprio_emb_dim',
    'model.predictor.pred_type', 'model.predictor.pred_depth', 'model.predictor.pred_embed_dim',
    'model.predictor.pred_num_heads', 'model.rollout_cfg.rollout_steps',
    'optimization.main_optimizer', 'optimization.transition_model.num_epochs',
    'optimization.transition_model.iterations_per_epoch', 'optimization.transition_model.ref_lr', 'evals.obs',
)
SMALL_CAP = 2 * 1024**2
BLOB_CAP = 512 * 1024**2
TOTAL_CAP = 2 * 1024**3
RESERVE = 200 * 1024**2


def require(value, why):
    if not value:
        raise ValueError(why)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def finite(obj):
    if isinstance(obj, float):
        require(math.isfinite(obj), 'nonfinite JSON/literal')
    elif isinstance(obj, dict):
        for v in obj.values():
            finite(v)
    elif isinstance(obj, list):
        for v in obj:
            finite(v)


def decode(raw):
    def unique(pairs):
        out = {}
        for k, v in pairs:
            require(k not in out, 'duplicate JSON key')
            out[k] = v
        return out
    obj = json.loads(raw, object_pairs_hook=unique)
    finite(obj)
    return obj


class Reader:
    def __init__(self, root):
        self.root = Path(root).absolute()
        self.bindings = {}
        self.bytes = 0

    def relative(self, path):
        p = Path(path)
        return p.relative_to(self.root).as_posix() if p.is_absolute() else p.as_posix()

    def path(self, rel):
        p = Path(rel)
        require(not p.is_absolute() and '..' not in p.parts, 'nonlocal path')
        p = self.root / p
        require(not any(q.is_symlink() for q in (p, *p.parents)), 'symlink input')
        require(stat.S_ISREG(p.stat().st_mode), 'nonregular input')
        return p

    def raw(self, rel, expected=None, blob=False):
        p = self.path(rel)
        before = p.stat()
        cap = BLOB_CAP if blob else SMALL_CAP
        require(before.st_size <= cap, 'input byte limit: ' + rel)
        h, parts, size = hashlib.sha256(), [], 0
        with p.open('rb') as stream:
            while True:
                chunk = stream.read(min(1024**2, cap - size + 1))
                if not chunk:
                    break
                size += len(chunk)
                self.bytes += len(chunk)
                require(size <= cap and self.bytes <= TOTAL_CAP, 'read budget')
                h.update(chunk)
                if not blob:
                    parts.append(chunk)
        after = p.stat()
        require((before.st_size, before.st_mtime_ns, before.st_ino) ==
                (after.st_size, after.st_mtime_ns, after.st_ino), 'input changed while hashing')
        hashed = h.hexdigest()
        require(expected is None or hashed == expected, 'SHA mismatch: ' + rel)
        rec = dict(sha256=hashed, bytes=size, scope='full_file_stream_hash_only' if blob else 'full_small_file_read')
        require(rel not in self.bindings or self.bindings[rel] == rec, 'input drift')
        self.bindings[rel] = rec
        return rec if blob else b''.join(parts)

    def read(self, rel, expected=None):
        return decode(self.raw(rel, expected))

    def stable(self):
        for rel, rec in list(self.bindings.items()):
            self.raw(rel, rec['sha256'], blob=rec['scope'] == 'full_file_stream_hash_only')


def selected_config(text):
    """Only selected mapping scalar leaves, NOT a general YAML loader.

    No tags, expressions, interpolation resolution, anchors or constructors.
    Other YAML (including lists) is retained only by its file hash.
    """
    stack, result = [], {}
    for line in text.splitlines():
        s = line.split('#', 1)[0].rstrip()
        if not s.strip() or s.lstrip().startswith('-'):
            continue
        match = re.fullmatch(r'( *)([A-Za-z_][A-Za-z_0-9]*):(?:\s+(.*))?', s)
        if not match:
            continue
        indent, key, value = len(match[1]), match[2], match[3]
        while stack and stack[-1][0] >= indent:
            stack.pop()
        path = '.'.join([x[1] for x in stack] + [key])
        if path in CONFIG_FIELDS:
            require(path not in result, 'duplicate config field: ' + path)
            v = (value or '').strip()
            require(v and v[0] not in '!&*' and '${' not in v, 'unsupported selected config scalar')
            if v in ('true', 'false', 'null'):
                out = {'true': True, 'false': False, 'null': None}[v]
            elif re.fullmatch(r'[-+]?\d+(\.\d*)?([eE][-+]?\d+)?', v):
                out = float(v) if any(x in v for x in '.eE') else int(v)
            else:
                require(re.fullmatch(r'[A-Za-z_][A-Za-z_0-9-]*', v) is not None, 'unsupported selected config string')
                out = v
            finite(out)
            result[path] = out
        stack.append((indent, key))
    require(set(result) == set(CONFIG_FIELDS), 'missing selected config field')
    return result


def nested(config, path):
    for k in path.split('.'):
        config = config[k]
    return config


def stats_literals(text):
    names = {'ACTION_MEAN': 2, 'ACTION_STD': 2, 'PROPRIO_MEAN': 4, 'PROPRIO_STD': 4}
    out = {}
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name not in names:
                continue
            v = node.value
            require(name not in out and isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute)
                    and isinstance(v.func.value, ast.Name) and v.func.value.id == 'torch'
                    and v.func.attr == 'tensor' and len(v.args) == 1 and not v.keywords, 'nonliteral normalizer')
            vals = ast.literal_eval(v.args[0])
            require(isinstance(vals, list) and len(vals) == names[name] and
                    all(type(x) in (float, int) and math.isfinite(x) for x in vals), 'normalizer shape/finite')
            require(not name.endswith('STD') or all(x > 0 for x in vals), 'nonpositive std')
            out[name] = vals
    require(set(out) == set(names), 'missing normalizer')
    return out


def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def project_action(u, mean, std):
    return f32(f32(f32(f32(100. * u) / f32(100.)) - f32(mean)) / f32(std))


def verify_history(receipt, stats):
    c = receipt['action_coordinates']
    for kind in ('mean', 'std'):
        require(c['action_' + kind + '_float32'] == [f32(x) for x in stats['ACTION_' + kind.upper()]], 'action stats receipt mismatch')
    history = c['history_native_u64']
    require(len(history) == 2 and all(len(p) == 5 and all(len(a) == 2 for a in p) for p in history), 'history 2x5x2')
    normalized = [[[project_action(a[i], stats['ACTION_MEAN'][i], stats['ACTION_STD'][i])
                    for i in range(2)] for a in p] for p in history]
    require(normalized == c['history_normalized_f32'], 'history normalization mismatch')
    nominal = [[project_action(0., stats['ACTION_MEAN'][i], stats['ACTION_STD'][i]) for i in range(2)]] * 5
    require(nominal == c['nominal_normalized_f32'] and c['nominal_native_u64'] == [[0., 0.]] * 5, 'native zero/normalized zero mismatch')
    return dict(verified_history_scalars=20, verified_nominal_scalars=10,
                normalized_native_zero=nominal[0], native_zero_is_normalized_zero=False,
                projection=c['dataset_projection'], inverse=c['inverse'])


def git_read(repo, args):
    # Fixed read-only Git operations only; never remote/config/checkout.
    allowed = (args == ['rev-parse', 'HEAD'] or args == ['status', '--porcelain=v1', '-z', '--untracked-files=all']
               or len(args) == 2 and args[0] == 'show' and args[1].startswith('HEAD:') and args[1][5:] in EDITS)
    require(allowed, 'git operation not allowed')
    with subprocess.Popen(['git', '--no-optional-locks', '-c', 'core.fsmonitor=false', '-c', 'core.hooksPath=/dev/null',
                           '-C', str(repo), *args], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as proc:
        raw = proc.stdout.read(SMALL_CAP + 1)
        if len(raw) > SMALL_CAP:
            proc.kill()
            proc.wait()
            raise ValueError('git output cap')
        require(proc.wait(timeout=20) == 0, 'local git read failed')
    return raw


def source_identity(reader, manifest, receipt):
    repo = reader.root / JEPA
    head = git_read(repo, ['rev-parse', 'HEAD']).decode().strip()
    require(re.fullmatch(r'[a-f0-9]{40}', head) is not None, 'invalid revision')
    status = git_read(repo, ['status', '--porcelain=v1', '-z', '--untracked-files=all'])
    entries = status.decode().split('\0')[:-1]
    # Version is deliberately restricted to the five actually inspected edits.
    require({e[3:] for e in entries} == set(EDITS) and all(e[:3] == ' M ' for e in entries), 'local edit census changed; review NEW version')
    records = []
    imports = set(receipt['imported_sources'].values())
    for path, explanation in EDITS.items():
        rel = JEPA + '/' + path
        current = reader.raw(rel, manifest['bindings'][str(reader.root / rel)])
        base = git_read(repo, ['show', 'HEAD:' + path])
        require(sha(base) != sha(current), 'reported modification is unchanged')
        records.append(dict(path=rel, head_blob_sha256=sha(base), current_sha256=sha(current),
                            C1_source_bound=True, C1_module_file_imported=str(reader.root / rel) in imports,
                            static_change_description=explanation, semantic_equivalence_certified=False))
    return dict(local_HEAD=head, tracked_edits=len(records), untracked_files=0,
                status_porcelain_sha256=sha(status), status_scope='tracked and untracked excluding gitignored files',
                edits=records, remote_commit_authentication='unknown; local Git object only')


def metadata_checkpoint(reader, row):
    current = reader.raw(reader.relative(row['path']), row['expected_sha256'], blob=True)
    require(row['status'] == 'exact_source_schema_match', 'saved metadata not matched')
    for rec in row['comparison'].values():
        require(rec['exact_match'] is True and all(rec[k] == [] for k in
                ('missing_keys', 'unexpected_keys', 'shape_mismatches', 'dtype_mismatches')), 'metadata shape comparison failed')
    cfg = row['config']
    shapes = {'.'.join(r['path']): r['shape'] for r in row['nested_metadata']
              if r['kind'] == 'tensor' and r['path'][:1] in (['predictor'], ['proprio_encoder'])}
    pd = row['native_interface']['proprio_dim']
    visual = cfg['model']['visual_encoder']['embed_dim']
    width = visual + cfg['model']['proprio_encoder']['proprio_emb_dim']
    expected = {'predictor.module.predictor_embed.weight': [visual, visual],
                'predictor.module.action_encoder.weight': [width, 10],
                'predictor.module.predictor_blocks.0.attn.qkv.weight': [3 * width, width],
                'predictor.module.predictor_blocks.0.adaLN_modulation.1.weight': [6 * width, width],
                'proprio_encoder.module.patch_embed.weight': [16, pd, 1]}
    require(all(shapes.get(k) == v for k, v in expected.items()), 'saved architecture shape contradiction')
    scalars = {'.'.join(r['path']): r['value'] for r in row['nested_metadata']
               if 'value' in r and (r['path'] == ['epoch'] or r['path'][:1] == ['stats'])}
    return dict(task=row['task'], path=reader.relative(row['path']), **current,
                native_interface=row['native_interface'], saved_meta_shapes=expected,
                saved_checkpoint_counters=scalars,
                tensor_scope='saved restricted-meta report bound to current full weight SHA; no new tensor load',
                official_authentication='unknown')


def verify_source_seal(reader, prep):
    seal = prep['source_seal']
    directory = reader.relative(seal['output_dir'])
    manifest = reader.read(directory + '/manifest.json', seal['manifest_sha256'])
    reader.raw(directory + '/args.json', manifest['args_sha256'])
    require(manifest['schema'] == 'd2p_prerun_source_seal.v1' and
            len(manifest['files']) == seal['source_count'] == 6, 'source seal census')
    total, rows = 0, []
    for i, rec in enumerate(manifest['files']):
        require(rec['archive'] == f'sources/{i:03d}.source', 'source archive ordering')
        raw = reader.raw(directory + '/' + rec['archive'], rec['sha256'])
        require(len(raw) == rec['size'], 'source seal size')
        rel = reader.relative(rec['source'])
        require(reader.raw(rel, rec['sha256']) == raw, 'current/source archive bytes differ')
        total += len(raw)
        rows.append(dict(path=rel, archive=directory + '/' + rec['archive'], sha256=rec['sha256']))
    require(total == seal['source_bytes'], 'source seal byte total')
    return dict(manifest=directory + '/manifest.json', source_count=6, source_bytes=total,
                actual_current_and_archived_bytes_match=True, files=rows)


def unknowns():
    return [dict(id=k, status='unknown', needed_from=who, required_evidence=why) for k, who, why in (
        ('remote_weight_authentication', 'publisher or trusted independent acquisition source',
         'Immutable upstream revisions and weight digests for all three JEPA tasks and encoder; local declarations are not remote authentication.'),
        ('license_usage_clearance', 'human owner/legal reviewer',
         'Confirm intended use complies with local CC-BY-NC-4.0 model/code declarations and third-party terms; not a legal determination.'),
        ('normalizer_statistical_lineage', 'publisher dataset/split/statistics record',
         'Which immutable training dataset revision, episodes/filter and estimator produced source literals; no recomputation from opened outcomes.'),
        ('actual_pretraining_cost', 'publisher run logs/model card with protocol',
         'Training examples/tokens/steps, hardware, GPU hours and energy/FLOPs if claimed. Epoch=50/config topology are not measured compute; zero counters are uninformative.'),
        ('local_edit_equivalence', 'independent approved source/runtime audit',
         'Local changes are fully byte-identified, not all semantically certified. Historical wrapper zero-padding and HF encoder shim cannot be called unmodified release behavior.'),
        ('observation_contract_acceptance', 'main/human predeclared comparison protocol',
         'Accept native RGB+agent xy/velocity proprio and consistent baseline access, or keep explicit different-information scope. Do not label RGB-only or silently remove proprio.'),
    )]


def produce(root=ROOT):
    require(shutil.disk_usage(root).free >= RESERVE, 'reserve below 200 MiB; no deletion authorized')
    reader = Reader(root)
    reader.raw('experiments/spectral_reachability/' + NAME + '.py')
    acq, meta, prep, c1, exe = (reader.read(p, PINS[p]) for p in (ACQUISITION, META, PREP, C1, EXEC))
    require(c1['manifest_file_sha256'] == PINS[PREP] and c1['status'] == 'selected_packet_replayed', 'C1 preparation link')
    require(c1['model_before'] == c1['model_after'] == prep['loader_manifest']['expected_model_receipts'] and
            c1['training'] is False and c1['P5_admission'] is False, 'frozen model identity')
    require(all(r['eval'] is True and r['frozen'] is True for r in c1['model_before'].values()), 'frozen flags')
    require(meta['P5_admission'] is False and meta['model_forward'] is False, 'metadata evidence scope')
    metadata_sources = []
    for rec in meta['sources']:
        rel = reader.relative(rec['path'])
        raw = reader.raw(rel)
        matches = sha(raw) == rec['sha256']
        # The original inspector received a later CUDA entry point. Preserve
        # historical analyzer drift; do not rewrite its old seal or pretend
        # current inspector bytes generated the saved restricted-meta report.
        if rel.startswith(JEPA + '/'):
            require(matches, 'native model/config drift since saved metadata: ' + rel)
        metadata_sources.append(dict(path=rel, saved_sha256=rec['sha256'],
                                     current_sha256=sha(raw), matches_saved=matches,
                                     scope='native_model_or_config' if rel.startswith(JEPA + '/') else 'historical_analysis_source'))
    compiler = c1['compiler_source']
    reader.raw(reader.relative(compiler['path']), compiler['sha256'])
    c1_seal = verify_source_seal(reader, prep)
    # All actual C1 native JEPA + DINOv2 imported source bytes, not generic shims.
    imported = {}
    for module, path in c1['imported_sources'].items():
        rel = reader.relative(path)
        require(rel.startswith((JEPA + '/', OFFICIAL + '/')), 'unexpected C1 native import')
        expected = prep['bindings'].get(path)
        require(expected is not None, 'C1 imported source missing upstream seal')
        reader.raw(rel, expected)
        imported[module] = rel
    source = source_identity(reader, prep, c1)
    git_acq = [r for r in acq['records'] if r['source'] == 'git' and r['path'] == str(reader.root / JEPA)]
    require(len(git_acq) == 1 and git_acq[0]['commit'] == source['local_HEAD'], 'acquisition/source revision')
    source['declared_repository'] = git_acq[0]['url']
    source['C1_imported_module_files'] = imported
    checkpoints = [metadata_checkpoint(reader, r) for r in meta['checkpoints']]
    require([r['task'] for r in checkpoints] == ['pusht', 'pointmaze', 'wall'], 'checkpoint census')
    config_contracts = []
    for row in meta['checkpoints']:
        folder = {'pusht': '/pt_sweep/', 'pointmaze': '/mz_sweep/', 'wall': '/wall_sweep/'}[row['task']]
        paths = [r['path'] for r in meta['sources'] if r['path'].endswith('.yaml') and folder in r['path']]
        require(len(paths) == 1, 'unique per-task source config')
        path = paths[0]
        rel = reader.relative(path)
        selected = selected_config(reader.raw(rel).decode())
        require(all(selected[k] == nested(row['config'], k) and type(selected[k]) is type(nested(row['config'], k))
                    for k in CONFIG_FIELDS), 'current/saved config mismatch')
        config_contracts.append(dict(task=row['task'], path=rel, selected_fields=selected,
                                     selected_scalar_scope_only=True, training_cost_measured=False))
    # Asset source claims are copied only after matching current bytes and card.
    card_record = next(r for r in acq['records'] if r.get('filename') == 'README.md' and r.get('repo_id') == 'facebook/jepa-wms')
    card = reader.raw(CARD, card_record['sha256']).decode()
    license_text = reader.raw(JEPA + '/LICENSE').decode()
    third = reader.raw(JEPA + '/THIRD-PARTY-LICENSES.md').decode()
    reader.raw(JEPA + '/README.md')
    require('license: cc-by-nc-4.0' in card and 'Attribution-NonCommercial 4.0 International' in license_text
            and 'CC-BY-NC 4.0' in third, 'license declaration mismatch')
    dino_license = reader.raw(OFFICIAL + '/LICENSE').decode()
    require('Apache License' in dino_license and 'Version 2.0' in dino_license, 'encoder code license')
    claims = [r for r in acq['records'] if r.get('filename') == 'jepa_wm_pusht.pth.tar']
    require(len(claims) == 1 and claims[0]['sha256'] == checkpoints[0]['sha256'], 'PushT acquisition SHA')
    source_claims = dict(pusht=dict(repo_id=claims[0]['repo_id'], filename=claims[0]['filename'],
                        matches_local_acquisition=True, claimed_revision=None),
                        other_tasks='local card advertises names; no task-specific immutable acquisition revision authenticated',
                        remote_access_performed=False, official_authenticated=False)
    encoder_config = reader.read(ENC + '/config.json', prep['bindings'][str(reader.root / ENC / 'config.json')])
    encoder_weight = reader.raw(ENC + '/model.safetensors', prep['bindings'][str(reader.root / ENC / 'model.safetensors')], blob=True)
    hf = reader.raw(ENC + '/.cache/huggingface/download/model.safetensors.metadata').decode().splitlines()
    require(len(hf) == 3 and re.fullmatch('[a-f0-9]{40}', hf[0]) and hf[1] == encoder_weight['sha256'], 'cached HF revision/ETag record')
    encoder = dict(path=ENC + '/model.safetensors', **encoder_weight, config=encoder_config,
                   cached_metadata_claim=dict(revision=hf[0], etag_sha256=hf[1], remote_authenticated=False),
                   source_revision_declared_by_directory=OFFICIAL.rsplit('_', 1)[1],
                   weights_scope='cached HF tensors mapped to official source implementation; not independently authenticated official encoder weights')
    stats = stats_literals(reader.raw(STATS, prep['bindings'][str(reader.root / STATS)]).decode())
    require(stats == prep['stats'], 'source/manifest normalizer mismatch')
    normalization = verify_history(c1, stats)
    normalization.update(source_path=STATS, literals=stats, statistical_training_split_lineage='unknown',
                         source_comment='precomputed dataset stats; estimator/dataset revision not specified')
    env = exe['native_environment_source']
    envtext = reader.raw(reader.relative(env['path']), env['sha256']).decode()
    require('proprio = state[:2]' in envtext and 'proprio = np.concatenate((proprio, state[5:]))' in envtext,
            'native proprio extraction changed')
    contract = exe['contract']
    require(contract['context_frames'] == [0, 5, 10] and exe['native_constructor']['with_velocity'] is True,
            'native history/proprio config')
    observation = dict(planner_inputs=['RGB', 'agent_x', 'agent_y', 'agent_vx', 'agent_vy'],
                       RGB_only=False, agent_proprio_privileged_relative_to_RGB_only=True,
                       object_pose_used_as_planner_proprio=False, object_pose_scope='evaluator/goal construction, not these four proprio inputs',
                       environment_source=reader.relative(env['path']), constructor=exe['native_constructor'],
                       history_frames=contract['context_frames'], history_between_action_packets=2,
                       outgoing_packet_shape=[5, 2], native_support=contract['native_action_support'],
                       native_support_scope=contract['support_scope'],
                       RGB_preprocess='C1 full-frame 224, RGB/255, ImageNet mean [.485,.456,.406]/std [.229,.224,.225]; no crop augmentation',
                       training_augmentation_is_not_C1_preprocess=True,
                       protocol_acceptance='human/main pending; different information from RGB-only LeWM must be disclosed')
    header_licenses = {}
    for rel in (STATS, JEPA + '/app/plan_common/models/dino.py', reader.relative(env['path'])):
        require('Licensed under the MIT License' in '\n'.join(reader.raw(rel).decode().splitlines()[:6]),
                'file-specific license header changed')
        header_licenses[rel] = 'MIT header declaration; preserve alongside main CC-BY-NC-4.0 and third-party terms'
    reader.stable()
    require(git_read(reader.root / JEPA, ['rev-parse', 'HEAD']).decode().strip() == source['local_HEAD'] and
            sha(git_read(reader.root / JEPA, ['status', '--porcelain=v1', '-z', '--untracked-files=all'])) == source['status_porcelain_sha256'],
            'source revision/status drift')
    return dict(schema=NAME, status='local_architecture_contract_verified_with_explicit_unknowns',
                official_authenticated=False, architecture_contract_complete=False,
                P1_admission=False, P5_admission=False, execution_authorized=False,
                new_neural_calls=0, new_physical_steps=0, training=False,
                source_identity=source, architecture=dict(family='JEPA-WM AdaLN action-conditioned dynamics',
                    visual_dim=384, proprio_embedding_dim=16, block_width=400, predictor_depth=6,
                    primitive_action_dim=2, packed_action_dim=10, architecture_task_weights_not_distinct_architectures=3,
                    proof_scope='current source/config hashes + saved restricted-meta shapes + actual C1 strict-load receipts',
                    C1_strict_loads=c1['strict_loads'], C1_frozen_state_identities=c1['model_before']),
                checkpoints=checkpoints, config_contracts=config_contracts, weights_source_claims=source_claims,
                metadata_source_lineage=metadata_sources,
                C1_source_seal=c1_seal,
                licenses=dict(JEPA_code='CC-BY-NC-4.0 local LICENSE', JEPA_model_card='CC-BY-NC-4.0 local card declaration',
                    encoder_source='Apache-2.0 local DINOv2 LICENSE', encoder_weight_license='unknown: no cached model card/license authenticated',
                    third_party=JEPA + '/THIRD-PARTY-LICENSES.md', unrestricted_commercial_permission=False,
                    file_specific_headers=header_licenses, legal_clearance=False), encoder=encoder, normalization=normalization,
                observation_history=observation,
                learning_cost=dict(existing_pretraining_required=True, extra_training_this_audit=False,
                    C1_model_training=False, source_optimizer='transition_model',
                    checkpoint_epoch_source='saved metadata; each current weight full-SHA matched',
                    zero_training_counters_are_not_zero_cost=True, actual_GPU_hours=None, actual_FLOPs=None,
                    config_topology_is_not_measured_compute=True, encoder_pretraining_dataset_and_compute='unknown in bound local materials'),
                unknowns=unknowns(), bindings=reader.bindings, bytes_read=reader.bytes,
                limits=['Not a scientific or execution admission gate', 'No remote authentication or legal opinion',
                        'No tensor load/unpickle/new runtime equivalence; no dataset statistics recomputation',
                        'Git status excludes ignored files; current C1 imported source files explicitly SHA checked',
                        'Only listed C1/model/interface source closure, not entire environment/binary provenance'])


def verify(saved, current):
    canonical = lambda x: json.dumps(x, sort_keys=True, ensure_ascii=False, allow_nan=False)
    require(canonical(saved) == canonical(current), 'contract stale/modified; review NEW snapshot')
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-contract', help='existing project-body-relative same-stem JSON, read-only')
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args(argv)
    try:
        result = produce()
        if args.verify_contract:
            require(Path(args.verify_contract).name == NAME + '.json', 'wrong contract stem')
            verify(Reader(ROOT).read(args.verify_contract), result)
        encoded = json.dumps(result, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)
        require(len(encoded.encode()) < 512 * 1024, 'contract output cap')
        print(encoded)
        return 3 if args.require_complete else 0
    except (ValueError, OSError, KeyError, TypeError, IndexError, OverflowError, RecursionError, subprocess.TimeoutExpired) as exc:
        print(json.dumps(dict(schema=NAME, status='invalid_evidence', official_authenticated=False,
                              execution_authorized=False, error=str(exc)), ensure_ascii=False))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
