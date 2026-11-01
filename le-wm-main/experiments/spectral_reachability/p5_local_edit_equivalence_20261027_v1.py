"""Offline AST/symbol-array contracts, NOT neural or official equivalence.

stdout only; 0=bounded engineering checks passed, 2=invalid/drift/reserve,
3=--require-full-equivalence (currently intentionally unavailable).
No target source execution, checkpoint deserialization or environment lookup.
"""
import argparse
import ast
import copy
import json
import math
from pathlib import Path
import shutil

import p5_architecture_provenance_20261027_v1 as provenance

NAME = 'p5_local_edit_equivalence_20261027_v1'
ROOT = Path(__file__).resolve().parents[2]
PROV = 'results/p5_architecture_provenance_20261027_v1.json'
PROV_SHA = '6c169f0685bb227913ae6deb821d74093e15eed29f063584e919bae892cde7b0'
PROV_CODE = 'experiments/spectral_reachability/p5_architecture_provenance_20261027_v1.py'
PROV_CODE_SHA = 'fda1d8d818e4f38a83f956a65ba10117fcbdff0ccf72bd79e9f1d8c14e22b3cd'
STRICT = 'results/p5_jepa_strict_metadata_20261027_v3/'
RECEIPTS = {
    'pusht': 'c1498ecc867523b7b10cf9028543151e402ce593b09ac8a85bca58b55cb450e0',
    'pointmaze': 'c830f3c00b307c307bc3347769997b848152985243bf1a7217473a9366651694',
    'wall': '7ad42328fe6ce8a60fdac35bdcf3bda52b66122f8af3f6e9a35125cf8dda02b9',
}
ENC_PAIR = 'results/p5_jepa_official_equivalence_20261027_v6/pusht_official_vs_hf_cuda_main_v1.json'
ENC_SHA = '78d5e5d8584e6dc78efe4e7b70c2a7d64e1247d84494d929afc07f61d8343b41'
need = provenance.require


def canonical(node):
    return ast.dump(node, include_attributes=False)


def expression(text):
    return ast.parse(text, mode='eval').body


def member(tree, cls, method):
    c = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    return next(n for n in c.body if isinstance(n, ast.FunctionDef) and n.name == method)


class Defaults(ast.NodeTransformer):
    def __init__(self, allowed):
        self.allowed, self.seen = allowed, []

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == 'getattr':
            for attr, default in self.allowed.items():
                if canonical(node) == canonical(expression(f'getattr(self, "{attr}", {default!r})')):
                    self.seen.append(attr)
                    return expression('self.' + attr)
        return node


def prove_defaults(base, current, cls, allowed):
    old, new = ast.parse(base), ast.parse(current)
    init = member(new, cls, '__init__')
    # Unconditional direct assignments, not merely any nested assignment.
    for attr in allowed:
        need(any(isinstance(n, ast.Assign) and
                 any(isinstance(t, ast.Attribute) and t.attr == attr and
                     isinstance(t.value, ast.Name) and t.value.id == 'self' for t in n.targets) and
                 canonical(n.value) == canonical(expression(attr))
                 for n in init.body), 'constructor attribute not unconditionally initialized')
    transform = Defaults(allowed)
    projected = transform.visit(copy.deepcopy(new))
    need(sorted(transform.seen) == sorted(allowed), 'getattr edit count/shape')
    need(canonical(old) == canonical(projected), 'non-getattr source change')
    return dict(projected_full_AST_equal=True, substitutions=transform.seen,
                condition='ordinary constructor-initialized instance; attributes retained; same runtime and inputs',
                missing_attribute='old AttributeError vs new default: NOT equivalent',
                numeric_kernel_pair_executed=False)


class DatasetBranch(ast.NodeTransformer):
    """Project ONLY the extant decord path, leaving all other AST unchanged."""
    def visit_Try(self, node):
        if canonical(node.body[0]) == canonical(ast.parse('import decord').body[0]):
            return node.body
        return self.generic_visit(node)

    def visit_If(self, node):
        if canonical(node.test) in {
                canonical(expression('decord is not None')),
                canonical(expression('reader is not None'))}:
            return node.body
        return self.generic_visit(node)


def prove_dataset(base, current):
    projected = DatasetBranch().visit(ast.parse(current))
    need(canonical(projected) == canonical(ast.parse(base)), 'decord branch AST mismatch')
    # Symbolic byte/channel/layout identities, NOT any codec execution.
    count = 0
    for t in range(2):
        for h in range(3):
            for w in range(4):
                rgb = [((t * 97 + h * 31 + w * 7 + c * 53) % 256) for c in range(3)]
                bgr = rgb[::-1]
                for c in range(3):
                    need(bgr[2-c] / 255.0 == rgb[c] / 255.0, 'channel scaling')
                    count += 1
    return dict(retained_decord_full_AST_equal=True, symbolic_channel_checks=count,
                condition='same decoded uint8 RGB bytes and frame order, same dtype/transform',
                unresolved='actual decord/OpenCV frame seek, color, codec and dtype comparison; decoder CPU test sufficient',
                CUDA_required='downstream encoder/predictor/derivative parity, not byte decoding itself')


def prove_dino(base, current):
    tree = ast.parse(current)
    tree.body = [n for n in tree.body if not (
        isinstance(n, ast.ImportFrom) and n.module == 'pathlib')]
    init = member(tree, 'DinoEncoder', '__init__')
    branch = next(n for n in init.body if isinstance(n, ast.If))
    need(len(branch.body) == 3 and isinstance(branch.body[-1], ast.If), 'loader edit structure')
    need(canonical(branch.body[-1].test) == canonical(expression('local_source.exists()')), 'local predicate')
    branch.body = branch.body[-1].orelse
    need(canonical(tree) == canonical(ast.parse(base)), 'absent-local fallback AST mismatch')
    return dict(absent_local_full_AST_equal=True,
                condition='local_source absent; no claim of safe/offline execution of original remote fallback',
                extant_default='external/local_dinov2/hubconf.py; HuggingFace shim, NOT same encoder implementation',
                unresolved='frozen same-input CUDA feature/output/JVP pair and source selection receipt',
                runtime_environment_override='not read or certified')


def prove_wrapper(base, current):
    tree = ast.parse(current)
    unroll = member(tree, 'EncPredWM', 'unroll')
    loops = [n for n in ast.walk(unroll) if isinstance(n, ast.For)]
    matched = 0
    for loop in loops:
        for i, node in enumerate(loop.body):
            if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == 'context_len' for t in node.targets):
                need(canonical(node.value) == canonical(
                    expression('vid_feats[:, -self.ctxt_window:].shape[1]')), 'context expression')
                pad_if = loop.body[i+1]
                need(isinstance(pad_if, ast.If) and canonical(pad_if.test) == canonical(
                    expression('act_feats.shape[1] < context_len')), 'padding predicate')
                expected_pad = ast.parse(
                    'if act_feats.shape[1] < context_len:\n'
                    '    pad_len = context_len - act_feats.shape[1]\n'
                    '    pad = torch.zeros(act_feats.shape[0], pad_len, *act_feats.shape[2:], '
                    'device=act_feats.device, dtype=act_feats.dtype)\n'
                    '    act_feats_for_pred = torch.cat([pad, act_feats], dim=1)\n'
                    'else:\n'
                    '    act_feats_for_pred = act_feats[:, -context_len:]\n').body[0]
                need(canonical(pad_if) == canonical(expected_pad), 'padding operation changed')
                # The complete patch is separately SHA pinned. Projection only
                # proves unchanged residual program, not equivalence of deletion.
                del loop.body[i:i+2]
                matched += 1
                break
    need(matched == 1, 'wrapper insertion count')
    class Argument(ast.NodeTransformer):
        count = 0

        def visit_Name(self, node):
            if node.id == 'act_feats_for_pred':
                self.count += 1
                return expression('act_feats[:, -self.ctxt_window:]')
            return node
    transform = Argument()
    projected = transform.visit(tree)
    need(transform.count == 1 and canonical(projected) == canonical(ast.parse(base)), 'wrapper residual AST')
    return dict(residual_full_AST_equal=True,
                condition='min(action_count, window) == min(visual_count, window), positive lengths',
                shortage='new normalized-zero tokens replace absent slots; original shorter sequence NOT equivalent',
                excess='if visual_count < window and action_count > visual_count, new truncation also differs',
                unresolved='frozen real-history vs zero-history model outputs/JVP and downstream decisions on CUDA')


def action_slices(visual, actions, window):
    need(all(type(n) is int and n > 0 for n in (visual, actions, window)), 'positive integer lengths')
    tokens = list(range(actions))
    context = min(visual, window)
    old = tokens[-window:]
    new = [None] * (context-actions) + tokens if actions < context else tokens[-context:]
    return old, new


def enumerate_slices():
    counts = dict(equal=0, zero_padding_change=0, truncation_change=0)
    examples = {}
    for v in range(1, 7):
        for a in range(1, 7):
            for w in range(1, 7):
                old, new = action_slices(v, a, w)
                equal = old == new
                need(equal == (min(a, w) == min(v, w)), 'slice equality criterion')
                kind = 'equal' if equal else 'zero_padding_change' if a < min(v, w) else 'truncation_change'
                counts[kind] += 1
                examples.setdefault(kind, dict(visual=v, actions=a, window=w, original=old, patched=new))
    return dict(scope='216 finite symbol-index cases, no learned arithmetic', counts=counts, examples=examples)


def packet_indices(steps=5, dims=2):
    need(type(steps) is int and type(dims) is int and steps > 0 and dims > 0, 'packet dimensions')
    packet = [[f't{t}d{d}' for d in range(dims)] for t in range(steps)]
    flat = [v for step in packet for v in step]
    need(all(flat[t*dims+d] == packet[t][d] for t in range(steps) for d in range(dims)), 'packing order')
    return flat


def state_schema(proprio_dim):
    need(proprio_dim in (2, 4), 'native proprio dimension')
    pred = {}
    def linear(prefix, out_dim, in_dim):
        pred[prefix + '.weight'] = [out_dim, in_dim]
        pred[prefix + '.bias'] = [out_dim]
    def norm(prefix):
        pred[prefix + '.weight'] = [400]
        pred[prefix + '.bias'] = [400]
    linear('predictor_embed', 384, 384)
    linear('action_encoder', 400, 10)
    for i in range(6):
        prefix = f'predictor_blocks.{i}.'
        norm(prefix + 'norm1')
        norm(prefix + 'norm2')
        for name, out_dim, in_dim in [
            ('attn.qkv', 1200, 400), ('attn.proj', 400, 400),
            ('mlp.fc1', 1600, 400), ('mlp.fc2', 400, 1600),
            ('adaLN_modulation.1', 2400, 400)]:
            linear(prefix + name, out_dim, in_dim)
    norm('predictor_norm')
    linear('predictor_proj', 384, 384)
    return dict(predictor=pred, proprio_encoder={
        'patch_embed.weight': [16, proprio_dim, 1], 'patch_embed.bias': [16]})


def compare_metadata(records, expected):
    actual = {part: {} for part in expected}
    for row in records:
        path = row['path']
        if row['kind'] != 'tensor' or not path or path[0] not in expected:
            continue
        need(len(path) == 2 and isinstance(path[1], str) and path[1].startswith('module.'), 'state key prefix')
        key = path[1][len('module.'):]
        need(key not in actual[path[0]], 'duplicate state key')
        need(row['dtype'] == 'torch.float32', 'state dtype')
        actual[path[0]][key] = row['shape']
    need(actual == expected, 'full state schema mismatch')
    return {part: dict(keys=len(schema), numel=sum(math.prod(s) for s in schema.values()),
                       schema_sha256=provenance.sha(json.dumps(schema, sort_keys=True).encode()))
            for part, schema in expected.items()}


def constructor_contract(source):
    """Recognize the saved strict-loader constructor without importing it."""
    tree = ast.parse(source)
    dims = [n for n in tree.body if isinstance(n, ast.Assign) and
            any(isinstance(t, ast.Name) and t.id == 'PROPRIO_DIMS' for t in n.targets)]
    need(len(dims) == 1 and ast.literal_eval(dims[0].value) ==
         {'pusht': 4, 'pointmaze': 4, 'wall': 2}, 'task dimension source')
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'validate_cuda')
    specs = {
        'vit_predictor_AdaLN': dict(img_size=224, patch_size=14, num_frames=4, tubelet_size=1,
            embed_dim=384, predictor_embed_dim=384, depth=6, num_heads=16,
            use_silu=False, use_rope=True, local_window=(3, -1, -1), action_dim=10,
            act_mlp=False, prop_mlp=False, proprio_encoder_inpred=False, action_encoder_inpred=True,
            proprio_encoding='feature', proprio_emb_dim=16, proprio_tokens=0),
        'ProprioceptiveEmbedding': dict(num_frames=4, tubelet_size=1, tokens_per_step=1,
            embed_dim=16, shift_input=False, use_mlp=False),
    }
    for name, spec in specs.items():
        calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and
                 isinstance(n.func, ast.Name) and n.func.id == name]
        need(len(calls) == 1 and not calls[0].args, 'constructor call count')
        kws = {k.arg: k.value for k in calls[0].keywords}
        dimension = 'proprio_dim' if name == 'vit_predictor_AdaLN' else 'in_chans'
        need(set(kws) == set(spec) | {dimension}, 'constructor keyword contract')
        need(canonical(kws[dimension]) == canonical(expression('PROPRIO_DIMS[task]')), 'task-specific input')
        need(all(canonical(kws[k]) == canonical(expression(repr(v))) for k, v in spec.items()),
             'constructor scalar contract')
    return dict(AST_constructor_keywords_verified=True, task_proprio_dims={'pusht': 4, 'pointmaze': 4, 'wall': 2},
                predictor_feature_width=384+16, scope='source only; instantiated by saved CUDA strict-load receipt')


def check_receipt(receipt, task, weight_sha, schemas):
    need(receipt['task'] == task and receipt['status'] == 'native_dynamics_strict_load_only', 'receipt identity')
    need(receipt['cuda_executed'] is True and receipt['model_forward'] is False and
         receipt['encoder_runtime_verified'] is False and receipt['P5_admission'] is False, 'receipt scope')
    need(receipt['checkpoint_sha256_before'] == receipt['checkpoint_sha256_after'] == weight_sha, 'weight receipt')
    need(receipt['sources_before'] == receipt['sources_after'], 'source receipt drift')
    for part, schema in schemas.items():
        rec, comp = receipt['receipts'][part], receipt['comparison'][part]
        need(comp['exact_match'] is True and all(not comp[k] for k in (
            'missing_keys', 'unexpected_keys', 'shape_mismatches', 'dtype_mismatches')), 'strict schema receipt')
        need(rec['state_tensor_count'] == comp['actual_key_count'] == comp['expected_key_count'] == schema['keys'],
             'state count receipt')
        need(rec['parameter_numel'] == schema['numel'], 'numel receipt')
        need(all(rec[k] is True for k in ('strict_load', 'all_state_tensors_cuda',
                                        'all_parameters_frozen', 'all_submodules_eval')), 'load receipt flags')


def produce(root=ROOT):
    need(shutil.disk_usage(root).free >= 200 * 1024**2, 'reserve below 200 MiB')
    reader = provenance.Reader(root)
    reader.raw(PROV_CODE, PROV_CODE_SHA)
    prov = reader.read(PROV, PROV_SHA)
    reader.raw('experiments/spectral_reachability/' + NAME + '.py')
    reader.raw('tests/test_' + NAME + '.py')
    identity = prov['source_identity']
    repo = Path(root) / provenance.JEPA
    need(provenance.git_read(repo, ['rev-parse', 'HEAD']).decode().strip() == identity['local_HEAD'], 'HEAD drift')
    status = provenance.git_read(repo, ['status', '--porcelain=v1', '-z', '--untracked-files=all'])
    need(provenance.sha(status) == identity['status_porcelain_sha256'], 'edit census drift')
    proofs = []
    for edit in identity['edits']:
        rel = edit['path']
        local = reader.raw(rel, edit['current_sha256']).decode()
        original = provenance.git_read(repo, ['show', 'HEAD:' + rel[len(provenance.JEPA)+1:]])
        need(provenance.sha(original) == edit['head_blob_sha256'], 'HEAD blob drift')
        if rel.endswith('prop_embedding.py'):
            proof = prove_defaults(original, local, 'ProprioceptiveEmbedding', {'use_mlp': False, 'tokens_per_step': 1})
        elif rel.endswith('/vit.py'):
            proof = prove_defaults(original, local, 'Attention', {'use_sdpa': False})
        elif rel.endswith('pusht_dset.py'):
            proof = prove_dataset(original, local)
        elif rel.endswith('/dino.py'):
            proof = prove_dino(original, local)
        else:
            proof = prove_wrapper(original, local)
        proofs.append(dict(path=rel, base_sha256=edit['head_blob_sha256'],
                           current_sha256=edit['current_sha256'],
                           C1_imported=edit['C1_module_file_imported'], contract=proof))
    # Bind actual sources/configs used in already-executed strict load receipts.
    meta = reader.read(provenance.META, provenance.PINS[provenance.META])
    checkpoints = []
    for weight in prov['checkpoints']:
        task = weight['task']
        reader.raw(weight['path'], weight['sha256'], blob=True)
        rows = [r for r in meta['checkpoints'] if r['task'] == task]
        need(len(rows) == 1, 'task metadata uniqueness')
        schemas = compare_metadata(rows[0]['nested_metadata'], state_schema(weight['native_interface']['proprio_dim']))
        receipt = reader.read(STRICT + task + '_native_strict_cuda_main_v1.json', RECEIPTS[task])
        check_receipt(receipt, task, weight['sha256'], schemas)
        for source in receipt['sources_before']:
            reader.raw(reader.relative(source['path']), source['sha256'])
        checkpoints.append(dict(task=task, sha256=weight['sha256'], state_contract=schemas,
                                primitive_action_dim=2, packet_length=5, packed_action_dim=10,
                                proprio_dim=weight['native_interface']['proprio_dim'],
                                saved_actual_CUDA='strict load only; no forward or encoder verification'))
    # Bind native construction/config code; shapes alone are not runtime equivalence.
    for rel, binding in prov['bindings'].items():
        if rel.endswith(('.yaml', '/AdaLN_vit.py', '/src/models/utils/modules.py',
                         '/video_wm.py', '/local_dinov2/hubconf.py')):
            reader.raw(rel, binding['sha256'])
    video = provenance.JEPA + '/app/vjepa_wm/video_wm.py'
    strict_code = 'experiments/spectral_reachability/p5_jepa_strict_metadata_20261027_v3.py'
    constructor = constructor_contract(reader.raw(strict_code).decode())
    shim = reader.raw('external/local_dinov2/hubconf.py').decode()
    shim_ast = ast.parse(shim)
    need(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and
             n.func.attr == 'from_pretrained' and any(
                 k.arg == 'local_files_only' and isinstance(k.value, ast.Constant) and k.value.value is True
                 for k in n.keywords) for n in ast.walk(shim_ast)), 'local shim source contract')
    video_ast = ast.parse(reader.raw(video).decode())
    encode = next(n for n in ast.walk(video_ast) if isinstance(n, ast.FunctionDef) and n.name == 'encode_act')
    identity_returns = [n for n in encode.body if isinstance(n, ast.If) and
                        canonical(n.test) == canonical(expression('self.action_encoder_inpred'))]
    need(len(identity_returns) == 1 and canonical(identity_returns[0].body[0]) ==
         canonical(ast.parse('return a').body[0]), 'action encode pass-through branch')
    c1 = reader.read(provenance.C1, provenance.PINS[provenance.C1])
    stats = provenance.stats_literals(reader.raw(provenance.STATS).decode())
    provenance.verify_history(c1, stats)
    history = c1['action_coordinates']['history_normalized_f32']
    vals = [x for packet in history for step in packet for x in step]
    need(len(vals) == 20, 'C1 two historical 5x2 packets')
    pair = reader.read(ENC_PAIR, ENC_SHA)
    need(pair['status'] == 'paired_equivalence_failed' and len(pair['rows']) == 8, 'saved encoder pair scope')
    need(all(r['passed'] is False for r in pair['rows']), 'encoder pair row scope')
    reader.stable()
    need(shutil.disk_usage(root).free >= 200 * 1024**2, 'ending reserve below 200 MiB')
    return dict(schema=NAME, status='bounded_static_contracts_verified_NOT_global_equivalence',
                evidence_class='actual local source/SHA and saved CUDA receipts; symbol arrays are contract fixtures',
                edits=proofs, checkpoints=checkpoints, slice_contract=enumerate_slices(),
                strict_loader_constructor=constructor,
                packet_contract=dict(order='time-major then component', symbols=packet_indices(),
                                     scope='5x2 -> 10 reshape, no policy/model evaluation'),
                C1_history=dict(normalized_values=len(vals), nonzero_values=sum(x != 0 for x in vals),
                                zero_padding_max_abs_input_delta=max(abs(x) for x in vals),
                                saved_f32_projection_recomputed=True,
                                interpretation='input mismatch only; no output harm or theoretical failure inferred'),
                encoder_saved_actual=dict(path=ENC_PAIR, status=pair['status'], episodes=8,
                                          scope='historical official-vs-HF CUDA feature pair, not a new run'),
                full_equivalence=False, official_authenticated=False, P5_admission=False,
                execution_authorized=False, new_neural_calls=0, new_physical_steps=0,
                remaining=['real decoder pair with fixed RGB/frame/dtype protocol',
                           'same-input CUDA encoder/predictor/output/JVP pairs for the selected three task interfaces',
                           'frozen explicit true-history contract vs normalized-zero wrapper; do not tune acceptance',
                           'remote/human weight authentication and provenance unknowns remain inherited',
                           'P1/independent roots/strong local paired baselines remain outside this code contract'],
                bindings=reader.bindings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', help='compare saved report against fresh local recomputation')
    parser.add_argument('--require-full-equivalence', action='store_true')
    args = parser.parse_args()
    try:
        report = produce()
        if args.verify:
            reader = provenance.Reader(ROOT)
            saved = reader.read(reader.relative(args.verify))
            need(saved == report, 'saved report differs from fresh contract')
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        return 3 if args.require_full_equivalence else 0
    except (ValueError, OSError, KeyError, StopIteration, SyntaxError) as exc:
        print(json.dumps(dict(schema=NAME, status='invalid', error=str(exc))))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
