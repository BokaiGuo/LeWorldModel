"""Frozen known-root response localization; prepare is non-neural, run is CUDA-only.

No simulator, Jacobian, fitting, action selection, training or P1 admission.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np

TAG = 'pusht_geometry_feedback_response_20261027_v1'
ROLES = ('selected', 'greedy', 'zero')
ARMS = ('imagined', 'actual')
ROOT = '23b8de7837c39fb4e0fcffe7678732a9d840c57de7a0c8376e22076770ef6050'
REPO = Path(__file__).resolve().parents[2]
DEFAULT_CAPABILITY = REPO/'results/pusht_geometry_capability_20261027_v1_cuda_v1/report.json'
DEFAULT_RGB = REPO/'results/p4_feedback_matched_20261027_v2_cuda_main_v1/root_1.actual.round_0.rgb.npy'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def array(value, shape):
    a = np.asarray(value)
    if a.dtype.kind not in 'fiu' or a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f'finite real array {shape} required')
    return a.astype(float)


def require_close(a, b, label, atol=2e-5):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all() or not np.allclose(a, b, rtol=0, atol=atol):
        raise ValueError(label+' mismatch')
    return float(np.max(np.abs(a-b)))


def pair_contexts(imagined, actual):
    """Test reencoding against history, then share EXACT older latent bytes."""
    for c in (imagined, actual):
        array(c['latent_history'], (3,192))
        array(c['action_history'], (2,10))
    delta = require_close(np.asarray(actual['latent_history'])[:2],
                          np.asarray(imagined['latent_history'])[:2], 'reencoded old latents')
    if actual['action_history'] != imagined['action_history']:
        raise ValueError('normalized incoming packet history mismatch')
    actual = deepcopy(actual)
    actual['latent_history'][:2] = deepcopy(imagined['latent_history'][:2])
    return actual, delta


def central_fd(vectors, bank, q):
    if set(vectors) != {b['probe_id'] for b in bank} or len(vectors) != 41:
        raise ValueError('exact unique 41 suffix slots required')
    q = array(q, (192,192))
    values = {k:array(v, (192,)) for k,v in vectors.items()}
    matrices = []
    for epsilon in (.05,.1):
        columns = []
        for axis in range(10):
            ids = [next(b['probe_id'] for b in bank if b['epsilon'] == epsilon and b['axis'] == axis and b['sign'] == sign) for sign in (-1,1)]
            columns.append(q @ (values[ids[1]]-values[ids[0]])/(2*epsilon))
        matrices.append(np.stack(columns, axis=1))
    return matrices


def matrix_comparison(a, b):
    denominator = float(np.linalg.norm(a)*np.linalg.norm(b))
    return dict(cosine=None if denominator == 0 else float(np.sum(a*b)/denominator),
                predicted_frobenius=float(np.linalg.norm(a)), real_frobenius=float(np.linalg.norm(b)),
                frobenius_error=float(np.linalg.norm(a-b)),
                relative_error=None if np.linalg.norm(b) == 0 else float(np.linalg.norm(a-b)/np.linalg.norm(b)))


def analyze(predicted, real, bank, q, goal):
    """Each nonzero probe uses its OWN physical terminal, never the zero slot."""
    pmat, rmat = central_fd(predicted, bank, q), central_fd(real, bank, q)
    q, goal = array(q, (192,192)), array(goal, (192,))
    probes = []
    for b in bank:
        key = b['probe_id']; p, r = array(predicted[key], (192,)), array(real[key], (192,))
        error = q @ (p-r)
        probes.append(dict(probe_id=key, predicted=p.tolist(), physical=r.tolist(),
            projected_error=error.tolist(), projected_mse=float(np.mean(error**2)),
            predicted_goal_residual=(q@(goal-p)).tolist(), physical_goal_residual=(q@(goal-r)).tolist()))
    return dict(probes=probes, zero_prediction_residual=probes[0], grids=[dict(epsilon=e,
        predicted_K=a.tolist(), real_K=b.tolist(), **matrix_comparison(a,b))
        for e,a,b in zip((.05,.1),pmat,rmat)],
        epsilon_sensitivity=dict(predicted_frobenius_delta=float(np.linalg.norm(pmat[1]-pmat[0])),
                                 real_frobenius_delta=float(np.linalg.norm(rmat[1]-rmat[0]))))


def prepare(capability=DEFAULT_CAPABILITY, root_rgb=DEFAULT_RGB):
    from experiments.spectral_reachability import pusht_geometry_capability_20261027_v1 as cap
    from experiments.spectral_reachability.d2p_prerun_source_seal import verify_seal
    from PIL import Image
    path = Path(capability).resolve(); report = read(path); old = read(path.parent/'protocol.json')
    if (report['schema'] != cap.TAG or report['status'] != 'diagnostic_recorded'
            or not report['gates'] or not all(v is True for v in report['gates'].values())
            or report['protocol_sha256'] != sha(path.parent/'protocol.json')
            or report['physical_env_step_returns'] != 1230 or report['attempted_encode_samples'] != 246):
        raise ValueError('complete sealed original capability required')
    for seal in report['source_seals']:
        verify_seal(output_dir=seal['output_dir'], expected_manifest_sha256=seal['manifest_sha256'])
    context = cap.prepare(old['planning_report'])
    base, checked, source, config, bindings, plans, upstream, refs, protocol = context
    cap.census(plans, report['rows'])
    if plans != old['plans'] or report['census']['qualified_roots'] != [ROOT]:
        raise ValueError('original single qualified root and frozen plans required')
    bindings = dict(bindings)
    def bind(p, expected=None):
        p = Path(p).resolve(); value = sha(p)
        if expected is not None and value != expected:
            raise ValueError('receipt mismatch: '+str(p))
        bindings[str(p)] = value
    bind(path); bind(path.parent/'protocol.json')
    if read(path.parent/'physical_report.json')['rows'] != report['rows']:
        raise ValueError('physical report differs')
    bind(path.parent/'physical_report.json')
    if read(path.parent/'endpoint_receipts.json') != report['endpoint_receipts']:
        raise ValueError('endpoint receipts differ')
    bind(path.parent/'endpoint_receipts.json')
    for s in report['source_seals']:
        for name in ('manifest.json','args.json'):
            bind(Path(s['output_dir'])/name)
    for row, plan in zip(report['rows'], plans, strict=True):
        if row['method'] != plan['method'] or not row['full_horizon_completed'] or row['completed_commands'] != 10:
            raise ValueError('complete ordered branch required')
        if [e['command'] for e in row['trace']] != plan['commands']:
            raise ValueError('physical commands differ from saved plan')
        branch = path.parent/f'root_{row["root_index"]}_{row["method"]}_replay0.physical.json'
        if read(branch) != row:
            raise ValueError('branch differs')
        bind(branch); bind(row['journal_path'], row['journal_sha256'])
    enc = {(e['method'],e['kind']):e for e in report['encodings']}
    slots = {(s['method'],s['kind']):s for s in report['endpoint_receipts']}
    if len(enc) != 246 or len(slots) != 246:
        raise ValueError('duplicate or missing physical slots')
    for key, e in enc.items():
        slot = slots[key]
        if (e['root_sha256'] != ROOT or slot['root_sha256'] != ROOT or e['status'] != 'completed'
                or e['comparable'] is not True or slot['status'] != 'observed'
                or slot['primitive_step'] != (5 if key[1] == 'prefix' else 10)
                or e['rgb_path'] != slot['rgb_path'] or e['rgb_sha256'] != slot['rgb_sha256']):
            raise ValueError('exact comparable physical slot required')
        array(e['vector'], (192,)); bind(e['rgb_path'], e['rgb_sha256'])
    root_rgb = Path(root_rgb).resolve()
    if not root_rgb.is_file():
        raise ValueError('original root RGB archive missing; ask main thread to reconstruct RGB metadata, never recreate environment here')
    frames = np.load(root_rgb, allow_pickle=False)
    if frames.shape != (3,224,224,3) or frames.dtype != np.uint8:
        raise ValueError('original uint8 RGB stack required')
    if hashlib.sha256(np.ascontiguousarray(frames).tobytes()).hexdigest() != refs[ROOT]['history_rgb_sha256']:
        raise ValueError('root archive not identical to original reference')
    bind(root_rgb)
    compiled = next(r['compiled'] for r in source['rows'] if r['root_sha256'] == ROOT)
    initial = dict(layout='states_with_between_state_actions_v1', context_length=3, latent_dim=192,
        action_dim=10, latent_history=compiled['input_context']['emb'][0],
        action_history=compiled['input_context']['actions'][0][1:])
    array(initial['latent_history'], (3,192)); array(initial['action_history'], (2,10))
    cases = []
    for role in ROLES:
        subset = [p for p in plans if p['role'] == role]
        anchor = subset[0]
        row = next(r for r in report['rows'] if r['method'] == role+'_zero')
        history = array(row['context_identity']['history'], (3,2))
        between = np.repeat(history[1:,None,:],5,axis=1)
        # Validate raw history on CUDA later against the saved normalized tokens.
        prefix = array(anchor['commands'], (10,2))[:5]
        for p in subset:
            if p['commands'][:5] != anchor['commands'][:5]:
                raise ValueError('prefix varies with suffix')
            if slots[p['method'],'prefix']['rgb_sha256'] != slots[role+'_zero','prefix']['rgb_sha256']:
                raise ValueError('physical prefix differs across probes')
        prefix_image = slots[role+'_zero','prefix']['rgb_path']
        with Image.open(prefix_image) as im:
            if np.asarray(im).shape != (224,224,3):
                raise ValueError('step5 RGB shape differs')
        cases.append(dict(role=role, prefix=prefix.tolist(), between_raw=between.tolist(), prefix_rgb=prefix_image,
            saved_predicted_prefix=anchor['saved_candidate']['predicted_prefix'],
            saved_actual_prefix=enc[role+'_zero','prefix']['vector'],
            real={p['probe']['probe_id']:enc[p['method'],'terminal']['vector'] for p in subset},
            slots={p['probe']['probe_id']:slots[p['method'],'terminal'] for p in subset}))
    frozen = dict(schema=TAG, capability=str(path), root_sha256=ROOT, root_rgb=str(root_rgb),
        initial_context=initial, cases=cases, bank=cap.suffix_bank(), bindings=bindings,
        goal=compiled['goal_vectors'][0], q=compiled['scoring']['metric']['projection'], policy=protocol['policy'],
        accounting=dict(predict_calls=249,suffix_predict_calls=246,prefix_predict_calls=3,
                        actual_encode_calls=3,actual_encode_frames=9,physics_steps=0,backwards=0),
        preprocessing=dict(api='p4_feedback_matched_20261027_v2.encode_cuda',
            frames='oldframes[1],oldframes[2],exact_step5_RGB',
            between='oldbetween[1],exact_saved_prefix_packet',
            shared_older_latents='compare reencoded history atol=2e-5 then use exact saved shifted older latents',
            prefix_comparison_atol=2e-5, cuda_matmul_allow_tf32=False,
            suffix='saved tanh(logits), central FD denominator 2*epsilon; not raw-command Jacobian'),
        P1_admission=False, training=False, fit=False, heldout_selection=False)
    return frozen, context


def freeze(output, capability=DEFAULT_CAPABILITY, root_rgb=DEFAULT_RGB):
    from experiments.spectral_reachability import p4_feedback_matched_20261027_v2 as feedback
    from experiments.spectral_reachability import pusht_composition_cuda_replay_v2 as adapter
    from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal
    from experiments.spectral_reachability.d2p_model_call_instrumentation import PredictInstrumentation
    from experiments.spectral_reachability.p3_matched_physical_20261027_v1 import write
    frozen, context = prepare(capability, root_rgb)
    base, checked, source, config = context[:4]
    output = Path(output).resolve()
    if not output.name.startswith(TAG):
        raise ValueError('exclusive tagged output required')
    output.mkdir(exist_ok=False)
    files = checked.sources_before_allocate(source,config)+[Path(__file__),Path(feedback.__file__),Path(adapter.__file__),
        Path(inspect.getfile(PredictInstrumentation)),REPO/f'tests/test_{TAG}.py',REPO/f'innerdocs/{TAG}.md']
    # Include every upstream capability-sealed source plus preprocessing code.
    for s in read(capability)['source_seals']:
        files += [Path(e['source']) for e in read(Path(s['output_dir'])/'manifest.json')['files']]
    files += [Path(inspect.getfile(base.p0.image_transform)),Path(inspect.getfile(base.p0.packets.normalize_live_packet))]
    files = sorted(set(p.resolve() for p in files))
    write(output/'protocol.json', frozen)
    seals = [create_seal(files=files[i:i+32], args_json=json.dumps(dict(protocol_sha256=sha(output/'protocol.json'))),
                         output_dir=output/f'source_seal_{i//32}') for i in range(0,len(files),32)]
    receipt = dict(status='prepared_not_run', protocol_sha256=sha(output/'protocol.json'), source_seals=seals,
                   P1_admission=False, actual_predict_calls=0, actual_encode_frames=0, physics_steps=0)
    write(output/'prepare_receipt.json', receipt)
    return receipt


def verify_frozen(output):
    from experiments.spectral_reachability.d2p_prerun_source_seal import verify_seal
    output = Path(output); receipt = read(output/'prepare_receipt.json'); f = read(output/'protocol.json')
    if receipt['status'] != 'prepared_not_run' or sha(output/'protocol.json') != receipt['protocol_sha256']:
        raise ValueError('prepared protocol changed')
    if not receipt['source_seals']:
        raise ValueError('prerun sources required')
    for s in receipt['source_seals']:
        verify_seal(output_dir=s['output_dir'],expected_manifest_sha256=s['manifest_sha256'])
    if any(sha(p) != h for p,h in f['bindings'].items()):
        raise ValueError('frozen input changed')
    return f


def run(output):
    """Explicit main-thread-reviewed host run. Never creates or steps an env."""
    from experiments.spectral_reachability import p4_feedback_matched_20261027_v2 as feedback
    from experiments.spectral_reachability import pusht_composition_cuda_replay_v2 as adapter
    from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
    from experiments.spectral_reachability import run_pusht_d2p_budget_guard_cuda as checked
    from experiments.spectral_reachability.d2p_model_call_instrumentation import PredictInstrumentation
    from experiments.spectral_reachability.p3_matched_physical_20261027_v1 import write
    output = Path(output).resolve(); f = verify_frozen(output)
    # Exclusive launch prevents accidental repeat or overwrite of partial runs.
    write(output/'launch_receipt.json', dict(protocol_sha256=sha(output/'protocol.json'),P1_admission=False))
    rows, contexts, failure, model = [], [], None, None
    image_work = dict(calls=0,frames=0)
    counter = None
    try:
        import torch
        from PIL import Image
        torch.backends.cuda.matmul.allow_tf32 = False
        host = base.require_cuda_allocation('cuda'); device = torch.device('cuda:0')
        model = base.load_compatible_pretrained(Path(f['policy']),device=device).eval().requires_grad_(False)
        before = checked.tensor_state_hash(model)
        replay = adapter.CudaReplay(model,device); counter = PredictInstrumentation(model)
        frames = np.load(f['root_rgb'],allow_pickle=False); transform = base.p0.image_transform()
        def predict(context, packet, method):
            with counter.phase('evaluation',method=method,horizon_tokens=1):
                value = replay(deepcopy(context),packet)
            torch.cuda.synchronize(device)
            return value
        with counter:
            for case in f['cases']:
                role = case['role']; prefix = np.asarray(case['prefix'])
                imagined = predict(f['initial_context'],prefix,role+'_prefix')
                prefix_delta = require_close(imagined['latent_history'][-1],case['saved_predicted_prefix'],'saved imagined prefix')
                with Image.open(case['prefix_rgb']) as im:
                    terminal = np.asarray(im.convert('RGB')).copy()
                actual_frames, between = feedback.advance_observed(frames, np.asarray(case['between_raw']),terminal,prefix,
                    dict(status='completed',completed_commands=5,remaining_commands_not_executed=0))
                # Check original normalized raw history, before its oldest packet drops.
                tokens = base.p0.packets.normalize_live_packet(torch.tensor(case['between_raw'],dtype=torch.float32,device=device),
                    mean=base.p0.PUBLIC_ACTION_MEAN,std=base.p0.PUBLIC_ACTION_STD).cpu().tolist()
                if tokens != f['initial_context']['action_history']:
                    raise ValueError('original raw/normalized history differs')
                image_work['calls'] += 1; image_work['frames'] += 3
                actual = feedback.encode_cuda(model,device,actual_frames,between,base.p0,transform)
                torch.cuda.synchronize(device)
                actual_delta = require_close(actual['latent_history'][-1],case['saved_actual_prefix'],'saved actual prefix')
                actual, older_delta = pair_contexts(imagined,actual)
                contexts.append(dict(role=role,imagined=imagined,actual=actual,between_raw=between.tolist(),
                    imagined_prefix_max_delta=prefix_delta,actual_prefix_max_delta=actual_delta,older_latent_max_delta=older_delta))
                write(output/f'{role}.contexts.json',contexts[-1])
                for arm, ctx in (('imagined',imagined),('actual',actual)):
                    predicted = {}
                    for b in f['bank']:
                        result = predict(ctx,np.tanh(b['logits']).reshape(5,2),role+'_'+arm+'_'+b['probe_id'])
                        predicted[b['probe_id']] = result['latent_history'][-1]
                        write(output/f'{role}.{arm}.{b["probe_id"]}.json',dict(predicted=predicted[b['probe_id']],
                            physical_slot=case['slots'][b['probe_id']],probe=b,P1_admission=False))
                    row = dict(role=role,arm=arm,analysis=analyze(predicted,case['real'],f['bank'],f['q'],f['goal']))
                    write(output/f'{role}.{arm}.analysis.json',row); rows.append(row)
        verify_frozen(output)
        if before != checked.tensor_state_hash(model) or any(m.training for m in model.modules()) or any(p.requires_grad for p in model.parameters()):
            raise ValueError('model not frozen/unchanged')
        if sum(r['predict_calls'] for r in counter.records) != 249 or len(rows) != 6 or image_work != dict(calls=3,frames=9):
            raise ValueError('actual work census differs')
    except BaseException as exc:
        failure = dict(type=type(exc).__name__,message=str(exc))
        raise
    finally:
        write(output/'report.json',dict(schema=TAG,status='integrity_fail' if failure else 'diagnostic_recorded',
            rows=rows,contexts=contexts,failure=failure,predict=[] if counter is None else counter.records,
            image_work=image_work,physics_steps=0,backwards=0,P1_admission=False,efficacy_verified=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','run'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--capability',type=Path,default=DEFAULT_CAPABILITY)
    parser.add_argument('--root-rgb',type=Path,default=DEFAULT_RGB)
    args = parser.parse_args()
    if args.command == 'prepare':
        print(json.dumps(freeze(args.output,args.capability,args.root_rgb)))
    else:
        run(args.output)
