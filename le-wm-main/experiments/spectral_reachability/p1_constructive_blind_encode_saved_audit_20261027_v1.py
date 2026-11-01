"""Independent saved-array audit; no producer, torch, model, certificate imports.

External receipt SHA is mandatory. Saved receipts attest calls, not neural truth.
"""
from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import zipfile

import numpy as np

NAME = 'p1_constructive_blind_encode_saved_audit_20261027_v1'
PRODUCER = 'p1_constructive_blind_encode_cuda_20261027_v1'
ROOT = Path(__file__).resolve().parents[2]
PREPARE_SHA = '7470bca322e14ca7f7e94a29f9d05ef46ed0108824158a098a2760ecec58d491'
NORMALIZER_SHA = '0a4eb4592951139e596b35becede17d175df7b48a2fde16679ab0aac7c4dce34'
TENSOR_SHA = '4b12c01f5859f4b46bea478f3a8d5e2d3f516df456a622a3feec31bde7e33546'
POLICY = Path('/home/gbk/.stable_worldmodel/checkpoints/pusht/lewm')
MODELS = {str(POLICY/'config.json'): '2564086e961e7b5c7c04dffc451091115b389a590645ff19653c64fd0bc16e09',
          str(POLICY/'weights.pt'): '48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607'}
BLIND = [str(ROOT/f'results/p1_constructive_reachable_source_20261027_v1_main01/s{i}_blind_inputs.npz') for i in range(2)]
BLIND_SHA = ['d12ef7b66b69e53444ae6a4bde51c3830bacd6ad449d79845030ea70e2908da3',
             '9c0f6451de986bbb48c8251a8cfaab2b3fed60ce15698a9c6454c6206e228957']
LIMIT = 2*1024**2


def need(ok, message):
    if not ok:
        raise ValueError(message)


def hash_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def task_hash(value):
    # Core task-ID serialization differs deliberately from receipt serialization.
    return hash_bytes(json.dumps(value, sort_keys=True, allow_nan=False).encode())


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def sha(path):
    path = Path(path)
    need(path.is_file() and not path.is_symlink(), 'regular non-symlink file')
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def unique(pairs):
    value = {}
    for key, item in pairs:
        need(key not in value, 'duplicate JSON key')
        value[key] = item
    return value


def parse(raw):
    value = json.loads(raw, object_pairs_hook=unique)
    canonical(value)
    return value


def read_bytes(path, expected=None):
    path = Path(path)
    need(path.is_file() and not path.is_symlink() and path.stat().st_size < LIMIT, 'bounded regular input')
    raw = path.read_bytes()
    if expected is not None:
        need(hash_bytes(raw) == expected, 'external/file SHA mismatch: '+path.name)
    return raw


def read_json(path, expected=None):
    raw = read_bytes(path, expected)
    if str(path).endswith('.gz'):
        with gzip.GzipFile(fileobj=BytesIO(raw)) as f:
            raw = f.read(LIMIT)
        need(len(raw) < LIMIT, 'expanded JSON cap')
    return parse(raw)


def blind_arrays(path, expected):
    raw = read_bytes(path, expected)
    with zipfile.ZipFile(BytesIO(raw)) as archive:
        entries = archive.infolist()
        need(len(entries) == 3 and {e.filename for e in entries} ==
             {'rgb_history.npy', 'raw_action_history.npy', 'goal_rgb.npy'}, 'blind archive only; no privileged fields')
        need(all(e.file_size <= 512*1024 and not e.flag_bits & 1 for e in entries) and
             sum(e.file_size for e in entries) < 1024**2, 'bounded blind members')
    with np.load(BytesIO(raw), allow_pickle=False) as f:
        result = {k: f[k].copy() for k in f.files}
    for k, shape, dtype in [('rgb_history', (3,224,224,3), np.uint8),
                             ('goal_rgb', (224,224,3), np.uint8),
                             ('raw_action_history', (10,2), np.float32)]:
        need(result[k].shape == shape and result[k].dtype == dtype, 'blind array '+k)
    need(not result['raw_action_history'].any(), 'actual raw zero warmup')
    return result


def normalizer(source):
    raw = read_bytes(source, NORMALIZER_SHA)
    values = {}
    for node in ast.parse(raw).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            key = node.targets[0].id
            if key in ('PUBLIC_ACTION_MEAN', 'PUBLIC_ACTION_STD'):
                need(isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == 'np.asarray', 'literal normalizer')
                values[key] = np.asarray(ast.literal_eval(node.value.args[0]), np.float32).tolist()
    need(len(values) == 2, 'normalizer constants')
    return dict(mean=values['PUBLIC_ACTION_MEAN'], std=values['PUBLIC_ACTION_STD'], source_sha256=NORMALIZER_SHA,
                training_provenance_verified=False, raw_semantics='relative raw already action_scale/100; no second division')


def reconstruct(p, prepare_sha, slot, packet, latents, norm):
    """Independent literal task assembly, NOT producer.task_export/core.task_id."""
    need(len(latents) == 4, 'four independently paid arrays')
    for z in latents:
        need(z.shape == (192,) and z.dtype == np.float32 and np.isfinite(z).all(), 'native float32 CLS192')
    pixels = [hash_bytes(rgb.tobytes()) for rgb in [*packet['rgb_history'], packet['goal_rgb']]]
    raw = packet['raw_action_history']
    # Time-major: rows0:5 -> first between-state packet, rows5:10 -> second.
    tokens = np.stack([((raw[k:k+5] - np.asarray(norm['mean'], np.float32)) /
                       np.asarray(norm['std'], np.float32)).reshape(10) for k in (0,5)])
    tc = dict(schema=PRODUCER+'.task_contract', prepare_sha256=prepare_sha,
              whitelist_sha=p['arguments']['whitelist_sha'], with_target=False, source_slot=slot['source_slot'],
              blind_source_sha256=slot['blind_sha256'], observed_steps=[0,5,10], raw_between_steps=10,
              normalization=norm, model_bindings=MODELS, metric='identity_native192_no_fit')
    task = dict(environment='swm/PushT-v1', task_contract_sha256=task_hash(tc),
                model_sha256=MODELS[str(POLICY/'weights.pt')], model_config_sha256=MODELS[str(POLICY/'config.json')],
                metric_source_sha256=task_hash(dict(metric='identity_native192',rank=192,no_fit=True,with_target=False)),
                goal_source_sha256=pixels[3], representation='native_projected_cls', latent_dim=192,
                horizon=2, packet_steps=5, control_dim=2, action_mean=norm['mean'], action_std=norm['std'],
                packet_logit_cap=float(.75*np.sqrt(5)), feedback='open_loop_model_only', projection=np.eye(192).tolist(),
                goal=latents[3].astype(float).tolist(), root_raw_action_history=raw.astype(float).tolist(),
                root_context=dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,action_dim=10,
                                  latent_history=[z.astype(float).tolist() for z in latents[:3]],
                                  action_history=tokens.astype(float).tolist()))
    return tc, task, task_hash(task), pixels


def expected_events(rows, pixels):
    events = []
    for i, row in enumerate(rows):
        for j, image in enumerate(row['images']):
            index = 4*i+j
            events.append(dict(kind='image_attempt',index=index,root_order=i,
                               role=['observed0','observed5','observed10','goal'][j],pixel_sha256=pixels[i][j]))
            for resource, n in [('encoder',2*index), ('projector',2*index+1)]:
                for kind, completed in [('module_attempt',False), ('module_return',True)]:
                    events.append(dict(kind=kind,entry=dict(index=n,image=index,resource=resource,completed=completed)))
            events.append(dict(kind='image_saved',index=index,latent=image))
        events.append(dict(kind='task_sealed',row=row))
    return events


def audit(report_path, report_sha, prepare_sha=PREPARE_SHA):
    """Strict complete-receipt audit; partial reports are rejected, never relabeled."""
    report_path = Path(report_path).absolute()
    r = read_json(report_path, report_sha)
    directory = report_path.parent
    need(r['schema'] == PRODUCER+'.report' and r['prepare_sha256'] == prepare_sha, 'report schema/prepare')
    need(r['status'] == 'blind_tasks_encoded_and_sealed' and r['errors'] == [] and r['first_error'] is None,
         'complete receipt required; partial/failures not success')
    expected_names = {'binding.json','image.events.jsonl',*[f'image_{i:02d}.native192.npy' for i in range(8)],
                      'task_00.json.gz','task_01.json.gz'}
    need(set(r['artifact_sha256']) == expected_names, 'complete 8-array/2-task artifact inventory')
    for name, h in r['artifact_sha256'].items():
        need(sha(directory/name) == h, 'saved artifact SHA '+name)
    b = read_json(directory/'binding.json')
    need(b['prepare_sha256'] == prepare_sha, 'binding prepare SHA')
    p = read_json(b['prepared'], prepare_sha)
    need(p['schema'] == PRODUCER+'.prepare' and p['model_bindings'] == MODELS and p['expected_model_hash'] == TENSOR_SHA,
         'original frozen model contract')
    wp = p['arguments']['approved_whitelist']; wh = p['arguments']['whitelist_sha']
    w = read_json(wp, wh)
    need(set(w) == set('schema approved_by purpose source_admission admission_sha256 with_target history_steps raw_action_steps slots'.split()),
         'strict blind whitelist keys')
    need(w['schema'] == PRODUCER+'.approved_blind_whitelist' and w['approved_by'] == 'MAIN' and
         w['purpose'] == 'calibration' and w['source_admission'] == 'MAIN_independently_admitted' and
         w['with_target'] is False and w['history_steps'] == [0,5,10] and w['raw_action_steps'] == 10, 'whitelist authority/chronology')
    admission = w['admission_sha256']
    need(isinstance(admission,str) and len(admission) == 64 and all(c in '0123456789abcdef' for c in admission), 'opaque admission SHA')
    need(p['authority'] == dict(approved_by='MAIN',admission_sha256=admission,sourcepack_path_not_provided=True), 'prepare admission binding')
    need(len(w['slots']) == len(p['slots']) == len(r['rows']) == 2, 'two authorized slots')
    norm = normalizer(ROOT/'experiments/run_pusht_counterfactual.py')
    need(norm == p['normalization'], 'independently parsed normalizer')
    # Hash code/model closure only. Never follow a privileged result/attestation path.
    closure = p['bindings_sha256']
    for path, h in closure.items():
        f = Path(path)
        permitted = path in [wp,*BLIND,*MODELS] or (f.suffix in ('.py','.md') and 'results' not in f.parts)
        need(permitted, 'no privileged/unknown binding path')
        need(sha(f) == h, 'source closure hash '+f.name)
    need(all(closure.get(path) == h for path,h in {wp:wh,**MODELS,**dict(zip(BLIND,BLIND_SHA))}.items()), 'complete input/model bindings')
    summary, all_pixels = [], []
    for i, (slot,row) in enumerate(zip(p['slots'],r['rows'])):
        need(w['slots'][i] == dict(order=i,source_slot=i,blind_npz=BLIND[i],blind_sha256=BLIND_SHA[i]), 'exact authorized source slot')
        need(all(slot[k] == v for k,v in w['slots'][i].items()), 'prepare source order')
        packet = blind_arrays(BLIND[i], BLIND_SHA[i])
        need(row['order'] == i and row['status'] == 'task_sealed' and row['blind_source_sha256'] == BLIND_SHA[i] and len(row['images']) == 4, 'row census')
        zs = []
        for j, image in enumerate(row['images']):
            path = directory/f'image_{4*i+j:02d}.native192.npy'
            need(image == dict(path=str(path),sha256=r['artifact_sha256'][path.name],dtype='float32',shape=[192]), 'image path/layout/hash')
            zs.append(np.load(BytesIO(read_bytes(path,image['sha256'])),allow_pickle=False))
        tc, task, ident, pixels = reconstruct(p,prepare_sha,slot,packet,zs,norm)
        need(slot['pixel_sha256'] == pixels and slot['raw_action_sha256'] == hash_bytes(packet['raw_action_history'].tobytes()), 'raw source pixel/action seals')
        path = directory/f'task_{i:02d}.json.gz'
        need(row['task_path'] == str(path) and row['task_file_sha256'] == r['artifact_sha256'][path.name], 'task file binding')
        export = read_json(path,row['task_file_sha256'])
        need(export['schema'] == PRODUCER+'.task_export' and export['task_contract'] == tc and export['task'] == task and
             export['task_id'] == row['task_id'] == ident, 'independent whole-task/context/normalization/taskID reconstruction')
        need(export['native_root_sha256'] is None and export['blind_source_sha256'] == export['blind_sha256'] == BLIND_SHA[i] and
             export['with_target'] is False and export['source_slot'] == i and export['P1_admission'] is False and
             export['new_independent_confirmation'] is False and export['global_freshness'] == 'unknown', 'no false root/admission identity')
        summary.append(dict(order=i,task_id=ident,latent_files=4,history_steps=[0,5,10],between_packet_shape=[2,10],
                            identical_observed_latents=all(np.array_equal(zs[0],z) for z in zs[1:3]),
                            normalized_zero_token=task['root_context']['action_history'][0],identity_Q=True))
        all_pixels.append(pixels)
    events = [parse(line) for line in read_bytes(directory/'image.events.jsonl').splitlines()]
    need(events == expected_events(r['rows'], all_pixels), 'exact causal event order/no missing/dedup/extra attempt')
    counts = {k:dict(attempted=sum(e['kind']=='module_attempt' and e['entry']['resource']==k for e in events),
                     completed=sum(e['kind']=='module_return' and e['entry']['resource']==k for e in events)) for k in ('encoder','projector')}
    need(counts == r['resources'] == {k:dict(attempted=8,completed=8) for k in counts}, '8+8 full resource ledger')
    need(all(r[k] == 0 for k in ('new_predict','new_actionencoder','new_AD','new_physics')) and r['training'] is False and
         r['with_target'] is False and r['P1_admission'] is False and r['source_verified'] is True and
         r['model_unchanged'] is True and r['model_hash_before'] == r['model_hash_after'] == TENSOR_SHA, 'inherited execution/hash claims')
    size = sum(f.stat().st_size for f in directory.iterdir() if f.is_file())
    need(size <= LIMIT and {f.name for f in directory.iterdir()} == expected_names | {'report.json'}, 'exclusive output inventory/cap')
    need(sha(report_path) == report_sha and sha(b['prepared']) == prepare_sha and sha(wp) == wh, 'final receipt/input TOCTOU')
    for path,h in r['artifact_sha256'].items():
        need(sha(directory/path) == h, 'final artifact TOCTOU')
    return dict(schema=NAME+'.report',status='saved_consistency_verified',report_sha256=report_sha,prepare_sha256=prepare_sha,
                whitelist_sha256=wh,admission_sha256=admission,rows=summary,resources=counts,event_count=len(events),
                saved_output_bytes=size,new_model_calls=0,new_physics=0,P1_admission=False,
                limitations=['No certificate/sourcepack/admission report read; admission remains MAIN attestation.',
                             'No neural recomputation: pixels-to-latent truth and runtime absence of unlogged calls not independently proven.',
                             'Model unchanged and zero forbidden calls are source-bound receipt claims, not a second execution.',
                             'No physical success, reachability, freshness, or scientific efficacy conclusion.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',required=True); parser.add_argument('--report-sha',required=True)
    parser.add_argument('--prepare-sha',default=PREPARE_SHA); parser.add_argument('--output',required=True)
    args = parser.parse_args()
    try:
        result = audit(args.report,args.report_sha,args.prepare_sha)
        code = 0
    except Exception as exc:
        result = dict(schema=NAME+'.report',status='saved_audit_rejected',report_sha256=args.report_sha,
                      error=dict(type=type(exc).__name__,message=str(exc)),P1_admission=False)
        code = 1
    result['auditor_sha256'] = sha(__file__)
    raw = canonical(result); out = Path(args.output)
    need(len(raw)<128*1024 and shutil.disk_usage(out.parent).free >= 200*1024**2+len(raw), 'small report/reserve')
    with out.open('xb') as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())
    print(raw.decode().strip())
    return code


if __name__ == '__main__':
    raise SystemExit(main())
