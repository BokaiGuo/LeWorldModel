"""Saved-only six-slot target seal and resource preparation; never dispatches."""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path

import numpy as np

NAME = 'agent_e1_six_slot_common_prepare_20261029_v1'
ROOT = Path(__file__).resolve().parents[2]
ARMS = ('arbitrary_latent_waypoint', 'existing_learned_prior', 'dynamics_compiled_witness')
CAP = 16 * 1024 * 1024
FILES = {
 'controller_prepare': ('results/agent_experiment1_common_controller_20261028_v1_prepared_MAIN01.json', '8c472485d44c40ce3d95f8043d2c58f9ec2694fef74786a840d1699d5a1a86bd'),
 'controller_report': ('results/agent_experiment1_common_controller_20261028_v1_cuda_MAIN01/report.json', 'bc3cd74c85f878cdfc7b170d0b5b555827e0dc4a59cdaea6fb5781e71f03767f'),
 'controller_receipt': ('results/agent_experiment1_common_controller_20261028_v1_consume_MAIN01.json', 'b4b72dea9d782c6d2e1b93f50523a1089e7f3d894e2e2c1171fc13f9b8cd425d'),
 'hi_manifest': ('results/agent_hilewm_rgb_high_low_20261028_v1_prepare_MAIN01/manifest.json', 'a3255d38e9cf718f9560c1b9e381abe38750edd9e91d3f04e5ad332cf414d7e9'),
 'hi_report': ('results/agent_hilewm_rgb_high_low_20261028_v1_cuda_MAIN01/report.json', '083d0d9aa190b561e6ccf865b09d7224b97d404b16ba38c792bd8af386f5622a'),
 'hi_receipt': ('results/agent_hilewm_rgb_high_low_20261028_v1_consume_MAIN01.json', '40f28414682c9540e9e195b361a36dfc514f816358adbeacb379fedc460e4ef0'),
 'parity_receipt': ('results/agent_p2_hilewm_computation_parity_20261029_v1_consume_MAIN01.json', '3555696c02329062084d791c0c40958084d7352e6f7680938bade0068afe9480'),
 'pairing_receipt': ('results/agent_p2_hilewm_pairing_catalog_20261029_v1_join_MAIN01.json', '1523183330336bd3e66884a15fa96e7a15161abd8a343d1f50dba7a0427494b8'),
 'controller_source': ('experiments/spectral_reachability/agent_experiment1_common_controller_20261028_v1.py', 'c34a303c5e2df56a2002f195118cc2272a7b4d10afb79f8ed63d730af1a732a9'),
 'normalizer_source': ('experiments/spectral_reachability/pusht_d2p_packet_contract.py', 'eda631348e929b45591807dac0adcbb72d10d84b7d48a7a558a492eab9c04aef'),
 'normalizer_constants': ('experiments/run_pusht_counterfactual.py', '0a4eb4592951139e596b35becede17d175df7b48a2fde16679ab0aac7c4dce34'),
}


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def hash_bytes(data):
    return hashlib.sha256(data).hexdigest()


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode() + b'\n'


def parse(data):
    def unique(pairs):
        result = {}
        for k, v in pairs:
            require(k not in result, 'duplicate key')
            result[k] = v
        return result
    def bad(_):
        raise ValueError('nonfinite JSON')
    require(len(data) <= CAP, 'bounded JSON')
    return json.loads(data, object_pairs_hook=unique, parse_constant=bad)


def bytes_at(path, sha, cap=CAP):
    p = Path(path)
    require(p.is_absolute() and p.is_file() and not p.is_symlink(), 'absolute regular input')
    require(p.stat().st_size <= cap, 'file cap')
    data = p.read_bytes()
    require(hash_bytes(data) == sha, 'file SHA mismatch: '+str(p))
    return data


def npy(data, spec):
    require(hash_bytes(data) == spec['file_sha256'] and len(data) == spec['file_bytes'], 'NPY file SHA/length')
    f = io.BytesIO(data)
    a = np.load(f, allow_pickle=False)
    require(f.tell() == len(data) and a.dtype.str == '<f4' and list(a.shape) == [1, 192]
            and spec['dtype'] == '<f4' and spec['shape'] == [1, 192] and np.isfinite(a).all(), 'NPY exact ABI/no trailing')
    raw = a.tobytes(order='C')
    require(hash_bytes(raw) == spec['data_sha256'], 'NPY data SHA')
    return raw


def vector_bytes(value):
    require(type(value) is list and len(value) == 192 and all(type(x) in (int, float) and math.isfinite(x) for x in value), 'literal target vector')
    arr = np.asarray(value, dtype='<f4')
    require(np.isfinite(arr).all() and all(float(a) == b for a, b in zip(arr, value)), 'target exact f32')
    return arr.tobytes()


def protocol():
    # Replay is explicitly NEW: whole two packets, segmented two packets, each
    # seeded from the same context and reloaded literal raw10; compare bitwise,
    # report differences, never alter scientific gates.
    return dict(schema=NAME+'.protocol', roots=[0,1], arms=list(ARMS), slots=6,
        common_normalizer='P2 original engineering mean/std once; NOT historical Hi scaler',
        all_targets_sealed_before_any_model_load_or_dispatch=True, all_six_new_controller_executions=True,
        controller=dict(horizon_packets=2, raw_steps=10, iterations=3, samples=24,
            candidates_per_slot=72, predict_API_per_slot=144, predictor_samples_per_slot=144,
            action_encoder_per_slot=144, pred_proj_per_slot=144, total_predict_API_cap=864,
            total_predictor_samples_cap=864, total_action_encoder_cap=864, total_pred_proj_cap=864,
            seed_by_root=[2028102811,2028102812], action_domain='original projected logit ball -> f32 tanh -> raw; normalize once'),
        replay=dict(required='whole H2 and sequential 1+1 from saved literal raw10, same original history',
            predict_API_per_slot=4, predictor_samples_per_slot=4, action_encoder_per_slot=4,
            pred_proj_per_slot=4, six_slot_each_cap=24, extra_model_predict_total_cap=888,
            bitwise_comparison='record exact, maxabs, ULP; mismatch does not become equivalence',
            score_barrier='all required replay outputs sealed before replay score',
            state='prospective_not_implemented'),
        physical=dict(branches=6, warmup_raw_per_branch=10, control_raw_per_branch=10,
            public_step_cap=120, physics_step_cap=1224, explicit_png_cap=132,
            constructor_cap=6, close_cap=6, reset_cap=12, set_state_cap=24, goal_render_cap=12, goal_bind_cap=6,
            assumption='original adapter 4 physics per branch +10/public; no early stop, no extra operation; new proof required',
            actual_native_meter=None, state='prospective_not_implemented'),
        resources=dict(proposal_history='full parent receipts separately; no free/matched total compute',
            encoder_new=0, AD_new=0, failed_attempts_refunded=False, retries=0, padding=0,
            output_byte_cap=None, disk_floor=None, gpu_memory_cap=None, wall_seconds=None,
            reason='six-slot real journal/NPY/PNG shapes and byte census and runtime not yet proven; unknown is not zero'),
        historical_hi_scaler=False, official_hilewm_reproduction=False, scientific_admission=False,
        P1_admission=False, production_ready=False, dispatch_allowed=False)


def join(cp, cr, cu, hm, hr, hu, parity, pairing, tasks, payloads):
    require(cp['schema'] == 'agent_experiment1_common_controller_20261028_v1.prepare'
            and cr['prepare_sha256'] == FILES['controller_prepare'][1] and cr['errors'] == []
            and cr['model_hash_before'] == cr['model_hash_after'] == cp['expected_model_hash']
            and cr['contract'] == cp['contract'], 'original controller report/prepare')
    require(cu['status'] == 'common_controller_calibration_verified' and cu['scientific_slots'] == 6
            and cu['dispatch_denominator'] == 4 and cu['learned_unknown_slots'] == 2
            and cu['rows'] == cr['rows'] and len(cu['plans']) == 4, 'controller full persisted receipt')
    require(hr['manifest_sha256'] == FILES['hi_manifest'][1] and hr['errors'] == []
            and hu['report_sha256'] == FILES['hi_report'][1]
            and hu['status'] == 'RGB_high_FIRST_to_embedded_low_delivery_verified_integration_only', 'Hi receipt links')
    require(parity['all_values_exact'] is True and parity['diagnostics']['normalizer_pairing'] is False
            and parity['raw_deployment'] is False and pairing['all_five_tensor_catalog_equal'] is True,
            'narrow saved pairing evidence only')
    require(len(cp['slots']) == len(tasks) == 2 and len(hm['slots']) == len(hr['rows']) == len(hu['rows']) == 4, 'complete source denominator')
    rows = []
    for root in (0, 1):
        slot, task, hs, hrow, hrec = cp['slots'][root], tasks[root], hm['slots'][root*2], hr['rows'][root*2], hu['rows'][root*2]
        require(slot['source_slot'] == task['source_slot'] == hs['root_slot'] == hrow['root_slot'] == hrec['root_slot'] == root
                and hs['condition'] == hrow['condition'] == hrec['condition'] == 'off'
                and slot['blind_sha256'] == task['blind_sha256'] == hs['blind_sha256']
                and task['task_id'] == slot['task_id'] and task['task']['task_contract_sha256'] == slot['task_contract_sha256'], 'root/blind/task identity')
        t = task['task']
        require(t['environment'] == 'swm/PushT-v1' and len(t['root_raw_action_history']) == 10
                and t['action_mean'] == [-0.008700000122189522, 0.006800000090152025]
                and t['action_std'] == [0.20190000534057617, 0.20020000636577606], 'actual P2 engineering mean/std and raw history')
        high_name, low_name = hrow['high_target'], hrow['low']['z_subgoal']
        high = npy(payloads[high_name], hr['arrays'][high_name])
        low = npy(payloads[low_name], hr['arrays'][low_name])
        require(high == low and hrec['target_delivered_exact'] is True and hash_bytes(high) == hrec['high_target_sha256'] == hrec['low_goal_sha256'], 'actual FIRST to low exact bytes')
        for j, arm in enumerate(ARMS):
            if arm == ARMS[1]:
                raw = high
                origin = dict(kind='inherited_hi_FIRST', report_row=root*2, npy=high_name,
                              npy_sha256=hr['arrays'][high_name]['file_sha256'], data_sha256=hash_bytes(high))
            else:
                idx = root*2 + (0 if j == 0 else 1)
                plan = cu['plans'][idx]
                target = cp['targets'][root]['targets'][arm]
                require(plan['source_slot'] == root and plan['arm'] == arm and plan['task_id'] == slot['task_id']
                        and plan['task_sha256'] == slot['task_sha256'] and plan['blind_sha256'] == slot['blind_sha256']
                        and plan['target'] == target, 'old target/report identity')
                if arm == ARMS[2]:
                    require(cp['compiled'][root]['source_slot'] == root and cp['compiled'][root]['target'] == target
                            and cp['compiled'][root]['prefix_context']['latent_history'][-1] == target, 'compiled witness target')
                raw = vector_bytes(target)
                origin = dict(kind='inherited_original_target', receipt_plan=idx, data_sha256=hash_bytes(raw))
            rows.append(dict(root_slot=root, arm=arm, task_id=slot['task_id'], task_sha256=slot['task_sha256'],
                             blind_sha256=slot['blind_sha256'], origin=origin,
                             target_f32_le_base64=base64.b64encode(raw).decode(), target_data_sha256=hash_bytes(raw)))
    require([(r['root_slot'],r['arm']) for r in rows] == [(r,a) for r in (0,1) for a in ARMS], 'six slots exact order')
    return rows


def build(root=ROOT):
    data = {k: bytes_at(root/p, h) for k,(p,h) in FILES.items()}
    objects = {k: parse(v) for k,v in data.items() if k not in ('controller_source','normalizer_source','normalizer_constants')}
    cp, hm, hr = (objects[k] for k in ('controller_prepare','hi_manifest','hi_report'))
    tasks, payloads, extra = [], {}, {}
    for i in (0,1):
        slot, hslot, row = cp['slots'][i], hm['slots'][i*2], hr['rows'][i*2]
        task_data = bytes_at(slot['task_export'],slot['task_sha256'])
        with gzip.GzipFile(fileobj=io.BytesIO(task_data)) as f:
            decoded = f.read(CAP+1)
        require(len(decoded) <= CAP, 'decompressed task cap')
        tasks.append(parse(decoded))
        bytes_at(hslot['blind_path'], hslot['blind_sha256'])
        extra[str(hslot['blind_path'])] = hslot['blind_sha256']
        extra[str(slot['task_export'])] = slot['task_sha256']
        for name in (row['high_target'], row['low']['z_subgoal']):
            require(Path(name).name == name, 'flat NPY path')
            spec = hr['arrays'][name]
            path = root/'results/agent_hilewm_rgb_high_low_20261028_v1_cuda_MAIN01'/name
            payloads[name] = bytes_at(path, spec['file_sha256'], 4096)
            extra[str(path)] = spec['file_sha256']
    rows = join(*(objects[k] for k in ('controller_prepare','controller_report','controller_receipt','hi_manifest','hi_report','hi_receipt','parity_receipt','pairing_receipt')), tasks, payloads)
    source_files = [root/'experiments/spectral_reachability'/(NAME+'.py'), root/'tests'/('test_'+NAME+'.py'), root/'innerdocs'/(NAME+'.md')]
    return dict(schema=NAME+'.prepare', protocol=protocol(), rows=rows,
        target_seal_sha256=hash_bytes(canonical(rows)), inputs={str(root/p):h for p,h in FILES.values()} | extra,
        self_sources={str(p):hash_bytes(p.read_bytes()) for p in source_files},
        evidence_boundary='persisted MAIN receipts checked by pinned bytes and selected fields; no fullstrict rebuilt',
        dispatch_allowed=False, P1_admission=False, scientific_admission=False)


def write_new(path, value):
    p=Path(path)
    require(p.is_absolute() and not p.is_symlink(), 'absolute exclusive output')
    data=canonical(value)
    require(len(data)<=CAP, 'output cap')
    with p.open('xb') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    return dict(path=str(p), sha256=hash_bytes(data))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=('prepare','verify'))
    ap.add_argument('--output'); ap.add_argument('--manifest'); ap.add_argument('--manifest-sha256')
    a=ap.parse_args()
    if a.command == 'prepare':
        require(a.output, 'output required'); result=write_new(a.output,build())
    else:
        require(a.manifest and a.manifest_sha256, 'manifest and external SHA required')
        require(parse(bytes_at(a.manifest,a.manifest_sha256)) == build(), 'whole prepare mismatch')
        result=dict(status='six_slot_targets_verified_not_dispatch_authorized')
    print(json.dumps(result,sort_keys=True))


if __name__ == '__main__':
    main()
