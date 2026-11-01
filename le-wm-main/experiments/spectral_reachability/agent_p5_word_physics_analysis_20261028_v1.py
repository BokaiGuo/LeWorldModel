"""All-12 P5 words: saved-only per-frame, separated agent/object/RGB analysis.

No neural/physics imports, fitting, success gate, selection or latent->pose map.
Requires actual MAIN strict physics receipt; reuses the frozen saved consumers.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np

TAG = 'agent_p5_word_physics_analysis_20261028_v1'
ROOT = Path(__file__).resolve().parents[2]
PHYSICAL = 'agent_p5_action_words_physical_20261028_v1'
PRODUCER = 'agent_p5_native_action_words_20261028_v2'
SEEDS = (2026102702, 2026102703, 2026102704)
WORDS = ('xy','yx','x_minus_x','zero_zero')
PAIRS = (('xy','yx'),('x_minus_x','zero_zero'))
CAP = 8*1024**2
PINS = {
    f'experiments/spectral_reachability/{PHYSICAL}.py': '6a28bef2cdd8141e7063b61144042b1401e0440dc6fedd7b8d92232b48d304fc',
    f'experiments/spectral_reachability/{PRODUCER}.py': '5dc5d158ff817d9f39dca0f82ea9eb562f0f9566cc3e74a2abfd2be1af4d74ce',
    'experiments/spectral_reachability/consume_p5_physical_MAIN_20261028_v1.py': '6d18e88c67239e9049d2d3d96f9af1f7aa28e1f758821015091f18f0cbebcab9',
    f'results/{PHYSICAL}_prepare_MAIN01/manifest.json': 'd33b539c77b35b4256a5e7099e39cdd1da619d8e8c3466d6f753ca3fded351d9',
    f'results/{PHYSICAL}_physics_MAIN01/report.json': '796d757c6ffcf623f4e307f094dba9e39eda2a4810a140e5ab3cf0897b3c73a5',
    'results/consume_p5_physical_MAIN_20261028_v1_MAIN01.json': '2248899fb03439bc4c52ee741dc0e2b497e15ce692d1d4d306f32fee7d476883',
    f'results/{PRODUCER}_prepare_MAIN01/manifest.json': '1515cb228c6ab0a4fe826074324fd566781b06812a1e5163c41652492518a905',
    f'results/{PRODUCER}_cuda_MAIN01/report.json': 'cdbccbdf894d8d6e7c078e02f7078bd8cd26e802c9a00d59421ad0e4b0883bbd',
    'results/consume_p5_words_MAIN_20261028_v1_MAIN01.json': '5237000615227a5cb6741e19c3feb4414c710077983399c46804b96c223b01db',
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def sha(path):
    path = Path(path)
    require(path.is_file() and not any(p.is_symlink() for p in (path,*path.parents)), 'regular nonsymlink input: '+str(path))
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):
            h.update(block)
    return h.hexdigest()


def read_json(path, expected=None):
    actual = sha(path)
    require(expected is None or actual==expected,'actual input SHA: '+str(path))
    require(Path(path).stat().st_size<=CAP,'JSON cap')
    def pairs(items):
        value = {}
        for k,v in items:
            require(k not in value,'duplicate JSON key')
            value[k] = v
        return value
    return json.loads(Path(path).read_text(),object_pairs_hook=pairs,
        parse_constant=lambda s:require(False,'nonfinite JSON '+s))


def load_source(name):
    rel = f'experiments/spectral_reachability/{name}.py'
    require(sha(ROOT/rel)==PINS[rel],'sealed saved consumer source')
    spec = importlib.util.spec_from_file_location(TAG+'_'+name,ROOT/rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def policy():
    return dict(seeds=list(SEEDS),words=list(WORDS),denominator=12,suffix_frames=list(range(11,21)),
        baseline_frame=10,frame_rows=120,pair_rows=60,pairs=[list(p) for p in PAIRS],
        delta_sign='left minus right; per-frame native body positions and velocities',
        units=dict(position='native pixel-coordinate length',velocity='native coordinate length/s',
                   angle='radian',angular_velocity='radian/s',coverage='native area fraction',
                   contact_points='native accumulated callback points within this control step, not distinct object contacts',
                   returned_n_contacts='native ceil(callback_points/10), not distinct contact count',
                   rgb='raw uint8 RGB intensity 0..255, full image, no segmentation',
                   patch_mse='model-only squared feature units, not physical distance'),
        task_success=None,scientific_admission=False,action_witness_established=False,
        object_noncommutativity_claim=False,latent_to_physical_fit=False,
        independence='three known roots, four paired words each; not twelve independent roots',
        interpretation='agent motion/velocity and full-image RGB differences do not establish object noncommutativity or task success')


def vector(value, shape):
    a = np.asarray(value,np.float64)
    require(a.shape==shape and np.isfinite(a).all(),'finite native quantity')
    return a


def principal(value):
    return float(np.arctan2(np.sin(value),np.cos(value)))


def rgb_difference(left, right):
    a,b = np.asarray(left),np.asarray(right)
    require(a.shape==b.shape==(224,224,3) and a.dtype==b.dtype==np.uint8,'raw RGB ABI')
    # Cast BEFORE subtracting: uint8 wrap would fabricate small differences.
    delta = a.astype(np.int16)-b.astype(np.int16)
    absolute = np.abs(delta)
    squared = delta.astype(np.float64)**2
    return dict(exact_equal=bool(np.array_equal(a,b)),changed_pixels=int(np.count_nonzero(np.any(delta!=0,axis=2))),
        changed_channels=int(np.count_nonzero(delta)),max_abs_intensity=int(absolute.max()),
        mae_intensity=float(absolute.mean()),mse_intensity_squared=float(squared.mean()),
        channel_mae_intensity=absolute.mean(axis=(0,1)).tolist(),
        channel_mse_intensity_squared=squared.mean(axis=(0,1)).tolist(),
        object_attribution='unknown: full raw image includes agent, object and native goal')


def quantities(public, transition, frame):
    """Use double-precision native body properties, not rounded native7 for pose."""
    agent,block,env = public['bodies']['agent'],public['bodies']['block'],public['env']
    agent_xy = vector(agent['position'],(2,))
    agent_velocity = vector(agent['velocity'],(2,))
    block_xy = vector(block['position'],(2,))
    block_velocity = vector(block['velocity'],(2,))
    angle = float(vector([block['angle']],(1,))[0])
    omega = float(vector([block['angular_velocity']],(1,))[0])
    coverage = vector(env['coverage_arr'],(frame,))
    require(frame>=10 and np.all((coverage>=0)&(coverage<=1)),'native coverage history')
    points = env['n_contact_points']
    require(type(points) is int and points>=0,'native callback points')
    require(transition['kind']=='step' and transition['done'] is False,'bound native done=False')
    info = transition['info']
    require(type(info['n_contacts']) is int and info['n_contacts']==int(np.ceil(points/10)), 'native contact formula')
    require(info['final_coverage']==float(coverage[-1]) and info['max_coverage']==float(coverage.max()),'native coverage receipt')
    reward = float(vector([transition['reward']],(1,))[0])
    require(reward==float(np.clip(coverage[-1]/env['success_threshold'],0,1)), 'native reward receipt')
    return dict(agent_xy=agent_xy.tolist(),agent_velocity=agent_velocity.tolist(),block_xy=block_xy.tolist(),
        block_angle_native_rad=angle,block_velocity=block_velocity.tolist(),block_angular_velocity=omega,
        coverage=float(coverage[-1]),max_coverage_through_frame=float(coverage.max()),
        callback_contact_points=points,returned_n_contacts=info['n_contacts'],native_reward=reward,native_done=False,
        native_goal_pose=env['goal_pose'],native_success_threshold=env['success_threshold'],task_success=None)


def difference(left, right):
    out = {}
    for name in ('agent_xy','agent_velocity','block_xy','block_velocity'):
        a,b = vector(left[name],(2,)),vector(right[name],(2,))
        d = a-b
        out[name] = dict(left_minus_right=d.tolist(),l2=float(np.linalg.norm(d)),exact_equal=bool(np.array_equal(a,b)))
    raw_angle = left['block_angle_native_rad']-right['block_angle_native_rad']
    out['block_angle'] = dict(left_minus_right_native_rad=raw_angle,
                             principal_left_minus_right_rad=principal(raw_angle),
                             exact_native_equal=left['block_angle_native_rad']==right['block_angle_native_rad'])
    for key in ('block_angular_velocity','coverage','callback_contact_points','returned_n_contacts'):
        out[key] = dict(left_minus_right=left[key]-right[key],exact_equal=left[key]==right[key])
    return out


def record_loader(out, row):
    indexed = {rec['frame']:rec for rec in row['records']}
    require(len(indexed)==len(row['records'])==21 and set(indexed)==set(range(21)), 'complete actual frame census')
    def load(frame):
        path = out/row['case']/f'frame_{frame:02d}'
        rec = indexed[frame]
        for name in ('rgb.npy','full_public_state.json','transition.json'):
            p = path/name
            require(rec['files'][name]==dict(sha256=sha(p),bytes=p.stat().st_size),'saved frame SHA/size')
        p = path/'rgb.npy'
        require(p.stat().st_size<=200000,'RGB size cap')
        rgb = np.load(p,allow_pickle=False,max_header_size=1024)
        require(rgb.shape==(224,224,3) and rgb.dtype==np.uint8,'native raw RGB')
        return (read_json(path/'full_public_state.json'),read_json(path/'transition.json'),rgb)
    return load


def assemble(branches, rows, model_rows, get_loader):
    """Pure saved arithmetic seam; fixtures prove mechanics, not real effects."""
    expected = [(s,w) for s in SEEDS for w in WORDS]
    require([(x['seed'],x['word']) for x in branches]==expected
            and [(x['seed'],x['word']) for x in rows]==expected,'all twelve ordered branches')
    require([x['seed'] for x in model_rows]==list(SEEDS),'three model rows')
    baselines,frames,pairs,summaries,parallel = [],[],[],[],[]
    banks = {}
    for branch,row in zip(branches,rows):
        for name in ('case','root_sha256','seed','word'):
            require(branch[name]==row[name],'root/word/source binding')
        require(row['status']=='completed' and row['error'] is None,'actual complete branch')
        identity = {k:branch[k] for k in ('case','seed','word','root_sha256')}
        load = get_loader(row)
        public,transition,rgb = load(10)
        baseline = quantities(public,transition,10)
        require(baseline['native_goal_pose']==branch['native_goal'],'original native goal')
        baselines.append(dict(**identity,frame=10,quantities=baseline))
        bank = {}
        for frame in range(11,21):
            p,t,img = load(frame)
            q = quantities(p,t,frame)
            require(q['native_goal_pose']==baseline['native_goal_pose'],'goal unchanged across suffix')
            bank[frame] = q,img
            frames.append(dict(**identity,frame=frame,suffix_step=frame-10,quantities=q,
                delta_from_prefix10=difference(q,baseline),rgb_vs_prefix10=rgb_difference(img,rgb)))
        banks[(branch['seed'],branch['word'])] = baseline,bank
    for seed,model in zip(SEEDS,model_rows):
        root_ids = {b['root_sha256'] for b in branches if b['seed']==seed}
        require(root_ids=={model['root_sha256']} and model['status']=='completed','producer/physics root join')
        contrasts = model['contrasts']
        for key in ('xy_vs_yx_patch_mse','x_minus_x_vs_zero_patch_mse'):
            require(np.isfinite(contrasts[key]) and contrasts[key]>=0,'actual model-only contrast')
        parallel.append(dict(seed=seed,root_sha256=model['root_sha256'],model_only_contrasts=contrasts,
                             actual_model_meter=model['meter'],scope='parallel endpoint feature units only; no latent-to-pose regression'))
        for left,right in PAIRS:
            lb,lframes = banks[(seed,left)]
            rb,rframes = banks[(seed,right)]
            require(lb==rb,'all words must share exact frame10 native baseline')
            pair_rows = []
            for frame in range(11,21):
                lq,li = lframes[frame]
                rq,ri = rframes[frame]
                item = dict(seed=seed,root_sha256=model['root_sha256'],left_word=left,right_word=right,
                            frame=frame,suffix_step=frame-10,delta=difference(lq,rq),rgb=rgb_difference(li,ri))
                pairs.append(item)
                pair_rows.append(item)
            summaries.append(dict(seed=seed,left_word=left,right_word=right,frames=10,
                agent_xy_max_l2=max(x['delta']['agent_xy']['l2'] for x in pair_rows),
                agent_velocity_max_l2=max(x['delta']['agent_velocity']['l2'] for x in pair_rows),
                block_xy_max_l2=max(x['delta']['block_xy']['l2'] for x in pair_rows),
                block_angle_max_abs_principal_rad=max(abs(x['delta']['block_angle']['principal_left_minus_right_rad']) for x in pair_rows),
                any_exact_block_pose_difference=any(not x['delta']['block_xy']['exact_equal'] or not x['delta']['block_angle']['exact_native_equal'] for x in pair_rows),
                any_rgb_difference=any(not x['rgb']['exact_equal'] for x in pair_rows),
                object_noncommutativity_claim=False,task_success=None))
    require(len(frames)==120 and len(pairs)==60 and len(baselines)==12,'unfiltered full-frame denominator')
    return dict(baselines=baselines,suffix_frame_rows=frames,pair_frame_rows=pairs,
                pair_descriptive_summaries=summaries,model_only_parallel=parallel)


def analyze():
    # Explicit existing paths only; no repository-wide asset scans.
    for rel,wanted in PINS.items():
        require(sha(ROOT/rel)==wanted,'required actual authority missing/drifted: '+rel)
    physical = load_source(PHYSICAL)
    producer = load_source(PRODUCER)
    mp = ROOT/f'results/{PHYSICAL}_prepare_MAIN01/manifest.json'
    rp = ROOT/f'results/{PHYSICAL}_physics_MAIN01/report.json'
    cp = ROOT/'results/consume_p5_physical_MAIN_20261028_v1_MAIN01.json'
    actual = physical.consume(mp,PINS[str(mp.relative_to(ROOT))],rp,PINS[str(rp.relative_to(ROOT))])
    require(actual==read_json(cp) and actual['status']=='all12_saved_native_replays_verified'
            and len(actual['verified_cases'])==12,'MAIN strict actual physics receipt required')
    pm = ROOT/f'results/{PRODUCER}_prepare_MAIN01/manifest.json'
    pr = ROOT/f'results/{PRODUCER}_cuda_MAIN01/report.json'
    pc = ROOT/'results/consume_p5_words_MAIN_20261028_v1_MAIN01.json'
    model_manifest = read_json(pm)
    model_actual = producer.consume(model_manifest,pr,PINS[str(pr.relative_to(ROOT))])
    require(model_actual==read_json(pc),'exact producer saved receipt')
    manifest,report,model_report = read_json(mp),read_json(rp),read_json(pr)
    require(report['global_error'] is None and report['status']=='all12_native_replays_recorded','all12 completed, no hidden error')
    require([r['root_sha256'] for r in model_manifest['roots']]==[r['root_sha256'] for r in model_report['rows']], 'producer root identities')
    data = assemble(manifest['branches'],report['rows'],model_report['rows'],lambda row:record_loader(rp.parent,row))
    inputs = {str(ROOT/p):h for p,h in PINS.items()}
    for p in (Path(__file__).resolve(),ROOT/'tests'/f'test_{TAG}.py',ROOT/'innerdocs'/f'{TAG}.md'):
        inputs[str(p)] = sha(p)
    # Both strict consumers checked original full closures, arrays, journals and
    # inventory. Preserve those receipt hashes, not only endpoint metrics.
    return dict(schema=TAG+'.report',status='all12_saved_word_frames_analyzed',policy=policy(),
        inputs=inputs,physics_consume=actual,model_consume=model_actual,
        physical_meter_rows=[dict(case=r['case'],native_call_meter=r['meter'],operation_meter=r['operation_meter']) for r in report['rows']],
        neural_executed=False,physics_executed=False,scientific_admission=False,task_success=None,**data)


def write_new(path, value):
    path = Path(path)
    require(path.parent.is_dir() and not any(p.is_symlink() for p in (path,*path.parents)),'new nonsymlink output')
    data = canonical(value)+b'\n'
    require(len(data)<=CAP,'analysis output cap')
    with path.open('xb') as f:
        require(f.write(data)==len(data),'short output write')
        f.flush()
        os.fsync(f.fileno())
    fd = os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['analyze','verify'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--output-sha256')
    args = parser.parse_args(argv)
    if args.command=='analyze':
        require(not args.output.exists(),'exclusive new output required')
        result = analyze()
        write_new(args.output,result)
    else:
        require(args.output_sha256,'actual saved analysis SHA required')
        require(read_json(args.output,args.output_sha256)==analyze(),'deterministic saved analysis mismatch')
    print(json.dumps(dict(path=str(args.output.resolve()),sha256=sha(args.output),
                         status='saved_analysis_written' if args.command=='analyze' else 'saved_analysis_recomputed_verified',
                         scientific_admission=False),sort_keys=True))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
