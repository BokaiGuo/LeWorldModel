"""Bounded offline checkpoint/action/camera lineage audit, never model or physics.

CPU image transforms and JPEG roundtrip only. Uses fixed archive rows0/5/10
and the three sealed original root RGB frames, not outcome-selected images.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.spectral_reachability import p1_feedback_goal_contract_audit_20261027_v1 as common

TAG = 'p1_checkpoint_action_camera_provenance_20261027_v1'
REPO = common.REPO
SITE = REPO/'.venv/lib/python3.12/site-packages'
CHECKPOINT = Path('/home/gbk/.stable_worldmodel/checkpoints/pusht/lewm')
DATA = Path('/home/gbk/.stable_worldmodel/datasets/pusht_expert_train.h5')


def function(path, name, namespace):
    """Execute only the selected, hash-bound local function, not its module."""
    text = Path(path).read_text()
    nodes = [n for n in ast.walk(ast.parse(text)) if isinstance(n,ast.FunctionDef) and n.name==name]
    common.require(len(nodes)==1, f'unique source function required: {name}')
    node = nodes[0]
    scope = dict(namespace)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
    return scope[name]


def metadata_record(path, content, *, lfs):
    lines = Path(path).read_text().splitlines()
    common.require(len(lines)==3 and len(lines[0])==40, 'HF local download metadata layout')
    data = Path(content).read_bytes()
    digest = hashlib.sha256(data).hexdigest() if lfs else hashlib.sha1(
        f'blob {len(data)}\0'.encode()+data).hexdigest()
    common.require(digest==lines[1], 'cached object digest mismatch')
    return dict(revision=lines[0],etag=lines[1],digest_verified=True,
                authority='local HF cache assertion, not an online publisher authentication')


def image_paths():
    return dict(train=REPO/'utils.py',live=REPO/'experiments/run_pusht_counterfactual.py',
                lance=SITE/'stable_worldmodel/data/formats/lance.py')


def image_checks(images):
    import torch
    from stable_pretraining import data as dt
    paths = image_paths()
    train = function(paths['train'],'get_img_preprocessor',dict(dt=dt))('pixels','pixels',224)
    live = function(paths['live'],'image_transform',dict(torch=torch,spt=__import__('stable_pretraining'),
        transforms=__import__('torchvision.transforms',fromlist=['v2']).v2))()
    encoder = function(paths['lance'],'_encode_frame',dict(np=np,io=io,Image=Image))
    records = []
    for identity, rgb in images:
        common.require(rgb.dtype==np.uint8 and rgb.shape==(224,224,3), 'fixed RGB image shape')
        tensor = torch.from_numpy(rgb.copy()).permute(2,0,1)
        trained = train({'pixels':tensor.clone()})['pixels']
        current = live(tensor.clone())
        common.require(torch.equal(trained,current), 'current train/live CHW preprocessing mismatch')
        blob = encoder(rgb,95)
        decoded = np.asarray(Image.open(io.BytesIO(blob)).convert('RGB')).copy()
        jpeg = live(torch.from_numpy(decoded).permute(2,0,1))
        # Probe the real NumPy branch separately. Readers normally supply tensors.
        wrong = dt.transforms.ToImage(**dt.dataset_stats.ImageNet,source='pixels',target='pixels')(
            {'pixels':rgb.copy()})['pixels']
        common.require(torch.equal(wrong,current.transpose(-2,-1)), 'NumPy branch behavior changed')
        records.append(dict(image_id=identity,rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
            train_tensor_vs_live_max_abs=float((trained-current).abs().max()),
            numpy_branch_vs_chw_max_abs=float((wrong-current).abs().max()),
            numpy_branch_is_xy_transpose=True,
            default_lance_jpeg95_bytes_sha256=hashlib.sha256(blob).hexdigest(),
            jpeg95_changed_channel_values=int(np.count_nonzero(decoded!=rgb)),
            jpeg95_pixel_MAE=float(np.abs(decoded.astype(float)-rgb).mean()),
            jpeg95_preprocessed_MAE=float((jpeg-current).abs().mean()),
            jpeg_scope='Installed writer default diagnostic ONLY; not proof checkpoint trained with JPEG95.'))
    return records


def packet_units(raw_pixel_delta, live_command, mean, std):
    d = common.array(raw_pixel_delta,(5,2)); u = common.array(live_command,(5,2))
    mean,std = common.array(mean,(2,)),common.array(std,(2,))
    common.require(np.all(std>0),'positive normalizer std')
    common.close(d/100,u,'raw pixel displacement/live relative-PD argument differs',atol=1e-12)
    native = (d/100-mean)/std
    direct = (u-mean)/std
    common.close(native,direct,'native/live unit algebra differs')
    return dict(native_pixel_delta_to_live_token_exact=True,
        extra_div100_token_difference_max=float(np.max(np.abs((u/100-mean)/std-direct))))


def saved_replay(report):
    common.require(report['status']=='passed' and report['simulator']=='official' and
                   all(v is True for v in report['gates'].values()), 'saved physical replay integrity')
    common.require([r['episode'] for r in report['episodes']]==list(range(16)), 'all fixed16 source episodes')
    errors = [float(r['dense']['max_state_errors'][0]) for r in report['episodes']]
    maes = [float(v) for r in report['episodes'] for v in r['dense']['boundary_rgb_archive_mae']]
    common.require(np.isfinite(errors+maes).all() and min(errors+maes)>=0,'invalid saved replay metrics')
    return dict(saved_episodes=16,saved_boundary_images=len(maes),
        first_transition_max_state_error=max(errors), archived_vs_replayed_RGB_MAE_min=min(maes),
        archived_vs_replayed_RGB_MAE_max=max(maes), archived_vs_replayed_RGB_MAE_mean=float(np.mean(maes)),
        exact_camera_identity_established=False,
        scope='Read prior real physics receipt; no rerun and no causal attribution of pixel differences.')


def audit(output):
    from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal,verify_seal
    common.require(not output.exists(),'exclusive NEW output required')
    inputs=common.Inputs()
    sources=[Path(__file__),REPO/f'tests/test_{TAG}.py',REPO/f'innerdocs/{TAG}.md',Path(common.__file__),
        REPO/'experiments/spectral_reachability/d2p_prerun_source_seal.py',
        REPO/'utils.py',REPO/'train.py',REPO/'eval.py',REPO/'config/train/data/pusht.yaml',
        REPO/'experiments/run_pusht_counterfactual.py',
        SITE/'stable_pretraining/data/transforms.py',SITE/'stable_pretraining/data/dataset_stats.py',
        SITE/'stable_worldmodel/data/formats/hdf5.py',SITE/'stable_worldmodel/data/formats/lance.py',
        SITE/'stable_worldmodel/data/dataset.py',SITE/'stable_worldmodel/envs/pusht/env.py',
        SITE/'stable_worldmodel/wm/utils.py',
        REPO/'external/jepa-wms/app/plan_common/datasets/pusht_dset.py']
    for p in sources: inputs.bind(p)
    cache=CHECKPOINT/'.cache/huggingface/download'
    checkpoint={}
    for name in ('README.md','config.json','weights.pt'):
        path=inputs.bind(CHECKPOINT/name); meta=inputs.bind(cache/(name+'.metadata'))
        checkpoint[name]=metadata_record(meta,path,lfs=name=='weights.pt')
    common.require(len({v['revision'] for v in checkpoint.values()})==1,'cached checkpoint revisions differ')
    stats=inputs.load(REPO/'results/pusht_d2p_archive_action_statistics_v1_20261026.json')
    replay_dir=REPO/'results/pusht_d2p_packet_contract_20261026_physical_v1'
    replay=inputs.load(replay_dir/'report.json')
    inputs.bind(replay_dir/'protocol.json',replay['protocol_sha256'])
    inputs.bind(replay_dir/'implementation.json',replay['implementation_sha256'])
    feedback=inputs.load(REPO/'results/pusht_geometry_feedback_response_20261027_v1_prepare_v1/protocol.json')
    frames_path=inputs.bind(feedback['root_rgb'],feedback['bindings'][feedback['root_rgb']])
    root_frames=np.load(frames_path,allow_pickle=False)
    common.require(root_frames.shape==(3,224,224,3) and root_frames.dtype==np.uint8,'original root RGB layout')
    # Exclude runtime kind/axis mistakes by source evidence, not shape guessing.
    texts={str(p.relative_to(REPO)):p.read_text() for p in sources if p.suffix in ('.py','.yaml')}
    hdf=texts[str((SITE/'stable_worldmodel/data/formats/hdf5.py').relative_to(REPO))]
    lance=texts[str((SITE/'stable_worldmodel/data/formats/lance.py').relative_to(REPO))]
    common.require('permute(0, 3, 1, 2)' in hdf and 'permute(2, 0, 1)' in lance,
                   'reader tensor-layout evidence changed')
    common.require('_DEFAULT_JPEG_QUALITY = 95' in lance,'installed default JPEG specification changed')
    env=texts[str((SITE/'stable_worldmodel/envs/pusht/env.py').relative_to(REPO))]
    for statement in ('action = self.agent.position + action * self.action_scale',
                      'self.k_p, self.k_v = 100, 20','self.dt = 0.01','self.action_scale = 100'):
        common.require(statement in env,'relative-PD source contract changed')
    common.require('name: pusht_expert_train.lance' in texts['config/train/data/pusht.yaml'], 'training dataset config changed')
    # Data is too large for a full hash; bind selected hyperslabs plus file stat.
    # No states/outcomes, no environment or model are read/constructed.
    import h5py
    import hdf5plugin  # registers the existing archive pixel codec only
    before=DATA.stat()
    images=[]; source_actions=[]
    with h5py.File(DATA,'r') as handle:
        attrs={str(k):str(v) for k,v in handle.attrs.items()}
        fields={k:list(handle[k].shape) for k in ('action','pixels','episode_idx','step_idx')}
        for row in (0,5,10):
            common.require(int(handle['episode_idx'][row])==0 and int(handle['step_idx'][row])==row,
                           'fixed episode0 source row alignment')
            images.append((f'archive_episode0_row{row}',np.asarray(handle['pixels'][row])))
        source_actions=np.asarray(handle['action'][:5])
    after=DATA.stat()
    common.require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'archive changed')
    images.extend((f'original_root1_frame{i}',rgb) for i,rgb in enumerate(root_frames))
    mean,std=common.public_action_stats(REPO/'experiments/run_pusht_counterfactual.py')
    units=packet_units(source_actions.astype(float)*100,source_actions,mean[:2],std[:2])
    output.mkdir(parents=True,exist_ok=False)
    def write(name,value):
        with (output/name).open('x') as f:
            json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    data_receipt=dict(path=str(DATA),size=before.st_size,mtime_ns=before.st_mtime_ns,
        scope='selected rows0/5/10 pixels and0:5 actions, NOT full archive or checkpoint training proof',
        action_sample_sha256=hashlib.sha256(source_actions.tobytes()).hexdigest(),
        image_hashes={name:hashlib.sha256(rgb.tobytes()).hexdigest() for name,rgb in images},
        fields=fields,attributes=attrs)
    protocol=dict(schema=TAG,bindings=inputs.bindings,data=data_receipt,
        CPU_scope='Six image preprocessing comparisons and JPEG95 diagnostics, no neural operations',
        P1_admission=False,root_selection=False,training=False,new_physical_steps=0,new_predict_calls=0)
    write('protocol.json',protocol)
    seal=create_seal(files=sources,args_json=json.dumps(dict(protocol_sha256=common.sha(output/'protocol.json'))),
                     output_dir=output/'source_seal')
    image_records=image_checks(images)
    inputs.unchanged()
    current=DATA.stat()
    common.require((before.st_size,before.st_mtime_ns)==(current.st_size,current.st_mtime_ns),'archive changed at end')
    verify_seal(output_dir=seal['output_dir'],expected_manifest_sha256=seal['manifest_sha256'])
    config=common.read(CHECKPOINT/'config.json')
    result=dict(schema=TAG,status='offline_audit_recorded',checkpoint=checkpoint,
        protocol_sha256=common.sha(output/'protocol.json'),source_seal=seal,
        source_images_preprocessed=6,neural_calls=0,physical_steps=0,training_steps=0,
        image_records=image_records,unit_algebra=units,saved_replay=saved_replay(replay),
        current_source_training_dataset='pusht_expert_train.lance',
        current_default_lance_path_present=(DATA.parent/'pusht_expert_train.lance').exists(),
        public_std=std[:2].tolist(),archived_sample_std=stats['statistics']['sample_std'],
        checkpoint_config_keys=sorted(config),
        findings=[
            dict(id='numpy_ToImage_xy_swap',status='confirmed_library_branch_not_P1_execution',
                 evidence='Real saved RGB NumPy input is transposed relative to CHW; both installed readers supply CHW tensors.',
                 causal_P1_bug_established=False),
            dict(id='lance_JPEG_vs_direct_RGB',status='confirmed_source_path_difference_checkpoint_link_unresolved',
                 evidence='Installed Lance writer JPEG default95 is lossy on fixed real images; no authenticated training Lance/version/codec provenance.',
                 causal_P1_bug_established=False),
            dict(id='normalizer_training_export_missing',status='unresolved_absolute_training_contract',
                 evidence='Current source fits dataset stats; checkpoint config only constructs model. Public JEPA constants are not checkpoint-linked.',
                 causal_P1_bug_established=False)],
        precise_exclusions=[
            'Current installed actuator is a relative positional PD target, not a pure-velocity command.',
            'Current CHW tensor train preprocessing equals current live preprocessing exactly on all six saved images.',
            'NumPy ToImage transpose cannot be assigned to installed HDF5/Lance tensor paths.',
            'Pixel displacement /100 and live relative argument are distinct input units, not an unconditional extra division.'],
        missing_provenance=['Publisher training-run revision + fitted normalizer/action units bound to checkpoint digest.',
            'Training data revision and HDF5/Lance conversion, image codec parameters, renderer settings and versions.',
            'Archived-image vs simulator RGB identity: old receipt reports nonzero differences; cause not isolated.'],
        limitations=['Cache metadata identifies claimed checkpoint revision locally; no online authentication performed.',
            'Current local train source is not authenticated as the source that generated official weights.',
            'No checkpoint-weight tensors loaded; no encoding, model inference, physics, tuning or outcome selection.',
            'JPEG95 exercise is a specified codec witness, not a proposal to preprocess deployment images or tune camera.'],
        inputs_sources_unchanged=True,P1_admission=False,efficacy_verified=False)
    write('report.json',result)
    print(json.dumps(dict(status=result['status'],checkpoint_revision=checkpoint['weights.pt']['revision'],
        train_live_tensor_max=max(v['train_tensor_vs_live_max_abs'] for v in image_records),
        jpeg_pixel_MAE=[v['jpeg95_pixel_MAE'] for v in image_records],
        causal_P1_bug_established=False,new_neural_calls=0,new_physical_steps=0),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    print('Offline fixed-source action/camera audit: CPU image transforms only; no checkpoint tensors, neural calls or physics.',flush=True)
    audit(args.output.resolve())
