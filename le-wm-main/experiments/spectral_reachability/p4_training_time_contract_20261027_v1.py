"""Source-derived dataset/label timing versus saved P4; no neural or physics calls."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sysconfig
from types import SimpleNamespace as NS

import numpy as np

from experiments.spectral_reachability import p4_feedback_admission_20261027_v1 as saved
from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal, verify_seal

TAG = 'p4_training_time_contract_20261027_v1'
REPO = Path(__file__).resolve().parents[2]
SITE = Path(sysconfig.get_paths()['purelib'])/'stable_worldmodel'
RUNTIME = REPO/'results/p4_multiround_feedback_runtime_20261027_v1_prepare_v1/report.json'
PRIOR = REPO/'results/p4_multiround_outcomes_20261027_v1_main_v1/report.json'
DATA = Path('/home/gbk/.stable_worldmodel/datasets/pusht_expert_train.h5')
CHECKPOINT = Path('/home/gbk/.stable_worldmodel/checkpoints/pusht/lewm')
require = saved.require

# Exact reviewed functions eligible for execution with metadata/NumPy stand-ins.
# Importing their modules (torch/model/data constructors) is deliberately avoided.
PROBES = {
    'train': (REPO/'train.py', None, 'lejepa_forward', 'e745ca95c04c5f573f184d08f3bebfe35089efb3bf11076ffa705c34a2468a33'),
    'item': (SITE/'data/dataset.py', 'Dataset', '__getitem__', 'd8098e688876d0ae534362f6ae7baa62ec812c20170e0ca4e8c5368001b4add2'),
    'chunk': (SITE/'data/dataset.py', 'Dataset', 'load_chunk', 'f8a27e45e61b6ab7acfe7e29117f66b3d0f576aefd52ecf0839a88654dd1ba40'),
    'h5': (SITE/'data/formats/hdf5.py', 'HDF5Dataset', '_load_slice', '676b10ff4359c222e96a216ed89fdee5aef7d460363638e823b8cc6abea43364'),
    'numeric': (SITE/'data/formats/lance.py', 'LanceDataset', '_prepare_numeric_tensor', '3949394fb144275b7bf893467c44a1cbe571dcc38ff3fe2e3656ba9e6458995f'),
    'batch': (SITE/'data/formats/lance.py', 'LanceDataset', '_process_batch', '84d383eb43290c9c4d87fc7f7a919bbc393bc0ae2c92de97d93e34de4ff83a7c'),
    'items': (SITE/'data/formats/lance.py', 'LanceDataset', '__getitems__', '4f66a90a91778896dbaf66a35ad14def6622eeb8dcf287f363f256192da27020'),
    'embed': (SITE/'wm/lewm/module.py', 'Embedder', 'forward', 'ecdc3c67a8894192903ae1086967e0db9a60c8d313a81bc44d1b23a5bdc3c410'),
}


def function(inputs, key):
    path, owner, name, digest = PROBES[key]
    text = inputs.raw(path).decode()
    tree = ast.parse(text)
    scope = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner) if owner else tree
    node = next(n for n in scope.body if isinstance(n, ast.FunctionDef) and n.name == name)
    code = ast.get_source_segment(text, node)
    require(hashlib.sha256(code.encode()).hexdigest() == digest, 'unreviewed probe source: '+key)
    # Future annotations prevent importing any type-only torch/Arrow dependencies.
    env = dict(np=np, torch=NS(from_numpy=np.asarray, tensor=np.asarray,
        nan_to_num=lambda x, unused: np.nan_to_num(x)))
    exec(compile('from __future__ import annotations\n'+code, str(path), 'exec'), env)
    return env[name], dict(path=str(path), owner=owner, function=name,
        start_line=node.lineno, end_line=node.end_lineno, function_sha256=digest, source=code)


class Marker:
    """Axis metadata only, never a learned tensor or a neural operation."""
    def __init__(self, value):
        self.value = np.asarray(value)

    def __getitem__(self, index):
        return Marker(self.value[index])

    def detach(self):
        return self

    def float(self):
        return self

    def permute(self, *axes):
        return Marker(self.value.transpose(axes))


def marker_probes(inputs):
    functions, evidence = {}, {}
    for key in PROBES:
        functions[key], evidence[key] = function(inputs, key)
    # Nonzero episode offset AND non-frame-aligned local start distinguish both offsets.
    indices = np.arange(200)
    storage = dict(state=indices, action=np.stack([1000+indices, 2000+indices], axis=1))
    obj = NS(offsets=[0, 97], frameskip=5, num_steps=4, span=20, transform=None,
        clip_indices=[(1, 2), (1, 3)], _keys=['state', 'action'], _cache=storage,
        image_columns=[], _fetch_columns=[], _open=lambda: None)
    obj._load_slice = lambda ep,s,e: functions['h5'](obj,ep,s,e)
    h5 = functions['item'](obj,0)
    chunk = functions['chunk'](obj,[1],[2],[22])[0]
    obj._prepare_numeric_tensor = lambda data,downsample: functions['numeric'](obj,data,downsample)
    obj._process_batch = lambda ep,start,batch,g_end=None: functions['batch'](obj,ep,start,batch,g_end)
    lance = functions['items'](obj,[0,1,0])
    expected = [99,104,109,114]
    require(h5['state'].tolist() == expected, 'episode/local chunk offset mismatch')
    require(np.array_equal(h5['action'], storage['action'][99:119].reshape(4,10)), 'time/channel flatten mismatch')
    for result in (chunk, lance[0], lance[2]):
        require(all(np.array_equal(result[k],h5[k]) for k in h5), 'HDF5/Lance/chunk timing differs')
    require(lance[1]['state'].tolist() == [100,105,110,115], 'batched local-start ordering differs')

    captured = {}
    class StopProbe(Exception):
        """Stop before any loss or training operation."""

    class Prediction:
        def __sub__(self, target):
            captured['target'] = target.value.tolist()
            raise StopProbe()

    def predict(ctx, actions):
        captured['context'] = ctx.value.tolist()
        captured['action'] = actions.tolist()
        return Prediction()
    cfg = NS(history_size=3, num_preds=1, loss=NS(sigreg=NS(weight=0)))
    model = NS(encode=lambda batch: dict(emb=Marker(h5['state'][None,:,None]),
        act_emb=batch['action']), predict=predict)
    try:
        functions['train'](NS(model=model),dict(action=h5['action'][None]),'probe',cfg)
    except StopProbe:
        captured['stopped_before_loss'] = True
    require(captured.get('target') == [[[104],[109],[114]]] and
        captured.get('context') == [[[99],[104],[109]]], 'training target/context shift mismatch')
    require(np.array_equal(captured['action'],h5['action'][None,:3]), 'training action slice mismatch')

    axes = {}
    def channel_probe(x):
        axes['conv_input_shape'] = list(x.value.shape)
        axes['conv_channels_at_token0'] = x.value[0,:,0].tolist()
        return x
    tokens = Marker(h5['action'][None,:3])
    result = functions['embed'](NS(patch_embed=channel_probe,embed=lambda x:x),tokens)
    require(axes['conv_input_shape'] == [1,10,3] and
        np.array_equal(result.value,tokens.value), 'action encoder time/channel transpose mismatch')
    return dict(evidence=evidence, sample_global_observation_rows=expected,
        sample_flat_action_tokens=h5['action'].tolist(), training=captured, action_encoder=axes,
        hdf5_lance_chunk_equivalent=True, neural_calls=0)


def source_contract(inputs):
    """Static anchors not executed: collector, causal predictor, and live transition."""
    exp = REPO/'experiments/spectral_reachability'
    specs = [
        (REPO/'config/train/lewm.yaml', ['history_size: 3','num_preds: 1']),
        (REPO/'config/train/data/pusht.yaml', ['frameskip: 5','pusht_expert_train.lance','${num_preds} + ${history_size}']),
        (REPO/'train.py', ['frameskip * dataset.get_dim("action")']),
        (SITE/'data/dataset.py', ['self.span = num_steps * frameskip','range(length - self.span + 1)']),
        (SITE/'data/formats/lance.py', ['blobs = values[:: self.frameskip]','self.offsets[ep_idx] + start']),
        (SITE/'world/world.py', ["ep['action'].append(ep['action'].pop(0))",'self.envs.step(actions, mask=mask)','on_step(self)']),
        (SITE/'wrapper/default.py', ["info['action'] = action"]),
        (SITE/'wm/lewm/module.py', ['kernel_size=1, stride=1','is_causal=causal','self.pos_embedding[:, :T]','self.transformer(x, c)']),
        (REPO/'module.py', ['kernel_size=1, stride=1','is_causal=causal']),
        (SITE/'wm/lewm/lewm.py', ["'b t d -> (b t) d'","'(b t) d -> b t d'"]),
        (REPO/'jepa.py', ['self.predictor(emb, act_emb)']),
        (exp/'preflight_pusht_d2p_p0.py', ['recorder.execute(packets.repeat_command(action))','recorder.frames[:-1]','return frames[1:]+[current], actions, goal']),
        (exp/'pusht_d2p_packet_contract.py', ['for command in commands:','env.step(command.copy())']),
        (exp/'pusht_composition_cuda_replay_v2.py', ['torch.cat((history, token[None, None]), dim=1)','[:, -1]','history = outgoing[:, 1:]']),
        (exp/'pusht_full_packet_compiler_20261027.py', ['[:, -1]']),
        (exp/'p4_feedback_matched_20261027_v2.py', ['np.stack([between[1],executed_packet])']),
        (exp/'p4_multiround_feedback_runtime_20261027_v1.py', ["observed if arm=='actual' else predicted",'feedback.advance_observed']),
    ]
    evidence = []
    for path, snippets in specs:
        text = inputs.raw(path).decode()
        compact = ''.join(text.split())
        for snippet in snippets:
            require(''.join(snippet.split()) in compact, 'unsupported temporal source anchor: '+str(path)+' '+snippet)
        evidence.append(dict(path=str(path), sha256=inputs.hash(path), anchors=snippets))
    return evidence


def dataset_witness(path=DATA):
    """Bounded numeric slices only: no whole-44GiB scan, no dataset/model constructor."""
    import h5py
    path = Path(path)
    before = path.stat()
    records = []
    with h5py.File(path,'r') as f:
        schema = {k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in f.items()}
        for ep in (0,1):
            offset, length = int(f['ep_offset'][ep]),int(f['ep_len'][ep])
            start = offset+2
            record = dict(episode=ep,episode_offset=offset,episode_length=length,local_start=2,
                global_start=start,global_stop_exclusive=start+20,slices={})
            require(length >= 22, 'bounded witness episode too short')
            for key in ('action','state','step_idx','episode_idx'):
                a = f[key][start:start+20]
                require(np.isfinite(a).all(),'nonfinite interior numeric witness')
                record['slices'][key] = dict(dtype=str(a.dtype),shape=list(a.shape),
                    sha256=hashlib.sha256(a.tobytes()).hexdigest(),values=a.tolist())
            require(np.all(np.diff(record['slices']['step_idx']['values']) == 1), 'dataset step indices discontinuous')
            tail = f['action'][offset+length-1]
            record['tail_action_nan_mask'] = np.isnan(tail).tolist()
            record['observation_global_rows'] = list(range(start,start+20,5))
            records.append(record)
    after = path.stat()
    require((before.st_size,before.st_mtime_ns,before.st_ino) ==
        (after.st_size,after.st_mtime_ns,after.st_ino),'dataset changed during bounded read')
    return dict(path=str(path),size_bytes=before.st_size,mtime_ns=before.st_mtime_ns,
        schema=schema,records=records,full_file_sha256=None,
        historical_checkpoint_training_dataset_identity_proven=False)


def timing(boundary):
    require(type(boundary) is int and boundary in (5,10,15),'unsupported fixed P4 boundary')
    return dict(boundary=boundary,rgb_state_steps=[boundary-10,boundary-5,boundary],
        between_action_intervals=[[boundary-10,boundary-5],[boundary-5,boundary]],
        proposed_action_interval=[boundary,boundary+5],predict_token_index=2,
        training_target_steps=[boundary-5,boundary,boundary+5],predicted_state_step=boundary+5,
        interval_convention='zero-based outgoing action index; half-open [start,end)')


def unique(events,kind,**fields):
    rows=[e for e in events if e['kind']==kind and all(e.get(k)==v for k,v in fields.items())]
    require(len(rows)==1,'missing/duplicate temporal event: '+kind+str(fields))
    return rows[0]


def audit_arm(p,events,arm,root_hashes):
    ev=[e for e in events if e.get('arm')==arm]
    attempts=[e for e in ev if e['kind']=='step_attempt']
    require([e['primitive_step'] for e in attempts] == list(range(1,21)), 'complete20 ordered action trace required')
    commands=np.asarray([e['command'] for e in attempts])
    require(commands.shape==(20,2) and np.array_equal(commands[:5],p['prefix']), 'prefix/action sequence differs')
    observations=[e for e in ev if e['kind']=='observation']
    require([e['primitive_step'] for e in observations]==list(range(1,21)), 'observation duplicate/gap/order differs')
    obs={e['primitive_step']:e for e in observations}
    require(len(obs)==20 and set(obs)==set(range(1,21)), 'complete20 observation trace required')
    initial=unique(ev,'initial')
    require(initial['pixel_sha256']==root_hashes[-1], 'root boundary0 differs')
    hashes={-10:root_hashes[0],-5:root_hashes[1],0:root_hashes[2],
        **{step:e['pixel_sha256'] for step,e in obs.items()}}
    raw=np.concatenate([np.asarray(p['between_raw']),commands.reshape(4,5,2)],axis=0)
    require(raw.shape==(6,5,2),'exact two past and four executed packets required')
    rows=[]
    previous=None
    for b in (5,10,15):
        begin=unique(ev,'boundary_begin',boundary=b)
        c=unique(ev,'planning_context',boundary=b)
        selection=unique(ev,'selection',boundary=b)
        complete=unique(ev,'packet_complete',boundary=b)
        expected=raw[b//5:b//5+2]
        require(np.array_equal(begin['between_raw'],expected) and
            np.array_equal(selection['incoming_between_raw'],expected), 'between packet time interval mismatch')
        require(begin['frames_pixel_sha256']==[hashes[t] for t in (b-10,b-5,b)], 'RGB boundary time mismatch')
        require(np.array_equal(selection['commands'],commands[b:b+5]), 'proposed/dispatched packet interval mismatch')
        require(complete['next_boundary']==b+5 and
            np.array_equal(complete['next_between_raw'],raw[b//5+1:b//5+3]), 'successor between packet shift mismatch')
        chosen=c['observed_context' if arm=='actual' else 'predicted_context']
        require(c['chosen_context']==chosen==selection['compiler_plan']['context'], 'conditioning arm/context mismatch')
        for key in ('chosen_context','observed_context','predicted_context'):
            require(np.shape(c[key]['latent_history'])==(3,192) and
                np.shape(c[key]['action_history'])==(2,10),'three-state/two-between layout mismatch')
        if previous is not None:
            require(c['observed_context']['latent_history'][:2] == previous['observed_context']['latent_history'][1:],
                'observed latent boundary shift mismatch')
            require(c['predicted_context']==unique(ev,'packet_complete',boundary=b-5)['predicted_context'],
                'imagined recurrence mismatch')
        last_return=unique(ev,'step_returned',primitive_step=b)
        first_attempt=attempts[b]
        require(last_return['event_index'] < obs[b]['event_index'] < begin['event_index'] <
            c['event_index'] < selection['event_index'] < first_attempt['event_index'], 'boundary dispatch order mismatch')
        rows.append(dict(**timing(b),arm=arm,conditioning='actual_saved_RGB' if arm=='actual' else 'imagined_latents',
            frame_pixel_sha256=begin['frames_pixel_sha256'],selection_sha256=selection['selection_sha256'],
            two_between_packets_match=True,new_packet_matches=True))
        previous=c
    return rows


def audit():
    inputs=saved.Inputs()
    sources=source_contract(inputs)
    probes=marker_probes(inputs)
    checkpoint=inputs.json(CHECKPOINT/'config.json')
    require(checkpoint['action_encoder']['input_dim']==10 and checkpoint['predictor']['num_frames']==3,
        'checkpoint architecture incompatible with visible contract')
    inputs.raw(CHECKPOINT/'README.md')
    prior=inputs.json(PRIOR)
    require(prior['status']=='saved_arithmetic_audit_passed','validated prior outcomes required')
    inputs.seal(prior['audit_source_seal'])
    for path,digest in prior['inputs'].items():
        require(inputs.hash(path)==digest,'previously validated evidence changed: '+path)
    report=inputs.json(RUNTIME)
    folder=RUNTIME.parent
    require(report['status']=='calibration_recorded' and report['protocol_sha256']==inputs.hash(folder/'protocol.json')
        and report['journal_sha256']==inputs.hash(folder/'runtime.events.jsonl'), 'runtime evidence binding differs')
    for seal in report['source_seals']:
        inputs.seal(seal)
    p=inputs.json(folder/'protocol.json')
    events=[saved.strict_json(line) for line in inputs.raw(folder/'runtime.events.jsonl',32*1024*1024).splitlines()]
    require([e['event_index'] for e in events]==list(range(len(events))), 'journal index sequence differs')
    roots=inputs.npy(p['root_rgb'],(3,224,224,3),np.uint8)
    hashes=[saved.rgb_sha(f) for f in roots]
    rows=[row for arm in ('actual','imagined') for row in audit_arm(p,events,arm,hashes)]
    witness=dataset_witness()
    inputs.stable()
    return dict(schema=TAG,status='visible_training_and_saved_P4_timing_consistent',source_evidence=sources,
        metadata_probes=probes,dataset_witness=witness,rows=rows,root_sha256=p['root_sha256'],
        visible_training_contract=dict(history_size=3,num_preds=1,frameskip=5,chunk_raw_rows=20,
            observation_relative_rows=[0,5,10,15],context_relative_rows=[0,5,10],
            target_relative_rows=[5,10,15],context_outgoing_action_intervals=[[0,5],[5,10],[10,15]],
            unused_action_interval_for_this_loss=[15,20],packet_channel_order='time-major xy pairs',
            encoder_conv_axis_order=['batch','packet_channels10','history_time3'],
            predictor_selected_token=2,last_token_target_offset_from_latest_RGB=5),
        original_checkpoint_training_contract_proven=False,specific_temporal_misalignment_found=False,
        checkpoint_architecture=dict(num_frames=3,action_input_dim=10),
        checkpoint_training_config_present=(CHECKPOINT/'config.yaml').exists(),
        configured_lance_present=(DATA.parent/'pusht_expert_train.lance').exists(),
        inputs=inputs.hashes,new_neural_calls=0,new_physics_steps=0,P1_admission=False,P4_admission=False,
        limitations=[
            'Visible current training source/config is not a historical checkpoint training manifest; original frame_skip/num_preds/dataset identity remain unproven.',
            'Configured Lance cache is absent; bounded HDF5 numeric slices corroborate the available data interface, not historical training provenance.',
            'Current collector rotates incoming returned-state actions one row left before writing; its callback records post-step states, not necessarily the reset state. No proof this collector produced the historical cache.',
            'Equal warmup pixels cannot establish elapsed steps alone: relative timestamps rely on bound collection/packet-loop source and dispatch order.',
            'Intervals count outer env.step transitions, not physics substeps or proven dataset wall-clock duration.',
            'No normalization, camera, Q, neural efficacy, force causality or scientific admission conclusion is drawn.'])


def run(output):
    output=Path(output).resolve()
    require(output.parent==REPO/'results' and output.name.startswith(TAG+'_') and not output.exists(),
        'new exclusive output under results required')
    output.mkdir(exist_ok=False)
    # Archive the whole reviewed source surface before reading live audit outcomes.
    own=[Path(__file__).resolve(),REPO/'tests'/('test_'+TAG+'.py'),REPO/'innerdocs'/(TAG+'.md')]
    inputs=saved.Inputs()
    files=own+[Path(saved.__file__).resolve(),REPO/'experiments/spectral_reachability/d2p_prerun_source_seal.py']
    files += [Path(e['path']) for e in source_contract(inputs)]
    files += [v[0] for v in PROBES.values()]
    seal=create_seal(files=list(dict.fromkeys(files)),args_json=json.dumps(dict(schema=TAG,
        runtime_report=str(RUNTIME),prior_outcomes=str(PRIOR),dataset_bounded_read=str(DATA))),output_dir=output/'source_seal')
    result=audit()
    verify_seal(output_dir=seal['output_dir'],expected_manifest_sha256=seal['manifest_sha256'])
    result['audit_source_seal']=seal
    with (output/'report.json').open('x') as stream:
        json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    print(json.dumps(dict(report=str(output/'report.json'),status=result['status'],rows=len(result['rows']),
        original_checkpoint_training_contract_proven=False)))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
