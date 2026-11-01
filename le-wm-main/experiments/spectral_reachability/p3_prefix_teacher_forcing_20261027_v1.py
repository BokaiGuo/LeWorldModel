"""Fixed-action causal teacher forcing on real P3 packet boundaries.

Diagnostic only: no optimization, physics, training or scientific admission.
"""
from copy import deepcopy
import argparse
from pathlib import Path
import shutil
import numpy as np
from experiments.spectral_reachability import p3_joint_prefix_cuda_encoding_20261027_v1 as base

NAME = 'p3_prefix_teacher_forcing_20261027_v1'
ROOT = base.REPO
PREFIX = ROOT/'results/p3_joint_prefix_cuda_encoding_20261027_v1_cuda_main01'
PREFIX_SHA = 'ca6cdef177b2b8fe5e9739f2a07d2254b1183730ff9c340f52d96707f322a802'
NUMERIC = ROOT/'results/p3_joint_prefix_cuda_encoding_20261027_v1_numerical_main01/report.json'
NUMERIC_SHA = '147c3cca6a66789fbd5052fc4f4017b4e0a6b63c59038cfac1fbc19c01da77de'
require = base.require


def causal_context(initial, segments, real_prefixes, boundary):
    """Only observed latent boundaries, with unchanged causal action history."""
    require(type(boundary) is int and 1 <= boundary < len(segments), 'interior boundary')
    require(all(k in real_prefixes for k in range(1, boundary+1)), 'all causal observed prefixes required')
    context = deepcopy(segments[boundary-1])
    base.context(initial)
    base.context(context)
    actual = [base.array(real_prefixes[k], (192,)).tolist() for k in range(1, boundary+1)]
    context['latent_history'] = (deepcopy(initial['latent_history']) + actual)[-3:]
    base.context(context)
    return context


def material():
    require(base.sha(PREFIX/'receipt.json') == PREFIX_SHA and base.sha(NUMERIC) == NUMERIC_SHA, 'actual prefix/consumer external hashes')
    receipt = base.read(PREFIX/'receipt.json')
    numeric = base.read(NUMERIC)
    binding = base.read(PREFIX/'execution_binding.json')
    p = base.verify(binding['prepared'], binding['prepare_sha256'])
    require(receipt['status'] == 'all_17_prefix_latents_saved' and receipt['errors'] == [] and
            receipt['source_verified'] is receipt['model_unchanged'] is True and
            numeric['status'] == 'saved_prefix_numerics_verified' and numeric['receipt_sha256'] == PREFIX_SHA,
            'completed original derived observations')
    physics = base.read(p['physical_report'])
    prepared = base.read(p['goal_q_source'])
    planner = base.read(base.PLANNER/'report.json')
    roots = {r['root_sha256']: r for r in planner['roots']}
    enc = {(r['root_sha256'], r['method']): r for r in physics['encodings']}
    observed = {}
    bindings = {str(PREFIX/'receipt.json'): PREFIX_SHA, str(NUMERIC): NUMERIC_SHA}
    for frame, row in zip(p['frames'], receipt['rows']):
        require(row['frame_id'] == frame['frame_id'] and row['status'] == 'encoded_and_saved', 'ordered actual prefix census')
        path = Path(row['latent_path'])
        require(base.sha(path) == row['latent_sha256'], 'actual native prefix hash')
        a = np.load(path, allow_pickle=False)
        require(a.shape == (192,) and a.dtype == np.float32 and np.isfinite(a).all(), 'native prefix192')
        observed.setdefault((frame['root_sha256'], frame['arm']), {})[frame['primitive_step']//5] = a.tolist()
        bindings[str(path)] = row['latent_sha256']
    tasks = []
    for plan in prepared['plans']:
        key = plan['root_sha256'], plan['method']
        arm = roots[key[0]]['arms'][key[1]]
        real = observed[key]
        h = plan['planned_horizon_tokens']
        require(set(real) == set(range(1, h)), 'full observed interior history')
        for b in range(1, h):
            # One next-packet prediction per observed boundary, not a suffix
            # reroll consuming predicted intermediates instead of observations.
            context = causal_context(plan['predictor_context'], arm['segments'], real, b)
            target = real[b+1] if b+1 < h else enc[key]['terminal_vector']
            tasks.append(dict(task_id=f'{len(tasks):02d}', root_sha256=key[0], arm=key[1],
                boundary_step=5*b, target_step=5*(b+1), context=context,
                commands=plan['commands'][5*b:5*(b+1)], actual_target=target,
                imagined_target=arm['segments'][b]['latent_history'][-1]))
    require(len(tasks) == 17, 'sixteen branches plus one H3 interior boundary')
    return dict(schema=NAME+'.protocol', tasks=tasks, prefix_prepare=binding,
                goal_q_source=p['goal_q_source'], goal_q_source_sha256=p['goal_q_source_sha256'],
                bindings=bindings, model_bindings=p['model_bindings'], policy=p['policy'],
                predict_cap=17, action_encoder_cap=17, new_physical_steps=0,
                training=False, P1_admission=False, P3_admission=False)


def prepare(output):
    output = Path(output).absolute()
    require(not output.exists() and shutil.disk_usage(output.parent).free > base.MIN_FREE, 'new bounded prepare/storage reserve')
    p = material()
    sources = base.source_files() + [Path(__file__).resolve(), ROOT/'experiments/spectral_reachability/pusht_composition_cuda_replay_v2.py',
        ROOT/'experiments/spectral_reachability/d2p_composition_witness_v2.py', ROOT/'tests'/('test_'+NAME+'.py')]
    p['sources_sha256'] = {str(f): base.sha(f) for f in sources}
    output.mkdir()
    # Bind original Q matrices by reference, without large input copies.
    raw = (base.io.canonical(p)+'\n').encode()
    require(len(raw) < 4_900_000, 'small protocol')
    with (output/'protocol.json').open('xb') as stream:
        stream.write(raw); stream.flush(); base.os.fsync(stream.fileno())
    seal = base.create_seal(files=sources, args_json=base.io.canonical(dict(protocol_sha256=base.sha(output/'protocol.json'))), output_dir=output/'source_seal')
    base.write(output/'prepare.json', dict(protocol_sha256=base.sha(output/'protocol.json'), source_seal=seal))


def verify(folder, expected):
    folder = Path(folder).absolute()
    require(base.sha(folder/'prepare.json') == expected, 'external prepare hash')
    r = base.read(folder/'prepare.json')
    require(base.sha(folder/'protocol.json') == r['protocol_sha256'], 'protocol hash')
    p = base.read(folder/'protocol.json')
    s = r['source_seal']; base.verify_seal(output_dir=s['output_dir'], expected_manifest_sha256=s['manifest_sha256'])
    rebuilt = material(); rebuilt['sources_sha256'] = p['sources_sha256']
    require(base.same(rebuilt, p), 'causal source reconstruction')
    for path, h in {**p['sources_sha256'], **p['bindings'], **p['model_bindings']}.items():
        require(base.sha(path) == h, 'source/model/observation drift')
    return p


class Cap:
    def __init__(self, journal, sync):
        self.journal, self.sync = journal, sync
        self.entries = []; self.active = None

    def dispatch(self, kind, original, args, kwargs):
        x = args[0]
        require(getattr(x.device, 'type', None) == 'cuda' and tuple(x.shape) ==
            ((1,3,192) if kind == 'predict' else (1,3,10)), 'native CUDA full3 only')
        order = [(f'{i:02d}', k) for i in range(17) for k in ('action_encoder','predict')]
        require(len(self.entries) < 34 and (self.active,kind) == order[len(self.entries)] and
            all(e['completed'] for e in self.entries), 'nonrefundable exact ordered cap/no retries')
        event = dict(task_id=self.active,kind=kind,completed=False)
        self.entries.append(event); self.journal(dict(event='attempt',**event))
        result = original(*args, **kwargs); self.sync(); event['completed'] = True
        self.journal(dict(event='return',**event))
        return result


def run(folder, expected, output):
    p = verify(folder, expected); output = Path(output).absolute()
    require(not output.exists() and shutil.disk_usage(output.parent).free > base.MIN_FREE, 'new run/200MiB reserve')
    base.write(Path(folder).absolute().with_name(Path(folder).name+'.launch.json'),dict(output=str(output),prepare_sha256=expected))
    output.mkdir(); rows=[]; errors=[]; hooks=base.Hooks(); model=None; before=after=None; source_ok=False; host=None
    with (output/'events.jsonl').open('x') as stream:
        def journal(e):
            stream.write(base.io.canonical(e)+'\n'); stream.flush(); base.os.fsync(stream.fileno())
        cap=None
        try:
            import torch
            from torch.nn.attention import SDPBackend, sdpa_kernel
            from experiments.checkpoint_compat import load_compatible_pretrained
            from experiments.gpu_preflight import require_cuda_allocation
            from experiments.spectral_reachability.pusht_composition_cuda_replay_v2 import CudaReplay
            host=require_cuda_allocation('cuda:0'); device=torch.device('cuda:0')
            torch.backends.cuda.matmul.allow_tf32=False
            model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
            base.frozen(model); before=base.tensor_hash(model)
            cap=Cap(journal,lambda:torch.cuda.synchronize(device))
            for obj,key,kind in ((model,'predict','predict'),(model.action_encoder,'forward','action_encoder')):
                original=getattr(obj,key)
                def call(*args,_kind=kind,_original=original,**kwargs):return cap.dispatch(_kind,_original,args,kwargs)
                hooks.install(obj,key,call)
            replay=CudaReplay(model,device)
            require(base.sha(p['goal_q_source'])==p['goal_q_source_sha256'],'original Q bytes')
            root_metrics=base.read(p['goal_q_source'])['root_metrics']
            for task in p['tasks']:
                cap.active=task['task_id']; journal(dict(event='task_intent',task_id=cap.active))
                with sdpa_kernel(SDPBackend.MATH):
                    predicted=replay(task['context'],task['commands'])
                # Persist raw context before any scoring can fail.
                base.write(output/(cap.active+'.context.json'),predicted)
                metric=root_metrics[task['root_sha256']]
                qs={k:v['projection'] for k,v in metric['metrics'].items()}
                scores=base.numeric(task['actual_target'],predicted['latent_history'][-1],metric['goal'],qs)
                original_scores=base.numeric(task['actual_target'],task['imagined_target'],metric['goal'],qs)
                row=dict(task_id=cap.active,root_sha256=task['root_sha256'],arm=task['arm'],
                    boundary_step=task['boundary_step'],target_step=task['target_step'],teacher_forced=scores,imagined=original_scores)
                base.write(output/(cap.active+'.result.json'),row); rows.append(row)
        except BaseException as exc:errors.append(base.error('execution',exc))
        finally:
            hooks.restore(errors)
            if model is not None:
                try:base.frozen(model); after=base.tensor_hash(model)
                except BaseException as exc:errors.append(base.error('final_model',exc))
            try:verify(folder,expected); source_ok=True
            except BaseException as exc:errors.append(base.error('final_source',exc))
            complete=not errors and len(rows)==17 and source_ok and before is not None and before==after and cap is not None and len(cap.entries)==34 and all(e['completed'] for e in cap.entries)
            report=dict(status='teacher_forcing_completed' if complete else 'partial_or_failed',rows=rows,errors=errors,
                entries=[] if cap is None else cap.entries,prepare_sha256=expected,model_hash_before=before,model_hash_after=after,
                host=host,journal_sha256=base.sha(output/'events.jsonl'),
                model_unchanged=before is not None and before==after,source_verified=source_ok,
                new_encoder_calls=0,new_predict_calls=0 if cap is None else sum(e['kind']=='predict' for e in cap.entries),
                new_AD_calls=0,new_physical_steps=0,training=False,P1_admission=False,P3_admission=False,
                scope='Known calibration trajectories; fixed actions and real causal packet histories; diagnostic, not feedback control efficacy')
            base.write(output/'report.json',report)
    require(complete,'teacher forcing incomplete; inspect preserved report, never retry automatically')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','verify','run'))
    parser.add_argument('--prepared',type=Path);parser.add_argument('--prepare-sha');parser.add_argument('--output',type=Path)
    a=parser.parse_args()
    if a.command=='prepare':prepare(a.output);print(base.sha(a.output/'prepare.json'))
    elif a.command=='verify':print(len(verify(a.prepared,a.prepare_sha)['tasks']))
    else:
        report=run(a.prepared,a.prepare_sha,a.output);print(base.io.canonical({k:v for k,v in report.items() if k not in ('rows','entries')}))


if __name__=='__main__':main()
