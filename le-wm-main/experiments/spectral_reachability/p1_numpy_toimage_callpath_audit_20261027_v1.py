"""CPU-only call-path audit; no model, simulator, network, or production edits."""
from __future__ import annotations

import argparse
import ast
import importlib.metadata
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from experiments.spectral_reachability import p1_feedback_goal_contract_audit_20261027_v1 as base
from experiments.spectral_reachability import p1_checkpoint_action_camera_provenance_20261027_v1 as prior

TAG = 'p1_numpy_toimage_callpath_audit_20261027_v1'
REPO, SITE = base.REPO, prior.SITE
ONLINE = {
    'eval.py': 'img_transform',
    'experiments/run_online_pusht_planning.py': 'image_transform',
    'experiments/run_live_reacher_planning.py': '_image_transform',
    'experiments/run_live_cube_action_adapter_planning.py': '_image_transform',
    'experiments/run_live_tworoom_planning.py': '_image_transform',
}
TRAIN = {'train.py', 'experiments/train_lewm_directional.py', 'experiments/latent_eigenspace_audit.py'}
EMPIRICAL = {
    'run_lewm_external_pusht_temporal_audit.py', 'run_external_trajectory_operator_audit.py',
    'run_external_action_permutation_audit.py', 'run_external_residual_basis_intervention.py',
    'run_external_energy_matched_dynamics_intervention.py', 'run_external_residual_transport_audit.py',
    'run_external_dynamics_basis_intervention.py', 'run_external_temporal_alignment_adapter.py',
    'run_external_band_rollout_intervention.py',
}
REPAIR = 'experiments/spectral_reachability/train_reacher_d2p_dynamics_repair.py'
OLD_AUDIT = 'experiments/spectral_reachability/p1_checkpoint_action_camera_provenance_20261027_v1.py'
SKIP = {'.git', '.venv', 'external', 'results', 'data', 'tests', '__pycache__', '.pytest_cache'}


def python_sources(root):
    """Physical first-party .py tree, including scripts; no symlink traversal."""
    paths = []
    for directory, folders, files in os.walk(root, followlinks=False):
        folders[:] = sorted(d for d in folders if d not in SKIP and not (Path(directory)/d).is_symlink())
        paths.extend(Path(directory)/f for f in sorted(files) if f.endswith('.py') and
                     not (Path(directory)/f).is_symlink())
    return sorted(paths)


def dotted(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return dotted(node.value)+'.'+node.attr
    return ''


def callsites(path, root):
    text = path.read_text()
    tree = ast.parse(text, filename=str(path))
    aliases = {}
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for a in node.names:
                aliases[a.asname or a.name] = (node.module or '')+'.'+a.name
                if a.name in {'get_img_preprocessor', 'image_transform', '_image_transform', 'img_transform'}:
                    imports.append(dict(line=node.lineno, local=a.asname or a.name,
                                        origin=aliases[a.asname or a.name]))
        elif isinstance(node, ast.Import):
            for a in node.names:
                aliases[a.asname or a.name.split('.')[0]] = a.name if a.asname else a.name.split('.')[0]
    rows = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = dotted(node.func)
        first, *rest = name.split('.')
        resolved = '.'.join([aliases.get(first, first), *rest])
        if resolved.split('.')[-1] in {'get_img_preprocessor', 'ToImage', 'to_image'}:
            rows.append(dict(path=str(path.relative_to(root)), line=node.lineno,
                             expression=name, resolved=resolved))
    return sorted(rows, key=lambda r: r['line']), imports


def classify(row):
    p, name = row['path'], row['resolved']
    if p == f'experiments/spectral_reachability/{TAG}.py':
        return 'this offline positive control (not production)'
    if p == OLD_AUDIT:
        return 'prior offline intentional NumPy positive control'
    if p in ONLINE and name == 'torchvision.transforms.v2.ToImage':
        return 'torchvision, not stable_pretraining; online policy supplies CHW Image'
    if p == 'utils.py' and name == 'stable_pretraining.data.transforms.ToImage':
        return 'stable factory; ingress traced separately through dataset/video/cross-environment readers'
    if p == REPAIR and name == 'stable_pretraining.data.transforms.ToImage':
        return 'reference factory only; production dataset transform=None, tensor manual preprocessing'
    if name == 'utils.get_img_preprocessor':
        if p in TRAIN:
            return 'installed HDF5/Lance readers supply CHW/TCHW tensors'
        if p == 'experiments/audit_frozen_cross_environment_prediction.py':
            return 'NumPy -> torch.from_numpy -> permute TCHW -> transform'
        if Path(p).name in EMPIRICAL and p.startswith('experiments/empirical_spectral_audit/'):
            return 'decord NumPy -> shared _load_video_frames TCHW tensor -> transform'
    raise ValueError(f'unclassified transform call: {row}')


def inventory(paths, root):
    rows, imports = [], []
    for path in paths:
        calls, edges = callsites(path, root)
        for row in calls:
            row['classification'] = classify(row)
        rows.extend(calls)
        imports.extend(dict(path=str(path.relative_to(root)), **e) for e in edges)
    return dict(scanned_python_files=len(paths), calls=rows, factory_import_edges=imports,
                limitation='Static explicit calls/import aliases only; not a proof about dynamic imports, '
                           'getattr, external plugins, excluded trees, or historical training runs.')


def runtime_checks(images):
    """Execute actual local factory/method AST only, never import planner modules."""
    import torch
    from PIL import Image
    import stable_pretraining as spt
    from torchvision import tv_tensors
    from torchvision.transforms import v2 as transforms
    torch.set_num_threads(1)
    ns = dict(torch=torch, np=np, tv_tensors=tv_tensors, transforms=transforms, spt=spt)
    prepare = prior.function(SITE/'stable_worldmodel/policy.py', '_prepare_info', ns)
    make_train = prior.function(REPO/'utils.py', 'get_img_preprocessor', dict(dt=spt.data))
    stable = make_train('pixels', 'pixels', 224)
    online_rows, positive_rows, p1_rows = [], [], []
    p1_path = REPO/'experiments/spectral_reachability/preflight_pusht_d2p_p0.py'
    p1_transform = prior.function(REPO/'experiments/run_pusht_counterfactual.py', 'image_transform', ns)()
    image_tensor = prior.function(p1_path, '_image_tensor', ns)
    frames_tensor = prior.function(p1_path, '_frames_tensor', dict(ns, _image_tensor=image_tensor))
    for identity, rgb in images:
        base.require(rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3, 'HWC uint8 required')
        chw = torch.from_numpy(rgb.copy()).permute(2, 0, 1)
        # No resize: prove CWH branch, even for non-square images, independently of visual impressions.
        bad = spt.data.transforms.ToImage(source='pixels', target='pixels')({'pixels': rgb.copy()})['pixels']
        correct = transforms.ToDtype(torch.float32, scale=True)(transforms.ToImage()(rgb.copy()))
        pil = spt.data.transforms.ToImage(source='pixels', target='pixels')({'pixels': Image.fromarray(rgb)})['pixels']
        tensor = spt.data.transforms.ToImage(source='pixels', target='pixels')({'pixels': chw})['pixels']
        base.require(torch.equal(bad, correct.transpose(-2, -1)), 'stable NumPy branch no longer CWH')
        base.require(torch.equal(pil, correct) and torch.equal(tensor, correct), 'PIL/tensor branch mismatch')
        positive_rows.append(dict(image_id=identity, input_shape=list(rgb.shape),
            bad_shape=list(bad.shape), correct_shape=list(correct.shape),
            numpy_is_exact_xy_transpose=True, pil_and_tensor_correct=True,
            normalized_resized_bad_vs_correct_MAE=float((stable({'pixels':rgb.copy()})['pixels']-
                stable({'pixels':chw})['pixels']).abs().mean()) if rgb.shape[0] == rgb.shape[1] else None))
        # The sealed feedback path does NOT use BasePolicy or either ToImage.
        reference = stable({'pixels':chw.clone()})['pixels']
        p1_single = image_tensor(rgb, transform=p1_transform, device=torch.device('cpu'))
        p1_three = frames_tensor([rgb.copy() for _ in range(3)], transform=p1_transform, device=torch.device('cpu'))
        base.require(torch.equal(p1_single[0,0], reference), 'P1 single frame preprocessing differs')
        base.require(torch.equal(p1_three, reference.expand(1,3,*reference.shape)), 'P1 context preprocessing differs')
        p1_rows.append(dict(image_id=identity, single_shape=list(p1_single.shape),
            three_frame_shape=list(p1_three.shape), max_abs_vs_CHW_reference=0.,
            route='p4.encode_cuda -> p0._frames_tensor -> p0._image_tensor -> '
                  'run_pusht_counterfactual.image_transform; NumPy converted to CHW before transform, no ToImage'))
        for path, factory in ONLINE.items():
            transform = prior.function(REPO/path, factory, ns)
            transform = transform(SimpleNamespace(eval=SimpleNamespace(img_size=224))) if path == 'eval.py' else transform()
            expected = transform(chw.clone())
            base.require(torch.equal(expected, stable({'pixels':chw.clone()})['pixels']), 'CHW pipeline differs')
            # torchvision's own direct NumPy branch is also correct; do not conflate class names.
            base.require(torch.equal(transform(rgb.copy()), expected), 'torchvision direct NumPy path differs')
            seen = []
            def spy(value):
                seen.append(dict(type=type(value).__module__+'.'+type(value).__name__,
                                 shape=list(value.shape), dtype=str(value.dtype)))
                base.require(isinstance(value, tv_tensors.Image), 'policy did not wrap as Image')
                base.require(torch.equal(value, chw), 'policy changed channel/spatial order')
                return transform(value)
            batch = np.broadcast_to(rgb, (2, 3, *rgb.shape)).copy()
            before = batch.copy()
            result = prepare(SimpleNamespace(transform={'pixels':spy, 'goal':spy}),
                             {'pixels':batch, 'goal':batch})
            for key in ('pixels', 'goal'):
                base.require(torch.equal(result[key], expected.expand(2, 3, *expected.shape)), 'policy output differs')
            base.require(np.array_equal(batch, before) and len(seen) == 12, 'mutation or frame count mismatch')
            online_rows.append(dict(image_id=identity, factory=path, policy_spy_calls=len(seen),
                observed_transform_input=seen[0], input_shape=list(batch.shape),
                output_shape=list(result['pixels'].shape), direct_numpy_vs_CHW_max_abs=0.,
                policy_vs_CHW_max_abs=0., input_unchanged=True))
    return dict(positive_controls=positive_rows, online=online_rows, sealed_feedback_preprocessing=p1_rows,
                scope='Actual factory and BasePolicy method AST replay on CPU with dummy self; '
                      'not a live rollout, model encode, or historical checkpoint execution.')


def evidence(paths):
    """Source evidence spans for ingress, not a runtime assertion about old training."""
    needles = {
        SITE/'stable_worldmodel/data/formats/hdf5.py': ['torch.from_numpy', 'permute(0, 3, 1, 2)', 'self.transform'],
        SITE/'stable_worldmodel/data/formats/lance.py': ['torch.from_numpy', 'permute(2, 0, 1)', 'self.transform'],
        REPO/'experiments/audit_frozen_cross_environment_prediction.py': ['torch.from_numpy(frames).permute(0, 3, 1, 2)'],
        REPO/'experiments/empirical_spectral_audit/run_lewm_external_pusht_temporal_audit.py':
            ['torch.from_numpy(reader.get_batch(frame_indices).asnumpy())', 't h w c -> t c h w'],
        REPO/REPAIR: ['transform=None', 'def _preprocess_pixels', 'pixels.dtype != torch.uint8'],
        REPO/'experiments/spectral_reachability/preflight_pusht_d2p_p0.py':
            ['from experiments.run_pusht_counterfactual import',
             'transform(torch.from_numpy(image).permute(2, 0, 1))',
             '_image_tensor(frame, transform=transform, device=device)'],
        REPO/'experiments/spectral_reachability/p4_feedback_matched_20261027_v2.py':
            ['pixels = p0._frames_tensor(frames,transform=transform,device=device)'],
        REPO/'experiments/spectral_reachability/pusht_geometry_feedback_response_20261027_v1.py':
            ['transform = base.p0.image_transform()', 'feedback.encode_cuda(model,device,actual_frames,between,base.p0,transform)'],
    }
    for name in EMPIRICAL:
        path = REPO/'experiments/empirical_spectral_audit'/name
        pixel_name = 'raw_frames' if name == 'run_lewm_external_pusht_temporal_audit.py' else 'frames'
        needles.setdefault(path, []).extend(['_load_video_frames(', 'preprocess({"pixels": '+pixel_name+'})'])
    records = []
    for path, terms in needles.items():
        lines = path.read_text().splitlines()
        for term in terms:
            matches = [dict(line=i+1, text=line) for i,line in enumerate(lines) if term in line]
            base.require(matches, f'ingress source evidence missing: {path}: {term}')
            records.append(dict(path=str(path), term=term, matches=matches))
    # No first-party production caller/import of the Reacher reference factory was found.
    repair_module = REPAIR[:-3].replace('/', '.')
    for path in paths:
        if path == REPO/REPAIR:
            continue
        calls, imports = callsites(path, REPO)
        base.require(not any(e['origin'] == repair_module+'._image_transform' for e in imports),
                     'Reacher reference factory acquired production importer')
    return records


def checkpoint_records(inputs):
    checkpoint = prior.CHECKPOINT
    cache = checkpoint/'.cache/huggingface/download'
    records = {}
    for name in ('.gitattributes', 'README.md', 'config.json', 'weights.pt'):
        content = inputs.bind(checkpoint/name)
        meta = inputs.bind(cache/(name+'.metadata'))
        records[name] = prior.metadata_record(meta, content, lfs=name == 'weights.pt')
    base.require(len({v['revision'] for v in records.values()}) == 1, 'checkpoint revision mismatch')
    readme = inputs.bind(REPO/'README.md').read_text().splitlines()
    provenance_sources = [inputs.bind(REPO/'config/train/data/pusht.yaml'),
                          inputs.bind(REPO/'experiments/checkpoint_compat.py'),
                          inputs.bind(SITE/'stable_worldmodel/wm/utils.py')]
    return dict(objects=records, local_files=sorted(str(p.relative_to(checkpoint)) for p in checkpoint.rglob('*') if p.is_file()),
                model_card=(checkpoint/'README.md').read_text(), config=json.loads((checkpoint/'config.json').read_text()),
                project_HF_instructions=[dict(line=i+1,text=v) for i,v in enumerate(readme) if 'quentinll/lewm-pusht' in v],
                source_paths=[str(p) for p in provenance_sources],
                current_training_data_config=provenance_sources[0].read_text(),
                current_default_lance_exists=(checkpoint.parents[2]/'datasets/pusht_expert_train.lance').exists(),
                downloaded_repo_identity_authenticated=False,
                actual_training_run_identified=False, authoritative_training_normalizer_found=False,
                authoritative_camera_transform_found=False,
                provenance_boundary='Local metadata verifies cached object digests/revision assertions only. '
                    'Model card dataset tag is not training-run, normalizer, camera, reader-version, or codec provenance. '
                    'No network, weights unpickling, or dataset-wide read. Current source is not historical run proof.')


def audit(output):
    from experiments.spectral_reachability import d2p_prerun_source_seal as seal
    base.require(not output.exists(), 'exclusive NEW output directory required')
    inputs = base.Inputs()
    paths = python_sources(REPO)
    for path in paths:
        inputs.bind(path)
    own = [Path(__file__), REPO/f'tests/test_{TAG}.py', REPO/f'innerdocs/{TAG}.md',
           Path(base.__file__), Path(prior.__file__), Path(seal.__file__)]
    critical = [SITE/'stable_worldmodel/policy.py', SITE/'stable_pretraining/data/transforms.py',
                SITE/'stable_pretraining/data/dataset_stats.py',
                SITE/'stable_worldmodel/data/formats/hdf5.py', SITE/'stable_worldmodel/data/formats/lance.py',
                REPO/'utils.py', REPO/'config/train/data/pusht.yaml', REPO/REPAIR,
                REPO/'README.md', REPO/'experiments/checkpoint_compat.py', SITE/'stable_worldmodel/wm/utils.py',
                REPO/'experiments/run_pusht_counterfactual.py',
                REPO/'experiments/spectral_reachability/preflight_pusht_d2p_p0.py',
                REPO/'experiments/spectral_reachability/p4_feedback_matched_20261027_v2.py',
                REPO/'experiments/spectral_reachability/pusht_geometry_feedback_response_20261027_v1.py',
                *(REPO/p for p in ONLINE)]
    critical.extend(SITE/'torchvision/transforms'/p for p in (
        'v2/_type_conversion.py', 'v2/functional/_type_conversion.py',
        'v2/_misc.py', 'v2/functional/_misc.py', 'v2/_geometry.py',
        'v2/functional/_geometry.py', 'functional.py', '_functional_pil.py', '_functional_tensor.py'))
    for path in own+critical:
        inputs.bind(path)
    feedback = inputs.load(REPO/'results/pusht_geometry_feedback_response_20261027_v1_prepare_v1/protocol.json')
    base.require(feedback['root_sha256'] == base.ROOT, 'fixed original root identity differs')
    rgb_path = inputs.bind(feedback['root_rgb'], feedback['bindings'][feedback['root_rgb']])
    checkpoint = checkpoint_records(inputs)
    output.mkdir(parents=True)
    (output/'checkpoint_receipt.json').write_text(json.dumps(checkpoint, indent=2, sort_keys=True)+'\n')
    inputs.bind(output/'checkpoint_receipt.json')
    # Full-tree hashes may exceed the bounded snapshot tool's 1 MiB/file limit.
    # Anchor this complete manifest by digest in the small snapshotted protocol.
    (output/'input_manifest.json').write_text(json.dumps(inputs.bindings.copy(), indent=2, sort_keys=True)+'\n')
    inputs.bind(output/'input_manifest.json')
    protocol = dict(schema=TAG, P1=False, fixed_root_index=1, root_sha256=base.ROOT, root_rgb=str(rgb_path),
        input_manifest_sha256=base.sha(output/'input_manifest.json'),
        input_manifest_path=str(output/'input_manifest.json'), exclusions=sorted(SKIP),
        package_versions={name:importlib.metadata.version(name) for name in
                          ('torch', 'torchvision', 'stable-pretraining', 'stable-worldmodel', 'numpy', 'Pillow')},
        planned_checks=['static explicit alias inventory', 'ingress source spans', 'CPU actual policy/factory replay',
                        'stable NumPy positive control', 'local cached checkpoint digest lineage'],
        neural_predict=0, neural_encode=0, physics_steps=0, network_requests=0)
    (output/'protocol.json').write_text(json.dumps(protocol, indent=2, sort_keys=True)+'\n')
    inputs.bind(output/'protocol.json')
    source_seal = seal.create_seal(files=list(dict.fromkeys(own+critical+[output/'checkpoint_receipt.json', output/'protocol.json'])),
        args_json=json.dumps(dict(protocol_sha256=base.sha(output/'protocol.json'))), output_dir=output/'source_seal')
    frames = np.load(rgb_path, allow_pickle=False)
    base.require(frames.shape == (3,224,224,3) and frames.dtype == np.uint8, 'sealed root RGB shape')
    images = [('asymmetric_5x7_fixture', np.arange(5*7*3, dtype=np.uint8).reshape(5,7,3))]
    images.extend((f'fixed_root_frame_{i}', frame) for i,frame in enumerate(frames))
    print('CPU-only call-path audit: distinguish a reproducible library defect from actual online ingress.', flush=True)
    static_inventory = inventory(paths, REPO)
    ingress_evidence = evidence(paths)
    runtime = runtime_checks(images)
    base.require(len(runtime['online']) == 20 and sum(v['policy_spy_calls'] for v in runtime['online']) == 240,
                 'online preprocessing budget differs')
    result = dict(schema=TAG, status='offline_audit_recorded', P1=False,
        protocol_sha256=base.sha(output/'protocol.json'), source_seal=source_seal,
        checkpoint_receipt_sha256=base.sha(output/'checkpoint_receipt.json'),
        inventory=static_inventory, ingress_evidence=ingress_evidence,
        runtime=runtime, checkpoint=checkpoint,
        verdict=dict(stable_direct_numpy_HWC_to_CWH_defect_reproduced=True,
            current_online_affected_branch_established=False, production_fix_created=False,
            reason='Online factories use torchvision; actual policy wraps CHW Image before transform. '
                   'Dataset/video/cross-environment paths supply tensors. No established affected online branch warrants a patch.',
            historical_checkpoint_training_camera_or_normalizer_mismatch_proven=False,
            physical_benefit_claim=False, code_wide_exclusion=False),
        counts=dict(fixed_real_RGB=3, asymmetric_fixture_RGB=1, online_factories=5,
                    online_image_factory_pairs=len(runtime['online']),
                    policy_transform_calls=sum(v['policy_spy_calls'] for v in runtime['online']),
                    P1_single_image_checks=len(runtime['sealed_feedback_preprocessing']),
                    P1_three_frame_context_checks=len(runtime['sealed_feedback_preprocessing']),
                    P1_transform_calls=4*len(runtime['sealed_feedback_preprocessing']),
                    neural_predict=0, neural_encode=0, physics_steps=0, network_requests=0))
    inputs.unchanged()
    seal.verify_seal(output_dir=output/'source_seal', expected_manifest_sha256=source_seal['manifest_sha256'])
    result['all_bound_inputs_unchanged'] = True
    result['final_input_bindings'] = inputs.bindings
    (output/'report.json').write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n')
    print(json.dumps(dict(report=str(output/'report.json'), sha256=base.sha(output/'report.json'), P1=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, required=True)
    audit(parser.parse_args().out_dir.resolve())
