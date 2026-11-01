"""Fixed Jason baseline launch inventory; hashes/JSON only, never executes commands."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil

NAME = 'agent_baseline_launch_readiness_20261028_v1'
ROOT = Path(__file__).resolve().parents[2]
MATRIX = '../docs/D2P_BASELINE_EXECUTION_MATRIX_JASON_20261028_v1.md'
MATRIX_SHA = 'd4e0a2a49dc911d37b534f86451435d8c7d261154bce50ccbf24dc0e1ff1cf3f'
REPORTED = 'results/p2_reported_baseline_table_20261027_v3/export/report.json'
REPORTED_SHA = '52e687811e6d6f728ab3eda60836ea05eff11f36da2bb4af975d1dd19ef380f1'
COMPARISON = 'results/p1_moving_baseline_comparison_20261027_v3_main01/comparison.json'
COMPARISON_SHA = '96cf27b143bb1a839edf7bc1e4de02f4a97e433c6475c4c731c78af76300d32e'
BUDGET = 'results/p1_moving_baseline_budget_matrix_20261027_v1_main01/budget_matrix.json'
BUDGET_SHA = '721d978d29516f8ecf1d1056e159736ef231723e6849490a83678bd76baa1ebc'
HILEWM_HELPER = 'experiments/spectral_reachability/agent_hilewm_asset_verify_20261028_v1.py'
HILEWM_HELPER_SHA = 'b86bf740654125b14b4bcc3e2d0d714830c89136492e19442ac8e4fa7b2ea447'
HILEWM_FAILURE = 'results/hilewm_archive_MAIN_20261028_v1/report.json'
HILEWM_FAILURE_SHA = '19c526da4a8e3da6b340addd23289df2bd3a57e5a18169290f9bd13dd4f0783b'
TASK_SHAS = ['a42bfbb1ea692782355e2572a5df2cd048b72cf3e6c41fee40eb12503674959b',
             '101add21a7a40591167a6ab8e8b22c3c62cb1fd700c741fd36cd2073e2d5df74']
MODEL_SHA = '4b12c01f5859f4b46bea478f3a8d5e2d3f516df456a622a3feec31bde7e33546'


@dataclass(frozen=True)
class Workflow:
    key: str
    method: str
    module: str
    source_sha: str
    prepared: str
    prepare_sha: str
    output: str
    report_sha: str
    status: str
    binding_count: int
    artifact_count: int
    wall: int


WORKFLOWS = (
    Workflow('moving_grasp', 'GRASP', 'p1_moving_grasp_baseline_cuda_20261027_v1',
             '52d3157574e4e2726f6d88b43bd7e5dc804ce79730b65512888e63e0dd7c7106',
             'results/p1_moving_grasp_baseline_cuda_20261027_v1_prepare01.json',
             'd1f6a9ac66774759ff47aa02cbfb00132399da5971f0458500deec359b2cca87',
             'results/p1_moving_grasp_baseline_cuda_20261027_v1_cuda_main01',
             '7593940c1503416f42ce87a5d8135e8512763121e027c0ef98dbe614bfb31f91',
             'moving_grasp_model_plans_sealed_not_admitted', 139, 6, 600),
    Workflow('matched_sampling', 'LeWM', 'moving_prediction_matched_sampling_cuda_20261027_v1',
             '72195df2a7f1d10948c2bf5c494dd26e66400eaa2e269737332cbd876e599d40',
             'results/moving_prediction_matched_sampling_cuda_20261027_v1_prepare_MAIN01.json',
             '4e4eededb3f8d498ff0c7993e47e7f8935129d99ebdd02270f30de9aae0dbf06',
             'results/moving_prediction_matched_sampling_cuda_20261027_v1_cuda_MAIN01',
             '60872ad114700731d82d293a616a4dede54a53b7be6514ad0cdf231d838d3ed8',
             'matched_sampling_plans_sealed_not_admitted', 129, 10, 900),
    Workflow('moving_gradient', 'LeWM', 'p1_moving_gradient_baseline_cuda_20261027_v1',
             '3224d6a094e74b3845c2c695c6c85fe29c6522cd4a88c91c57f2cc03d127a70e',
             'results/p1_moving_gradient_baseline_cuda_20261027_v1_prepare_main01.json',
             'c7537a7d4aacd37dca797b6dd92e7d8492dd5a51472d63b60d97e35d7d957328',
             'results/p1_moving_gradient_baseline_cuda_20261027_v1_cuda_main01',
             '96c140caee14bba43bf766d981c21be1724a36c0047fde4abcc2892c24bf5b7d',
             'moving_gradient_model_plans_sealed_not_admitted', 121, 6, 600),
)


def issue(code, path, expected=None, actual=None, detail=None):
    return dict(code=code, path=str(path), expected=expected, actual=actual, detail=detail)


class Files:
    """No recursion/globs/import of runners. Cache hash results across fixed recipes."""
    def __init__(self, root):
        self.root = root.resolve()
        self.hashes = {}
        self.bytes_hashed = 0

    def path(self, name):
        # Preserve venv interpreter symlink: resolving it can silently select system Python.
        return Path(os.path.abspath(self.root / name))

    def allowed(self, path):
        return (path == self.root.parent / 'docs/D2P_BASELINE_EXECUTION_MATRIX_JASON_20261028_v1.md'
                or path.is_relative_to(self.root)
                or path.is_relative_to(Path('/home/gbk/.stable_worldmodel/checkpoints/pusht/lewm')))

    def digest(self, path):
        path = Path(path).resolve()
        if not self.allowed(path):
            raise ValueError('outside fixed baseline paths: ' + str(path))
        if path not in self.hashes:
            h = hashlib.sha256()
            with path.open('rb') as f:
                before = os.fstat(f.fileno())
                for b in iter(lambda: f.read(1024 * 1024), b''):
                    self.bytes_hashed += len(b)
                    h.update(b)
                after = os.fstat(f.fileno())
                if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                    raise ValueError('file changed while hashing: ' + str(path))
            self.hashes[path] = h.hexdigest()
        return self.hashes[path]

    def json(self, path):
        with Path(path).open('rb') as f:
            raw = f.read(2 * 1024 * 1024 + 1)
        if len(raw) > 2 * 1024 * 1024:
            raise ValueError('metadata >2MiB')
        return json.loads(raw)

    def is_file(self, path):
        return Path(path).is_file()


def check(files, path, expected, errors):
    try:
        if not re.fullmatch('[0-9a-f]{64}', expected or ''):
            raise ValueError('expected SHA256 unavailable')
        actual = files.digest(path)
        if actual != expected:
            errors.append(issue('sha_mismatch', path, expected, actual))
            return False
        return True
    except Exception as exc:
        errors.append(issue(type(exc).__name__, path, expected, detail=str(exc)))
        return False


def read_pinned(files, path, expected, errors):
    if check(files, path, expected, errors):
        try:
            return files.json(path)
        except Exception as exc:
            errors.append(issue('invalid_metadata', path, expected, detail=str(exc)))
    return None


def command(files, module, *args, role='MAIN_saved_only'):
    argv = [str(files.path('.venv/bin/python')), '-B', '-m',
            'experiments.spectral_reachability.' + module, *map(str, args)]
    env = {'PYTHONDONTWRITEBYTECODE': '1', 'OPENBLAS_NUM_THREADS': '1',
           'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
    return dict(cwd=str(files.root), argv=argv, env=env, role=role,
                shell=shlex.join(['env', *[k+'='+v for k,v in env.items()], *argv]), executed=False)


def local_workflow(files, w, shared_ok):
    errors = []
    prepared, output = files.path(w.prepared), files.path(w.output)
    source = files.path('experiments/spectral_reachability/' + w.module + '.py')
    check(files, source, w.source_sha, errors)
    p = read_pinned(files, prepared, w.prepare_sha, errors)
    r = read_pinned(files, output/'report.json', w.report_sha, errors)
    closure = 0
    if p is not None:
        if p.get('schema') != w.module+'.prepare' or p.get('expected_model_hash') != MODEL_SHA:
            errors.append(issue('prepare_protocol', prepared, w.module+'.prepare', p.get('schema')))
        slots = p.get('slots', [])
        if ([s.get('source_slot') for s in slots] != [0,1]
                or [s.get('task_sha256') for s in slots] != TASK_SHAS):
            errors.append(issue('fixed_two_root_identity', prepared, TASK_SHAS))
        bindings = p.get('bindings_sha256', {})
        if not isinstance(bindings, dict) or len(bindings) != w.binding_count:
            errors.append(issue('closure_count', prepared, w.binding_count,
                                len(bindings) if isinstance(bindings, dict) else None))
        else:
            for name, expected in bindings.items():
                path = Path(name)
                if not path.is_absolute():
                    errors.append(issue('nonabsolute_closure_path', name, expected))
                else:
                    closure += int(check(files, path, expected, errors))
        c = p.get('contract', {})
        if c.get('P1_admission') is not False or c.get('training') is not False or c.get('with_target') is not False:
            errors.append(issue('frozen_protocol_scope', prepared, 'P1=false training=false with_target=false'))
        if w.key == 'moving_gradient' and c.get('feedback_MPC') is not False:
            errors.append(issue('gradient_is_open_loop', prepared, False, c.get('feedback_MPC')))
    if r is not None:
        if (r.get('schema') != w.module+'.report' or r.get('status') != w.status or r.get('errors') != []
                or r.get('prepare_sha256') != w.prepare_sha or r.get('prepared') != str(prepared)):
            errors.append(issue('saved_report_binding', output/'report.json', w.status, r.get('status')))
        artifacts = r.get('artifact_sha256', {})
        if not isinstance(artifacts, dict) or len(artifacts) != w.artifact_count:
            errors.append(issue('artifact_count', output/'report.json', w.artifact_count))
        else:
            for name, expected in artifacts.items():
                if '/' in name or '\\' in name or name in ('', '.', '..') or Path(name).is_absolute():
                    errors.append(issue('unsafe_artifact_name', name))
                else:
                    check(files, output/name, expected, errors)
    if not files.is_file(files.path('.venv/bin/python')):
        errors.append(issue('interpreter_missing', files.path('.venv/bin/python')))
    ready = shared_ok and not errors
    commands = []
    if ready:
        commands = [command(files, w.module, 'verify', '--prepared', prepared, '--prepare-sha', w.prepare_sha),
                    command(files, w.module, 'consume', '--output', output, '--report-sha', w.report_sha)]
    return dict(workflow=w.key, method=w.method,
                status='ready_for_MAIN_dry_preflight' if ready else 'blocked',
                errors=errors, closure_files_verified=closure,
                main_dry_preflight=commands, strict_verifier_executed=False,
                strict_consumer_executed=False, cuda_launch_authorized=False,
                actual_run_command=None, new_run_required=False,
                local_scope='known2 moving calibration, not official replication or fresh roots',
                source=str(source), source_sha256=w.source_sha,
                prepared=str(prepared), prepare_sha256=w.prepare_sha,
                saved_report=str(output/'report.json'), saved_report_sha256=w.report_sha,
                wall_seconds=w.wall, output_cap_bytes=64*1024**2,
                disk_floor_bytes=200*1024**2,
                resource_claim='separate API/samples/AD/encoder/physics/wall axes; never total FLOPs equality')


def build(files):
    shared_errors = []
    shared_ok = check(files, files.path(MATRIX), MATRIX_SHA, shared_errors)
    evidence = []
    for kind, name, expected in [('author_reported_context_only', REPORTED, REPORTED_SHA),
                                 ('local_saved_comparison', COMPARISON, COMPARISON_SHA),
                                 ('local_resource_vector', BUDGET, BUDGET_SHA)]:
        errors = []
        okay = check(files, files.path(name), expected, errors)
        evidence.append(dict(kind=kind, path=str(files.path(name)), expected_sha256=expected,
                             file_verified=okay, errors=errors, affects_launch_readiness=False,
                             values_not_used_for_readiness=True))
    workflows = [local_workflow(files, w, shared_ok) for w in WORKFLOWS]
    h_errors = []
    helper_ok = check(files, files.path(HILEWM_HELPER), HILEWM_HELPER_SHA, h_errors)
    failure = read_pinned(files, files.path(HILEWM_FAILURE), HILEWM_FAILURE_SHA, h_errors)
    h_commands = []
    if helper_ok and failure is not None:
        h_commands = [command(files, 'agent_hilewm_asset_verify_20261028_v1', 'receipt',
                              '--download-report', files.path(HILEWM_FAILURE),
                              '--download-report-sha256', HILEWM_FAILURE_SHA)]
        h_commands[0]['expected_exit_code'] = 1
    h_missing = [issue('complete_high_object_missing',
                       'package/checkpoints/pusht/main/pusht_hi_lewm_epoch15_object.ckpt',
                       {'expanded_bytes':122424517, 'crc32':'d59f865b', 'author_member_sha256':None},
                       detail='Need MAIN verified complete object/whole archive; no partial read or retry.'),
                 issue('normalizer_pairing_unverified', 'pusht_expert_train + main object',
                       detail='Need exact dataset/statistics and embedded-low tensor/config identity; no refit.'),
                 issue('macro_bank_unverified', 'Hi-LeWM-C empirical bank',
                       detail='Need selected protocol/encoder/dataset/row indices/seed/horizon/bank SHA receipt.'),
                 issue('trusted_official_runner_unestablished', 'archive code/h_le_wm/eval/hierarchical.py',
                       detail='Only saved source texts; dependency/class/license/MAIN CUDA validation missing.')]
    methods = []
    for method in ('GRASP', 'Hi-LeWM', 'LeWM', 'PlanWAM'):
        own = [w for w in workflows if w['method'] == method]
        if method == 'Hi-LeWM':
            missing, official = h_missing, h_missing
        elif method == 'PlanWAM':
            missing = [issue('official_code_checkpoint_missing', 'PlanWAM official inference code + checkpoint/config',
                             detail='Need actual local paths, immutable revisions, digests and license; unknown.'),
                       issue('driving_protocol_missing', 'NAVSIM-v2 / HUGSIM evaluator and split',
                             detail='Need four-camera/ego history manifest, evaluator and exact split/config SHA.'),
                       issue('local_entrypoint_unknown', 'PlanWAM', detail='No authenticated local CLI. Do not invent planwam.run or substitute PushT.')]
            official = missing
        else:
            missing = [e for w in own for e in w['errors']] + list(shared_errors)
            official = [issue('paper_protocol_unbound', method+' paper protocol',
                              detail='Need per-table checkpoint/config/data/seed/optimizer budget binding; local adaptation is not official replication.')]
            if method == 'GRASP':
                official.append(issue('official_backbone_unbound', 'GRASP DINO-WM checkpoint/config',
                                      detail='Pinned solver text exists; paper experimental weights/config equivalence unknown.'))
            else:
                official.append(issue('official_eval_disallowed', 'eval.py / config/eval/pusht.yaml',
                                      detail='Existing entry fits StandardScaler and permits CPU fallback; no dispatch under frozen-normalizer/CUDA-only rule.'))
        methods.append(dict(method=method,
                            local_status='ready_for_MAIN_dry_preflight' if own and all(w['status'].startswith('ready_') for w in own) else 'blocked',
                            local_workflows=[w['workflow'] for w in own], missing=missing,
                            official_status='blocked', official_missing=official,
                            official_run_command=None, baseline_runnable=None,
                            diagnostic_commands=h_commands if method == 'Hi-LeWM' else [],
                            diagnostic_errors=h_errors if method == 'Hi-LeWM' else []))
    return dict(schema=NAME+'.report', matrix_path=str(files.path(MATRIX)), matrix_sha256=MATRIX_SHA,
                shared_errors=shared_errors, methods=methods, local_workflows=workflows,
                evidence_columns=evidence, author_values_are_local_results=False,
                all_commands_generated_not_executed=True, scientific_admission=False,
                independent_freshness=False, P1_admission=False,
                network_calls=0, neural_calls=0, physics_calls=0,
                raw_archive_or_pickle_opened=False)


def propose_run(files, report, workflow, output, purpose):
    """Generate only; never dispatch, prepare, create output, or authorize a launch."""
    row = next(w for w in report['local_workflows'] if w['workflow'] == workflow)
    if row['status'] != 'ready_for_MAIN_dry_preflight':
        raise ValueError('blocked workflow: no run command')
    if not purpose.strip():
        raise ValueError('MAIN must state a new experiment purpose; existing results need no rerun')
    output = Path(output)
    if (not output.is_absolute() or output.exists() or not output.parent.is_dir()
            or not output.is_relative_to(files.root/'results')):
        raise ValueError('need NEW absolute results output with existing parent')
    if shutil.disk_usage(output.parent).free < row['disk_floor_bytes'] + row['output_cap_bytes']:
        raise ValueError('64MiB output plus 200MiB floor unavailable')
    module = next(w.module for w in WORKFLOWS if w.key == workflow)
    row['actual_run_command'] = command(files, module, 'run', '--prepared', row['prepared'],
                                        '--prepare-sha', row['prepare_sha256'], '--output', output,
                                        role='MAIN_HOST_CUDA_only_after_strict_verify_and_new_authorization')
    row['main_new_purpose'] = purpose
    row['remaining_launch_gates'] = ['MAIN code review and actual strict verify/consume',
                                    'MAIN exclusive HOST CUDA/device/version/memory approval',
                                    'No old result overwrite; no automatic physics/RGB dispatch']


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, default=ROOT)
    p.add_argument('--method', choices=['all','GRASP','Hi-LeWM','LeWM','PlanWAM'], default='all')
    p.add_argument('--new-run-workflow', choices=[w.key for w in WORKFLOWS])
    p.add_argument('--new-run-output', type=Path)
    p.add_argument('--main-purpose', default='')
    args = p.parse_args(argv)
    try:
        files = Files(args.project)
        report = build(files)
        if args.new_run_workflow or args.new_run_output:
            if not (args.new_run_workflow and args.new_run_output):
                raise ValueError('new run requires workflow and actual NEW output')
            target = next(w.method for w in WORKFLOWS if w.key == args.new_run_workflow)
            if args.method not in ('all', target):
                raise ValueError('new run workflow outside selected method')
            propose_run(files, report, args.new_run_workflow, args.new_run_output, args.main_purpose)
        if args.method != 'all':
            report['methods'] = [r for r in report['methods'] if r['method'] == args.method]
            report['local_workflows'] = [r for r in report['local_workflows'] if r['method'] == args.method]
        report['hash_reads_unique'] = len(files.hashes)
        report['bytes_hashed'] = files.bytes_hashed
        raw = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        if len(raw.encode()) > 2 * 1024 * 1024:
            raise ValueError('report exceeds 2MiB')
        print(raw)
        return int(bool(report['shared_errors']) or any(r['local_status']=='blocked' for r in report['methods']))
    except Exception as exc:
        print(json.dumps({'schema':NAME+'.error','error':type(exc).__name__,'detail':str(exc)}))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
