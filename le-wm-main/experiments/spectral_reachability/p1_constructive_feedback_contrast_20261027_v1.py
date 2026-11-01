"""Saved-only same-prefix suffix contrast. No model, physics, fit or selection.

This audits already selected actions; observed outcomes never select a plan.
External report hashes must be provided by MAIN, not inferred from filenames.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import numpy as np

NAME = 'p1_constructive_feedback_contrast_20261027_v1'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False)+'\n').encode()


class Reader:
    def __init__(self):
        self.bindings = {}

    def raw(self, path, expected):
        path = Path(path).absolute()
        require(isinstance(expected, str) and len(expected) == 64, 'external SHA required')
        require(path.is_file() and not path.is_symlink() and path.stat().st_size < 2*1024**2,
                'bounded regular saved artifact')
        data = path.read_bytes()
        require(hashlib.sha256(data).hexdigest() == expected, 'saved artifact SHA: '+str(path))
        require(str(path) not in self.bindings or self.bindings[str(path)] == expected, 'identity drift')
        self.bindings[str(path)] = expected
        return data

    def read(self, path, expected):
        data = self.raw(path, expected)
        if str(path).endswith('.gz'):
            with gzip.GzipFile(fileobj=io.BytesIO(data)) as f:
                data = f.read(4*1024**2+1)
            require(len(data) <= 4*1024**2, 'bounded gzip')
        def pairs(items):
            result = {}
            for key, value in items:
                require(key not in result, 'duplicate JSON key')
                result[key] = value
            return result
        result = json.loads(data, object_pairs_hook=pairs)
        canonical(result)
        return result

    def latent(self, parent, entry, artifacts):
        name = entry['file']
        require(Path(name).name == name and name.endswith('.native192.npy') and
                entry['shape'] == [192] and entry['dtype'] == 'float32' and
                artifacts[name] == entry['sha256'], 'native array descriptor')
        data = self.raw(Path(parent).parent/name, entry['sha256'])
        stream = io.BytesIO(data)
        value = np.load(stream, allow_pickle=False)
        require(stream.tell() == len(data) and value.shape == (192,) and
                value.dtype == np.dtype('float32') and np.isfinite(value).all(), 'native float32 vector')
        return value

    def finish(self):
        for path, expected in list(self.bindings.items()):
            self.raw(path, expected)


def cost(value, goal):
    value, goal = np.asarray(value, float), np.asarray(goal, float)
    require(value.shape == goal.shape == (192,) and np.isfinite(value).all() and
            np.isfinite(goal).all(), 'finite192 cost')
    return float(np.mean((value-goal)**2))


def contrast(candidates, selected_index, old_actual, new_actual):
    require(len(candidates) == 3 and [c['index'] for c in candidates] == [0, 1, 2] and
            [c['kind'] for c in candidates] ==
            ['zero_nominal', 'original_imagined_suffix', 'fresh_observed_ridge'], 'fixed candidate pool')
    values = [c['terminal_cost'] for c in candidates]
    require(np.isfinite(values).all() and min(values) >= 0 and
            selected_index == min(range(3), key=lambda i: (values[i], i)), 'model-only stable choice')
    require(np.isfinite([old_actual, new_actual]).all(), 'complete actual costs')
    predicted_delta = values[selected_index]-values[1]
    actual_delta = new_actual-old_actual
    return dict(selected_kind=candidates[selected_index]['kind'], changed_suffix=selected_index != 1,
                original_predicted_Q=values[1], selected_predicted_Q=values[selected_index],
                predicted_Q_delta=predicted_delta, original_actual_Q=old_actual,
                selected_actual_Q=new_actual, actual_Q_delta=actual_delta,
                strict_rank_reversal=bool(predicted_delta*actual_delta < 0),
                unchanged_choice_control=selected_index == 1)


def check_suffix(result, plan, report_commands):
    require(result['selected_index'] == plan['selected_index'] and report_commands == plan['commands'] and
            result['winner'] == result['candidates'][result['selected_index']], 'own chosen winner')
    require(result['winner']['commands'] == plan['suffix_commands'] == plan['commands'][5:] and
            plan['original_prefix_commands'] == plan['commands'][:5], 'winner commands bind actual full plan')


def analyze(old_rgb, old_sha, new_rgb, new_sha, suffix_report, suffix_sha,
            prefix_audit, prefix_sha):
    reader = Reader()
    old = reader.read(old_rgb, old_sha)
    new = reader.read(new_rgb, new_sha)
    suffix = reader.read(suffix_report, suffix_sha)
    prefix = reader.read(prefix_audit, prefix_sha)
    require(prefix['status'] == 'exact_saved_prefix_replay_verified' and
            prefix['distinct_report_inputs'] is True and len(prefix['rows']) == 4 and
            all(x['all_exact'] for x in prefix['rows']), 'independent actual prefix replay gate')
    for r, schema in ((old, 'p1_constructive_selected_rgb_cuda_20261027_v1.report'),
                      (new, 'p1_constructive_selected_rgb_cuda_20261027_v2.report')):
        require(r['schema'] == schema and r['status'] == 'selected_rgb_encoded_not_admitted' and
                r['errors'] == [] and r['source_verified'] is True and r['model_unchanged'] is True and
                r['model_hash_before'] == r['model_hash_after'] == old['model_hash_before'] and
                len(r['rows']) == 8, 'complete actual original model encodings')
        require(r['resources'] == {k: dict(attempted=8, completed=8) for k in ('encoder', 'projector')},
                'complete actual image resource counts')
    require(old['inputs']['physical_sha'] == prefix['old_report_sha256'] and
            new['inputs']['physical_sha'] == prefix['new_report_sha256'], 'prefix audit covers these encodings')
    require(suffix['schema'] == 'p1_constructive_observed_suffix_cuda_20261027_v1.report' and
            suffix['status'] == 'observed_suffix_plans_sealed_not_admitted' and suffix['errors'] == [] and
            suffix['source_verified'] is True and suffix['model_unchanged'] is True and
            suffix['model_hash_after'] == old['model_hash_after'] and suffix['suffix_commands_released'] is True,
            'sealed suffix model run')
    require(len(suffix['rows']) == 4, 'all four selected suffixes')
    physical = reader.read(new['inputs']['physical_report'], new['inputs']['physical_sha'])
    journal_data = reader.raw(Path(new['inputs']['physical_report']).parent/'journal.jsonl',
                              physical['journal_sha256'])
    start = json.loads(journal_data.splitlines()[0])
    require(start['kind'] == 'run_started' and len(start['prepare']['plans']) == 4,
            'actual physical selection recorded before dispatch')
    selected_plans = start['prepare']['plans']
    results = []
    for branch in range(4):
        a, b = old['rows'][2*branch:2*branch+2], new['rows'][2*branch:2*branch+2]
        sr = suffix['rows'][branch]
        dispatch = selected_plans[branch]
        require(dispatch['compiler_report_SHA'] == suffix_sha and
                Path(dispatch['compiler_report']).absolute() == Path(suffix_report).absolute() and
                dispatch['root_slot'] == sr['source_slot'] and dispatch['arm'] == sr['arm'] and
                dispatch['commands'] == sr['commands'] and
                dispatch['selection_digest'] == b[0]['provenance']['selection_digest'],
                'these actual RGBs executed this exact sealed suffix plan')
        for left, right in zip(a, b, strict=True):
            require(all(left[k] == right[k] for k in
                        ('index', 'branch', 'root_slot', 'arm', 'role', 'task_id', 'task_sha256')),
                    'same opaque task, arm and image role')
            require(left['root_slot'] == sr['source_slot'] and left['arm'] == sr['arm'], 'own suffix branch')
        require(a[0]['role'] == b[0]['role'] == 'prefix5' and a[1]['role'] == b[1]['role'] == 'terminal10',
                'no endpoint substitution')
        prefix_old = reader.latent(old_rgb, a[0]['latent'], old['artifact_sha256'])
        prefix_new = reader.latent(new_rgb, b[0]['latent'], new['artifact_sha256'])
        require(prefix_old.tobytes() == prefix_new.tobytes(), 'actual prefix neural encoding bit-exact')
        orig_z = reader.latent(old_rgb, a[1]['latent'], old['artifact_sha256'])
        new_z = reader.latent(new_rgb, b[1]['latent'], new['artifact_sha256'])
        goal = a[1]['provenance']['goal']
        require(goal == b[1]['provenance']['goal'], 'original fixed task goal')
        oq, nq = cost(orig_z, goal), cost(new_z, goal)
        require(oq == a[1]['metrics']['actual_Q'] and nq == b[1]['metrics']['actual_Q'], 'actual Q independent recomputation')
        path = Path(suffix_report).parent
        result_name = f'branch_{branch}.result.json.gz'
        result = reader.read(path/result_name, suffix['artifact_sha256'][result_name])
        plan = reader.read(path/sr['plan_file'], suffix['artifact_sha256'][sr['plan_file']])
        check_suffix(result, plan, sr['commands'])
        require(plan['record']['RGB_report_sha256'] == old_sha and
                plan['record']['prefix_latent_sha256'] == a[0]['latent']['sha256'] and
                plan['observed_context']['latent_history'][-1] == prefix_old.astype(float).tolist(),
                'model correction used this actual original prefix')
        for c in result['candidates']:
            require(cost(c['context']['latent_history'][-1], goal) == c['terminal_cost'], 'independent predicted Q')
        require(b[0]['provenance']['selection_digest'] == b[1]['provenance']['selection_digest'], 'one physical selected branch')
        row = contrast(result['candidates'], result['selected_index'], oq, nq)
        row.update(root_slot=sr['source_slot'], arm=sr['arm'], branch=branch,
                   prefix_state_PNG_exact=True, prefix_latent_exact=True)
        if row['unchanged_choice_control']:
            require(orig_z.tobytes() == new_z.tobytes(), 'unchanged original suffix actual control')
        results.append(row)
    reader.finish()
    return dict(schema=NAME+'.analysis', status='same_prefix_saved_contrast_complete', rows=results,
                changed_branches=sum(x['changed_suffix'] for x in results),
                strict_rank_reversals=sum(x['strict_rank_reversal'] for x in results),
                input_bindings=reader.bindings, new_neural_calls=0, new_physics=0,
                P1_admission=False, independent_confirmation=False,
                scope='known two-root saved feedback intervention diagnostic; not unique-cause or theory verdict')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('old-rgb', 'old-sha', 'new-rgb', 'new-sha', 'suffix-report', 'suffix-sha', 'prefix-audit', 'prefix-sha'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--output')
    a = p.parse_args()
    result = analyze(a.old_rgb, a.old_sha, a.new_rgb, a.new_sha, a.suffix_report,
                     a.suffix_sha, a.prefix_audit, a.prefix_sha)
    data = canonical(result)
    if a.output:
        path = Path(a.output)
        require(len(data) < 128*1024 and shutil.disk_usage(path.parent).free > 200*1024**2+len(data), 'small report and floor')
        with path.open('xb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
    print(json.dumps(dict(status=result['status'], changed_branches=result['changed_branches'],
                          strict_rank_reversals=result['strict_rank_reversals'])))


if __name__ == '__main__':
    main()
