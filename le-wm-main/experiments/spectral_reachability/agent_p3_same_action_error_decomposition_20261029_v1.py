"""Saved-only numeric decomposition; inherits MAIN encoder receipt, never reruns strict."""
from __future__ import annotations

import argparse
import gzip
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import time

import numpy as np

NAME = 'agent_p3_same_action_error_decomposition_20261029_v1'
ENC = 'agent_p3_h3_witness_encoding_20261029_v1'
ENC_SHA = 'c3c7f4421ffe843acc845ba9ef1bc19424ae66e4d08fa96b7a3fc63f22bb4748'
ROOT = Path(__file__).resolve().parents[2]
CAP = 128 * 1024**2
IDENTITY = ('index', 'plan_index', 'root_slot', 'condition', 'arm', 'task_id',
            'checkpoint', 'selection_digest', 'causal_raw10_bits', 'physical_status')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def checked_bytes(path, expected, cap=CAP):
    path = Path(path)
    require(type(expected) is str and len(expected) == 64 and
            all(c in '0123456789abcdef' for c in expected), 'actual SHA256 required')
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= cap,
            'bounded regular saved input')
    data = path.read_bytes()
    require(len(data) <= cap and sha(data) == expected, 'saved input SHA256')
    return data


def decode(data):
    def pairs(items):
        out = {}
        for k, v in items:
            require(k not in out, 'duplicate JSON key')
            out[k] = v
        return out
    def bad(value):
        raise ValueError('nonfinite JSON ' + value)
    return json.loads(data, object_pairs_hook=pairs, parse_constant=bad)


def vector(value):
    require(type(value) is list and len(value) == 192 and
            all(type(v) in (int, float) for v in value), '192 numeric vector, no bool')
    a = np.asarray(value, dtype=np.float64)
    require(np.isfinite(a).all(), 'finite vector')
    require(np.array_equal(a, a.astype(np.float32).astype(np.float64)), 'original float32 vector')
    return a


def scalar(value):
    require(type(value) in (int, float) and np.isfinite(value), 'finite numeric scalar, no bool')
    return value


def equal_scalar(value, expected):
    if expected is None:
        require(value is None, 'unknown scalar stays null')
    else:
        require(scalar(value) == expected, 'independent scalar recomputation')


def q(x, goal):
    return float(np.mean((x - goal)**2))


def decompose_point(predicted192, actual192, goal192):
    p, g = vector(predicted192), vector(goal192)
    fields = dict(predicted192=p.tolist(), actual192=None, goal192=g.tolist(),
                  model_Q=q(p, g), actual_encoded_Q=None, error_mse=None,
                  actual_residual_cross_term=None, model_minus_actual_Q=None,
                  absolute_Q_error=None, prediction_residual_form_Q_error=None,
                  identity_residual=None, prediction_form_identity_residual=None,
                  identity_roundoff_bound=None, alignment_cosine=None)
    if actual192 is None:
        return fields
    a = vector(actual192)
    e, r, s = p-a, a-g, p-g
    energy = float(np.mean(e*e))
    cross = float(2*np.mean(r*e))
    actual = q(a, g)
    delta = fields['model_Q']-actual
    alternate = float(2*np.mean(s*e))-energy
    # Conservative fixed f64 bound for differences/products and 192-term reductions.
    # An arithmetic diagnostic only, never a relaxation of encoder native gates.
    u = np.finfo(np.float64).eps/2
    gamma = 4096*u/(1-4096*u)
    scale = float(np.mean(s*s+r*r+e*e+2*np.abs(r*e)+2*np.abs(s*e)))
    bound = gamma*scale + 4096*np.nextafter(0., 1.)
    residual, residual2 = delta-(energy+cross), delta-alternate
    require(abs(residual) <= bound and abs(residual2) <= bound, 'f64 decomposition identity bound')
    denom = float(np.linalg.norm(r)*np.linalg.norm(e))
    fields.update(actual192=a.tolist(), actual_encoded_Q=actual, error_mse=energy,
                  actual_residual_cross_term=cross, model_minus_actual_Q=delta,
                  absolute_Q_error=abs(delta), prediction_residual_form_Q_error=alternate,
                  identity_residual=residual, prediction_form_identity_residual=residual2,
                  identity_roundoff_bound=bound,
                  alignment_cosine=None if denom == 0 else float(np.dot(r, e)/denom))
    return fields


def load_latent(folder, descriptor, inventory):
    require(type(descriptor) is dict and set(descriptor) ==
            {'file', 'sha256', 'array_sha256', 'shape', 'dtype'}, 'native latent descriptor ABI')
    name = descriptor['file']
    require(type(name) is str and Path(name).name == name and name not in ('.', '..')
            and descriptor['shape'] == [192] and descriptor['dtype'] == 'float32', 'flat latent192')
    require(inventory[name] == descriptor['sha256'], 'report inventory joins latent')
    data = checked_bytes(Path(folder)/name, descriptor['sha256'], 2*1024**2)
    with gzip.GzipFile(fileobj=BytesIO(data)) as f:
        expanded = f.read(2*1024**2+1)
    require(len(expanded) <= 2*1024**2, 'bounded expanded latent')
    stream = BytesIO(expanded)
    a = np.load(stream, allow_pickle=False)
    require(stream.tell() == len(expanded) and a.shape == (192,) and a.dtype == np.float32
            and np.isfinite(a).all() and sha(a.tobytes()) == descriptor['array_sha256'],
            'saved actual latent array shape/bytes')
    return a.tolist()


def analyze_documents(p, r, receipt, load):
    """Independent arithmetic + joins, not an encoder event/IO replay consumer."""
    require(p['schema'] == ENC+'.prepare' and r['schema'] == ENC+'.report'
            and receipt['schema'] == ENC+'.consume', 'exact encoder schemas')
    require(r['status'] == 'terminal_encoded_science_blocked' and receipt['status'] ==
            'all16_H3_witness_RGB_verified_NOT_admitted', 'completed inherited encoder authority')
    require(r['errors'] == r['blocked'] == [] and r['model_unchanged'] is True
            and r['source_verified'] is True and r['model_hash_before'] == r['model_hash_after']
            == p['expected_model_hash'], 'inherited frozen model receipt')
    for obj in (r, receipt):
        require(all(obj[k] is False for k in ('P1_admission', 'P3_admission', 'C3', 'physical_release')),
                'scientific gates stay false')
    require(p['contract'] == r['contract'] and p['contract']['fixed_denominator'] == 16
            and p['contract']['plan_denominator'] == 4 and p['contract']['checkpoints'] == [5,10,15,20]
            and receipt['fixed_denominator'] == 16 and receipt['plan_denominator'] == 4
            and receipt['known_roots'] == 2 and receipt['fresh_roots'] == 0,
            'fixed full denominators')
    require(len(p['rows']) == len(r['rows']) == len(receipt['rows']) == 16 and len(p['plans']) == 4,
            'all16 rows and four plans')
    require(r['counts'] == receipt['counts'], 'inherited resource join')
    for key in ('physical_report_sha256', 'physical_receipt_sha256'):
        require(p[key] == receipt[key], 'inherited physical authority join')
    points, anchors = [], []
    for i, (spec, row, saved) in enumerate(zip(p['rows'], r['rows'], receipt['rows'], strict=True)):
        b, j = divmod(i, 4)
        root, bank = b//2, ('endpoints', 'midpoints')[b % 2]
        require(type(spec['index']) is int and type(spec['plan_index']) is int
                and type(spec['checkpoint']) is int and type(spec['root_slot']) is int
                and (spec['index'],spec['plan_index'],spec['root_slot'],spec['condition'],spec['arm'],spec['checkpoint'])
                == (i,b,root,'off',bank,(5,10,15,20)[j]), 'fixed ordered branch/checkpoint identities')
        require(canonical({k:v for k,v in row.items() if k not in
                ('status','latent','pixels','preprocessed','modules')}) == canonical({k:v for k,v in spec.items()
                if k not in ('status','latent')}), 'report preserves original prepare provenance')
        require(all(canonical(saved[k]) == canonical(spec[k]) for k in IDENTITY), 'receipt identity join')
        plan = p['plans'][b]
        witness = plan['source_witness']
        require(spec['selection_digest'] == plan['selection_digest'] and spec['task_id'] == plan['task_id']
                and spec['model_hash'] == p['expected_model_hash']
                and spec['witness_sha256'] == witness['witness_sha256']
                and spec['task_binding'] == witness['task']['binding'], 'original selected plan binding')
        commands = np.asarray(plan['commands'], np.float32)
        require(commands.shape == (20,2) and np.isfinite(commands).all()
                and canonical(spec['full20']) == canonical(plan['commands'])
                and spec['raw_float32_bits'] == commands.tobytes().hex(), 'full20 exact rawbits')
        suffix = np.asarray(witness['winner']['actions'], np.float32)
        original = np.asarray(witness['request']['raw10'], np.float32)
        require(suffix.shape == (15,2) and suffix.tobytes() == commands[5:].tobytes()
                and original.shape == (10,2) and np.isfinite(original).all()
                and original[-5:].tobytes() == commands[:5].tobytes(), 'fixed prefix/H3 winner raw actions')
        history = np.asarray(spec['causal_raw10'], np.float32)
        expected_history = np.concatenate((original,commands[5:spec['checkpoint']]))[-10:]
        require(history.shape == (10,2) and np.isfinite(history).all()
                and history.tobytes().hex() == spec['causal_raw10_bits']
                and history.tobytes() == expected_history.tobytes(), 'causal raw10 bits')
        g = vector(spec['goal192'])
        require(np.array_equal(g, vector(witness['task']['goal'])) and
                canonical(saved['goal192']) == canonical(spec['goal192']), 'original goal192')
        a = None if row['latent'] is None else load(row['latent'])
        require((spec['source'] is None) is (a is None) and saved['latent_missing'] is (a is None)
                and canonical(saved['actual192']) == canonical(a), 'actual array/receipt/missing join')
        require(canonical(saved['source']) == canonical(spec['source']) and
                canonical(saved['native_metrics']) == canonical(spec['native_metrics']) and
                canonical(saved['response']) == canonical(spec['response']), 'same-time native/source join')
        identity = {k: spec[k] for k in IDENTITY}
        identity.update(witness_sha256=spec['witness_sha256'], model_hash=spec['model_hash'],
                        available=a is not None, reason=None if a is not None else 'missing_physical_checkpoint',
                        native_metrics=spec['native_metrics'], response=spec['response'])
        if j == 0:
            require(spec['predicted192'] is saved['predicted192'] is None and saved['role'] ==
                    'observed_anchor_reencode', 'cp5 is not a forecast')
            anchor = vector(spec['observed_anchor192'])
            if a is not None:
                require(np.array_equal(anchor, vector(a)), 'exact observed cp5 anchor')
            for key in ('model_Q','model_minus_actual_Q','absolute_Q_error','prediction_vs_encoded_MSE'):
                equal_scalar(saved[key], None)
            equal_scalar(saved['actual_encoded_Q'], None if a is None else q(vector(a), g))
            equal_scalar(saved['observed_anchor_reencode_MSE'], None if a is None else 0.)
            anchors.append(identity | dict(actual192=a, observed_anchor192=anchor.tolist(),
                goal192=g.tolist(), observed_anchor_reencode_MSE=None if a is None else 0.))
        else:
            pred = vector(spec['predicted192'])
            require(np.array_equal(pred, vector(witness['winner']['whole'][j-1]['latent_history'][-1]))
                    and canonical(saved['predicted192']) == canonical(spec['predicted192'])
                    and saved['role'] == 'same_action_H3_prediction', 'original H3 forecast endpoint')
            values = decompose_point(spec['predicted192'], a, spec['goal192'])
            equal_scalar(spec['model_Q'], values['model_Q'])
            for key in ('model_Q','actual_encoded_Q','model_minus_actual_Q','absolute_Q_error'):
                equal_scalar(saved[key], values[key])
            equal_scalar(saved['prediction_vs_encoded_MSE'], values['error_mse'])
            equal_scalar(saved['observed_anchor_reencode_MSE'], None)
            points.append(identity | values)
    pairs = []
    for b in range(4):
        rows = points[3*b:3*b+3]
        for x,y in ((0,1),(1,2),(0,2)):
            a,c = rows[x], rows[y]
            available = a['available'] and c['available']
            keys = ('predicted_Q_change','actual_Q_change','change_gap',
                    'error_mse_change','cross_term_change','identity_residual')
            v = dict.fromkeys(keys)
            if available:
                dp = c['model_Q']-a['model_Q']; da = c['actual_encoded_Q']-a['actual_encoded_Q']
                de = c['error_mse']-a['error_mse']
                dc = c['actual_residual_cross_term']-a['actual_residual_cross_term']
                v.update(predicted_Q_change=dp,actual_Q_change=da,change_gap=dp-da,
                         error_mse_change=de,cross_term_change=dc,identity_residual=(dp-da)-(de+dc))
            pairs.append(dict(plan_index=b,root_slot=a['root_slot'],arm=a['arm'],condition='off',
                from_checkpoint=a['checkpoint'],to_checkpoint=c['checkpoint'],available=available,
                reason=None if available else 'missing_endpoint',**v))
    return dict(anchors=anchors,points=points,temporal_pairs=pairs,
                coverage={k:dict(fixed=len(v),available=sum(x['available'] for x in v),
                                missing=sum(not x['available'] for x in v))
                          for k,v in (('anchors',anchors),('forecasts',points),('temporal_pairs',pairs))})


def build_saved_decomposition(*, prepared_path, prepare_sha256, output_dir, report_sha256,
                              receipt_path, receipt_sha256):
    started = time.monotonic()
    report_path = Path(output_dir).resolve()/'report.json'
    source = ROOT/'experiments/spectral_reachability'/(ENC+'.py')
    checked_bytes(source, ENC_SHA)
    inputs = ((prepared_path,prepare_sha256),(report_path,report_sha256),(receipt_path,receipt_sha256))
    p,r,receipt = [decode(checked_bytes(path,h)) for path,h in inputs]
    require(Path(r['prepared']).resolve() == Path(prepared_path).resolve() and
            r['prepare_sha256'] == prepare_sha256 and receipt['report_sha256'] == report_sha256,
            'explicit prepare/report/receipt SHA chain')
    result = analyze_documents(p,r,receipt,lambda d:load_latent(output_dir,d,r['artifact_sha256']))
    for path,h in inputs:
        checked_bytes(path,h)
    checked_bytes(source,ENC_SHA)
    return dict(schema=NAME+'.saved',**result,
        authority=dict(prepared_path=str(Path(prepared_path).resolve()),prepare_sha256=prepare_sha256,
            report_path=str(report_path),report_sha256=report_sha256,
            receipt_path=str(Path(receipt_path).resolve()),receipt_sha256=receipt_sha256,
            encoder_source_sha256=ENC_SHA,physical_report_sha256=p['physical_report_sha256'],
            physical_receipt_sha256=p['physical_receipt_sha256']),
        evidence_scope=dict(inherited_MAIN_full_encoder_receipt=True,full_upstream_rebuilt_here=False,
            native_IO_or_neural_parity_revalidated_here=False,
            independently_recomputed='latent arrays, identities, action bits, goals, Q and error arithmetic',
            not_replayed='encoder events/preprocessing/module IO, physics, all upstream bindings'),
        resources=dict(new_encode=0,new_projector=0,new_predict=0,new_AD=0,new_physics=0,
            saved_CPU_wall_seconds=time.monotonic()-started,inherited_counts=receipt['counts'],total_FLOPs=None),
        known_roots=2,fresh_roots=0,equal_total_compute=False,P1_admission=False,P3_admission=False,
        C3=False,physical_release=False,physical_benefit_claim=False,
        own_source_sha256=sha(Path(__file__).read_bytes()))


def write(path,value):
    path=Path(path).absolute(); data=canonical(value)+b'\n'
    require(path.name.startswith(NAME+'_') and len(data)<=CAP and not path.is_symlink(),
            'exclusive own bounded saved output')
    with path.open('xb') as f:
        require(f.write(data)==len(data),'full saved output')
        f.flush(); os.fsync(f.fileno())
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)
    return dict(path=str(path),sha256=sha(data))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['analyze'])
    for field in ('prepared-path','prepare-sha256','output-dir','report-sha256','receipt-path','receipt-sha256','saved-output'):
        parser.add_argument('--'+field,required=True)
    args=vars(parser.parse_args()); args.pop('command'); destination=args.pop('saved_output')
    print(canonical(write(destination,build_saved_decomposition(**args))).decode())


if __name__=='__main__':
    main()
