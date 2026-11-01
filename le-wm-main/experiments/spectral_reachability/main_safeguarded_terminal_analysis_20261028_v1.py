"""Independent new-terminal NPY arithmetic with fixed native-effect authority."""
import argparse
import gzip
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from experiments.spectral_reachability import main_midfeedback_physical_effect_20261028_v1 as base

NAME = 'main_safeguarded_terminal_analysis_20261028_v1'
OUTPUT = base.ROOT/'results/agent_p2_midfeedback_terminal_rgb_20261028_v2_cuda_MAIN01'
REPORT_SHA = '9a63b5a3db263a3f9accdc8d54faa31b40787c362ca52e5e29ca793f785c7941'
NATIVE = base.ROOT/'results/main_safeguarded_physical_effect_20261028_v1_MAIN01.json'
NATIVE_SHA = '5bbb09ac688e4592aecbd07aaed3d4f88dd04d95c4551c5479b325cbb904de6c'


def mse(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    base.require(a.shape == b.shape == (192,) and np.isfinite(a).all() and np.isfinite(b).all(), 'finite full192')
    return float(np.mean((a-b)**2))


def compare(row, latent, strict, native):
    base.require(row['index'] == strict['index'] == native['index'] and row['status'] == 'encoded' and
        (row['root_slot'], row['condition'], row['arm']) ==
        (native['root_slot'], native['condition'], native['arm']), 'all same actual fixed plan')
    base.require(latent.dtype == np.float32 and latent.shape == (192,), 'literal native float32')
    q, pred = mse(latent,row['goal192']), mse(row['predicted192'],row['goal192'])
    old_q, old_pred = row['old_fixed_actual_Q'], row['old_fixed_actual_mid_predicted_Q']
    dq, dp = q-old_q, pred-old_pred
    expected = dict(new_suffix_actual_Q=q, new_suffix_predicted_Q=pred,
        new_minus_old_actual_Q=dq, new_minus_old_predicted_Q=dp,
        new_endpoint_MSE=mse(latent,row['predicted192']))
    base.require(all(strict[k] == v for k,v in expected.items()) and pred == row['new_suffix_Q'] and
        old_pred == native['incumbent_Q'] and pred == native['selected_Q'], 'independent new NPY metrics vs strict/compile')
    xy = native['new_minus_control']['joint_xy_error']
    angle = native['new_minus_control']['wrapped_angle_error']
    return dict(index=row['index'], **expected, jointXY_delta=xy, angle_delta=angle,
        predicted_to_actual_effect_reversal=dp*dq < 0,
        actual_to_jointXY_effect_reversal=dq*xy < 0,
        actual_to_angle_effect_reversal=dq*angle < 0)


def build(receipt, expected):
    for p,h in ((OUTPUT/'report.json',REPORT_SHA),(NATIVE,NATIVE_SHA),(Path(receipt),expected)):
        base.require(base.sha(p) == h, 'pinned actual report/native/strict receipt')
    r, native, strict = [json.loads(p.read_bytes()) for p in (OUTPUT/'report.json',NATIVE,Path(receipt))]
    base.require(strict['status'] == 'safeguarded_v2_terminal_RGB_verified_NOT_science' and
        strict['report_sha256'] == REPORT_SHA and len(r['rows']) == len(native['rows']) == len(strict['rows']) == 8,
        'full8 terminal strict dependency')
    rows = []
    for row,s,n in zip(r['rows'],strict['rows'],native['rows'],strict=True):
        e = row['latent']; name = e['file']; path = OUTPUT/name
        base.require(Path(name).name == name and not path.is_symlink() and base.sha(path) == e['sha256'] and
            path.stat().st_size < 1024*1024, 'exact bounded saved terminal file')
        b = gzip.decompress(path.read_bytes()); base.require(len(b) < 1024*1024, 'bounded expanded NPY')
        stream = BytesIO(b); z = np.load(stream,allow_pickle=False)
        base.require(stream.tell() == len(b), 'complete NPY, no trailing bytes')
        rows.append(compare(row,z,s,n))
    return dict(schema=NAME,rows=rows,fixed_plans=8,known_roots=2,fresh_roots=0,
        actual_Q_better=sum(r['new_minus_old_actual_Q'] < 0 for r in rows),
        actual_Q_worse=sum(r['new_minus_old_actual_Q'] > 0 for r in rows),
        reversals={key:[r['index'] for r in rows if r[key]] for key in
            ('predicted_to_actual_effect_reversal','actual_to_jointXY_effect_reversal','actual_to_angle_effect_reversal')},
        authority=dict(report_sha256=REPORT_SHA,native_sha256=NATIVE_SHA,strict_receipt_sha256=expected),
        independent_new_NPY_arithmetic=True,old_Q_inherited_from_full_strict=True,full_upstream_rebuilt_here=False,
        new_neural_calls=0,new_physics_calls=0,P1_admission=False,independent_confirmation=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('receipt','receipt-sha','output'): p.add_argument('--'+k,required=True)
    a=p.parse_args(); output=Path(a.output).absolute()
    base.require(output.resolve() == output and output.parent == base.ROOT/'results' and
        output.name.startswith(NAME+'_') and output.suffix == '.json','own exclusive result')
    result=build(a.receipt,a.receipt_sha)
    with output.open('xb') as f: f.write(json.dumps(result,sort_keys=True,allow_nan=False).encode()+b'\n')
    print(json.dumps(dict(output=str(output),sha256=base.sha(output),actual_Q_better=result['actual_Q_better'],
        actual_Q_worse=result['actual_Q_worse'],reversals=result['reversals'])))


if __name__ == '__main__': main()
