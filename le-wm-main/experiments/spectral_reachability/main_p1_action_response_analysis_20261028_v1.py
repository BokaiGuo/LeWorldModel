"""Independent exact scalar arithmetic on the strictly consumed fixed80 panel.

No deployment correction is fitted or selected. Renderer changes affect images,
not physical endpoints; all counts are descriptive on two known roots.
"""
import argparse
from fractions import Fraction
import itertools
import json
from pathlib import Path

from experiments.spectral_reachability import main_midfeedback_physical_effect_20261028_v1 as base

NAME = 'main_p1_action_response_analysis_20261028_v1'
REPORT = base.ROOT/'results/agent_p1_prospective_action_effect_20261028_v2_MAIN01/report.json'
REPORT_SHA = '2e7794d32602ca2b045202099fedb6b99ac5988d6255ee68870032219530084c'
RECEIPT = base.ROOT/'results/agent_p1_prospective_action_effect_20261028_v4_MAIN01.json'
RECEIPT_SHA = '5a5c1948dd37ff0725447887218f6d57e59aaae54a3f1df28925fb8fe32b01a8'


def vec(values):
    base.require(len(values) == 192, 'full192, no projection')
    return tuple(Fraction.from_float(float(x)) for x in values)


def inner(a, b):
    return sum((x*y for x, y in zip(a, b, strict=True)), Fraction())/192


def minus(a, b):
    return tuple(x-y for x, y in zip(a, b, strict=True))


def unpack(value):
    return Fraction(int(value['numerator']), int(value['denominator']))


def analyze_anchor(anchor, rows, strict):
    base.require(len(rows) == 5 and all(r['status'] == 'terminal25_encoded' for r in rows), 'all five actual25')
    p = list(map(vec, anchor['seal']['predictions']))
    a = [vec(r['endpoint']) for r in rows]
    b = [vec(r['alternate_endpoint']) for r in rows]
    g = vec(anchor['seal']['goal'])
    qp, qa, qb = [[inner(minus(x, g), minus(x, g)) for x in xs] for xs in (p, a, b)]
    e0 = minus(a[0], p[0])
    h = [minus(minus(a[w], a[0]), minus(p[w], p[0])) for w in range(5)]
    native = [base.metrics(r['end_state'], r['goal']) for r in rows]
    pairs = []
    for (i, j), saved in zip(itertools.combinations(range(5), 2), strict['effects']['pairs'], strict=True):
        mp, ma = qp[i]-qp[j], qa[i]-qa[j]
        common = 2*inner(minus(p[i], p[j]), e0)
        # Independently form the quadratic expansion of action-conditioned errors.
        hi, hj = h[i], h[j]
        response = (2*inner(minus(p[i], g), hi)-2*inner(minus(p[j], g), hj)
            + 2*inner(e0, minus(hi, hj))+inner(hi, hi)-inner(hj, hj))
        base.require(saved['i'] == i and saved['j'] == j and saved['available'] and
            unpack(saved['predicted_margin']) == mp and unpack(saved['observed_margin']) == ma and
            unpack(saved['shared_zero_projection']) == common and unpack(saved['response_remainder']) == response and
            ma == mp+common+response and unpack(saved['identity_residual']) == 0, 'independent exact margin expansion vs strict')
        reverse = mp*ma < 0
        base.require(saved['strict_reversal'] == reverse, 'exact strict sign')
        pairs.append(dict(i=i, j=j, model_to_actual_reversal=reverse,
            common_deletion_removes_strict_reversal=bool(reverse and mp*(ma-common) >= 0),
            response_deletion_removes_strict_reversal=bool(reverse and mp*(ma-response) >= 0),
            contributions_opposite_sign=common*response < 0,
            actual_to_jointXY_reversal=ma*Fraction.from_float(native[i]['joint_xy_error']-native[j]['joint_xy_error']) < 0,
            actual_to_angle_reversal=ma*Fraction.from_float(native[i]['wrapped_angle_error']-native[j]['wrapped_angle_error']) < 0,
            fixed_goal_renderer_ranking_reversal=ma*(qb[i]-qb[j]) < 0))
    response_errors = [inner(x, x) for x in h]
    renderer_errors = [inner(minus(minus(b[w], b[0]), minus(a[w], a[0])),
                             minus(minus(b[w], b[0]), minus(a[w], a[0]))) for w in range(5)]
    for w in range(5):
        base.require(response_errors[w] == unpack(strict['effects']['words'][w]['response_error_mse']) and
            renderer_errors[w] == unpack(strict['renderer']['words'][w]['effect_change_mse']), 'independent response/renderer exact arithmetic')
    chosen = anchor['seal']['selected_word']
    base.require(chosen == strict['selected_word'] == min(range(5), key=lambda w:(qp[w], w)), 'sealed model argmin, no outcome selection')
    return dict(anchor_index=anchor['anchor_index'], selected_word=chosen, pairs=pairs,
        response_error_mse=list(map(float, response_errors)), renderer_action_effect_change_mse=list(map(float, renderer_errors)),
        selected_native=native[chosen], all_native=native,
        exact_common_zero_mse=float(inner(e0, e0)))


def build():
    for path, expected in ((REPORT, REPORT_SHA), (RECEIPT, RECEIPT_SHA)):
        base.require(base.sha(path) == expected, 'pinned actual report and full strict receipt')
    report, receipt = [json.loads(p.read_bytes()) for p in (REPORT, RECEIPT)]
    base.require(receipt['status'] == 'strict_phase_saved_adapter_audited_not_P1_admission' and
        receipt['report_sha256'] == REPORT_SHA and receipt['preprocessing']['checked'] == 88 and
        receipt['phase_validation']['raw_returns'] == 1000 and len(report['rows']) == 40 and
        len(report['anchors']) == 8, 'actual complete phase/RGB/model closure dependency')
    strict = receipt['v2_adapted_consume']
    base.require(strict['available_pairs'] == strict['fixed_pairs'] == 80 and strict['unknown_pairs'] == 0, 'fixed80 complete, not old76 replacement')
    rows = [analyze_anchor(a, report['rows'][i*5:i*5+5], strict['analyses'][i]) for i, a in enumerate(report['anchors'])]
    pairs = [p for r in rows for p in r['pairs']]
    keys = tuple(pairs[0])[2:]
    return dict(schema=NAME, anchors=rows, pair_counts={k:sum(p[k] for p in pairs) for k in keys},
        fixed_pairs=80, known_roots=2, fresh_roots=0,
        nonzero_action_response_errors=sum(v > 0 for r in rows for v in r['response_error_mse'][1:]),
        nonzero_renderer_action_effect_changes=sum(v > 0 for r in rows for v in r['renderer_action_effect_change_mse'][1:]),
        selected_success=sum(r['selected_native']['native_success'] for r in rows),
        authority=dict(report_sha256=REPORT_SHA, strict_receipt_sha256=RECEIPT_SHA, independent_exact_arithmetic=True,
            full_upstream_rebuilt_here=False),
        new_neural_calls=0, new_physics_calls=0, P1_admission=False, independent_confirmation=False,
        interpretation='Exact descriptive known-root diagnostics; algebraic term deletion is not a deployed correction or unique cause. Fixed-goal renderer Q is not renderer-matched goal evaluation. Old80/76 panel remains separate.')


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', required=True); a = p.parse_args()
    output = Path(a.output).absolute()
    base.require(output.resolve() == output and output.parent == base.ROOT/'results' and
                 output.name.startswith(NAME+'_') and output.suffix == '.json', 'own exclusive result')
    result = build()
    with output.open('xb') as f:
        f.write(json.dumps(result, sort_keys=True, allow_nan=False).encode()+b'\n')
    print(json.dumps(dict(output=str(output), sha256=base.sha(output), pair_counts=result['pair_counts'],
        response_errors_nonzero=result['nonzero_action_response_errors'],
        renderer_effect_changes_nonzero=result['nonzero_renderer_action_effect_changes'], selected_success=result['selected_success'])))


if __name__ == '__main__': main()
