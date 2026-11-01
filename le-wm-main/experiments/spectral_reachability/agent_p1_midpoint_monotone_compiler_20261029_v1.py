"""P1 model-only midpoint-monotone compiler core; no production authority.

The host must supply the original verified requests, backend and durable event
sink. This module neither launches a model nor grants physical execution.
"""
from copy import deepcopy
import numpy as np
from experiments.spectral_reachability import agent_nonlinear_suffix_observed_recompile_20261028_v2 as original

NAME = 'agent_p1_midpoint_monotone_compiler_20261029_v1'
numeric = original.numeric
frozen = numeric.frozen


def contract():
    return dict(slots=4, candidates_per_slot=9, midpoint_checks_per_slot=8,
                inherited_predict=72, inherited_reverse=32, added_predict=32,
                added_reverse=0, encoder=0, physics=0, training=0,
                inherited_cost_scope='original observed recompile including one winner replay per slot',
                candidate_order=list(range(9)), midpoint_fraction=0.5,
                selection='strict Q(candidate)<Q(midpoint)<Q(incumbent); min (Q,iteration); otherwise abstain',
                task='original identity192 goal Q, identical actual-cp5 context',
                fixed_denominator=4, candidate_denominator=36, midpoint_denominator=32,
                P1_admission=False, physical_release=False, production_ready=False)


def midpoint_logits(a, b, cap):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != (10,) or b.shape != (10,) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('finite 10-dimensional logits required')
    # The original solver's incumbent may have float32 roundoff at its boundary;
    # no new projection, threshold or modified source-domain acceptance here.
    x = (a + b) * 0.5
    if not np.isfinite(x).all() or np.linalg.norm(x) > max(float(cap), np.linalg.norm(a), np.linalg.norm(b)):
        raise ValueError('midpoint not in convex endpoint ball')
    return x


def select(q0, rows):
    if len(rows) != 8 or [r['iteration'] for r in rows] != list(range(1, 9)):
        raise ValueError('all eight ordered checks required')
    if not np.isfinite(q0) or any(not np.isfinite(r['candidate_Q']) or not np.isfinite(r['midpoint_Q']) for r in rows):
        raise ValueError('nonfinite Q')
    eligible = [r for r in rows if r['candidate_Q'] < r['midpoint_Q'] < q0]
    return min(eligible, key=lambda r: (r['candidate_Q'], r['iteration']))['iteration'] if eligible else None


def compile_slot(spec, backend, journal):
    """Execute the original real compiler and eight additional real callbacks.

    The caller must provide the frozen original spec and actual backend; this
    function does not certify their source authority, resources or persistence.
    An exception leaves emitted intent/return events intact and publishes no plan.
    """
    base = original.compile_row(spec, backend, journal)
    q = spec['request']
    original.strict_numeric(q, base['numeric'], base['numeric_events'])
    its = base['numeric']['iterates']
    if len(its) != 9 or [v['iteration'] for v in its] != list(range(9)):
        raise ValueError('complete original nine iterates required')
    rows = []
    for i in range(1, 9):
        x = midpoint_logits(its[0]['logits'], its[i]['logits'], frozen.CAP)
        commands = frozen.chart(x)
        intent = dict(kind='midpoint_intent', iteration=i, context_sha256=original.digest(q['observed_context']),
                      logits=x.tolist(), commands=commands.tolist(), raw_bits=original.F.bits(commands))
        journal.event(deepcopy(intent))
        after = backend.step(deepcopy(q['observed_context']), commands, f'midpoint_{i}')
        original.core.checked_path(q['task'], q['observed_context'], commands.tolist(), [after])
        row = dict(iteration=i, midpoint_logits=x.tolist(), midpoint_commands=commands.tolist(),
                   midpoint_raw_bits=original.F.bits(commands), midpoint_context=deepcopy(after),
                   midpoint_Q=frozen.cost(q['task']['goal'], after), candidate_Q=its[i]['Q'])
        rows.append(row)
        journal.event(dict(kind='midpoint_return', row=deepcopy(row)))
    chosen = select(its[0]['Q'], rows)
    result = dict(schema=NAME+'.result', original=base, midpoint_rows=rows, selected_iteration=chosen,
                  selected_commands=None if chosen is None else deepcopy(q['incumbent']['prefix'] + its[chosen]['commands']),
                  status='model_candidate' if chosen is not None else 'abstain',
                  physical_release=False, P1_admission=False, production_ready=False)
    journal.event(dict(kind='selection', selected_iteration=chosen, status=result['status']))
    return result
