"""Saved single-group native replay and scalar audit; not a production host."""
from copy import deepcopy
import math
import numpy as np
from experiments.spectral_reachability import agent_p1_fixed_prefix_refinement_20261029_v1 as method

F, core = method.F, method.core
require, raw = F.require, F.raw
CONTROL = {'prefix_intent', 'prefix_return', 'score_intent', 'score_return',
           'gradient_intent', 'gradient_return', 'winner_replay_intent',
           'winner_replay_return', 'group_return'}


def scalar_audit(task, result):
    goal = np.asarray(task['goal'], dtype=np.float32)
    require(goal.shape == (192,) and np.isfinite(goal).all(), 'finite 192 goal')
    gradients = result['gradients']
    require(type(gradients) is list and len(gradients) == 8, 'eight scalar losses')
    largest = 0.0
    for i, entry in enumerate(gradients, 1):
        z = np.asarray(entry['context']['latent_history'][-1], dtype=np.float32)
        loss = entry['native_float32_loss']
        require(z.shape == goal.shape and np.isfinite(z).all() and type(loss) in (int, float)
                and math.isfinite(loss) and loss >= 0, 'finite endpoint and loss')
        with np.errstate(over='ignore', invalid='ignore'):
            delta = np.subtract(z, goal, dtype=np.float32)
            sq = np.multiply(delta, delta, dtype=np.float32)
        require(np.isfinite(sq).all(), 'objective overflow')
        expected = float(np.mean(sq, dtype=np.float64))
        error = abs(loss - expected)
        require(error <= 3e-5 * expected + float(np.finfo(np.float32).tiny),
                f'suffix iteration {i}: scalar objective mismatch')
        largest = max(largest, error)
    return dict(checked_scalar_losses=8, max_absolute_error=largest,
                scope='saved endpoint/goal identity-Q MSE only', P1_admission=False)


def schedule(q, result):
    rows = result['iterates']
    require(type(rows) is list and len(rows) == 9 and
            [r['iteration'] for r in rows] == list(range(9)), 'nine ordered iterates')
    mid = q['expected_mid']
    plan = [('prefix_replay', q['task']['root_context'], F.chart(q['prefix_logits']).tolist()),
            ('suffix_0', mid, rows[0]['commands'])]
    for i in range(1, 9):
        plan.extend([(f'gradient_{i}', mid, rows[i-1]['commands']),
                     (f'suffix_{i}', mid, rows[i]['commands'])])
    selected = min(rows, key=lambda row: (row['terminal_Q'], row['iteration']))
    plan.append(('winner_replay', mid, selected['commands']))
    return plan


def audit(q, result, receipt, events):
    plan = schedule(q, result)
    native = [e for e in events if e['kind'] not in CONTROL]
    starts, ends, linked = F.audit_native(q['task'], receipt, native, plan)
    require(len(starts) == len(ends) == 19 and receipt['caps'] == F.caps(19, 8), 'complete native ledger')
    grad = {}
    for i, (label, _, a) in enumerate(plan):
        if not label.startswith('gradient_'):
            continue
        phase = f'f{i:03}'
        op = next(x for x in receipt['events'] if x['resource'] == 'grad' and x['scope']['forward'] == phase)
        cbop = next(x for x in receipt['events'] if x['resource'] == 'raw_backward' and x['scope']['forward'] == phase)
        ai, ao = linked[op['index']]
        cb = linked[cbop['index']][0]
        core.keys(ai['arrays'], ('grad:loss', 'grad:raw'))
        core.keys(ao['arrays'], ('grad:output',))
        core.keys(cb['arrays'], ('value',))
        loss, leaf, g, b = (F.signal.unblob(x) for x in
                            (ai['arrays']['grad:loss'], ai['arrays']['grad:raw'], ao['arrays']['grad:output'], cb['arrays']['value']))
        require(loss.shape == (1,) and leaf.shape == g.shape == b.shape == (5, 2)
                and leaf.dtype == g.dtype == b.dtype == loss.dtype == np.dtype('float32')
                and leaf.tobytes() == np.asarray(a, np.float32).tobytes() and g.tobytes() == b.tobytes(), 'exact AD buffers')
        grad[i] = dict(context=deepcopy(ends[i]['context']), native_float32_loss=float(loss[0]), raw_gradient=g.tolist())

    class Replay:
        def __init__(self): self.i = 0
        def pull(self, c, a, label, is_gradient):
            i = self.i
            self.i += 1
            require(i < len(plan) and plan[i][0] == label and raw(c) == raw(plan[i][1])
                    and F.bits(a) == F.bits(plan[i][2]), 'saved callback identity')
            return deepcopy(grad[i] if is_gradient else ends[i]['context'])
        def step(self, c, a, label): return self.pull(c, a, label, False)
        def gradient(self, c, a, label): return self.pull(c, a, label, True)

    replay, controls = Replay(), []
    rebuilt = method.run_group(q, replay, controls.append)
    require(replay.i == 19 and raw(rebuilt) == raw(result) and
            raw(controls) == raw([e for e in events if e['kind'] in CONTROL]), 'complete saved method replay')
    positions = {id(e): i for i, e in enumerate(events)}
    for e in events:
        if e['kind'] not in ('prefix_intent', 'score_intent', 'gradient_intent', 'winner_replay_intent'):
            continue
        returns = [x for x in events if x['kind'] == e['kind'].replace('_intent', '_return')
                   and positions[id(x)] > positions[id(e)]]
        require(returns and sum(positions[id(e)] < positions[id(s)] < positions[id(returns[0])] for s in starts) == 1,
                'one native forward within each control pair')
    return dict(index=q['index'], native_forwards=19, scalar_reverse=8,
                scalar_objective=scalar_audit(q['task'], result), P1_admission=False)
