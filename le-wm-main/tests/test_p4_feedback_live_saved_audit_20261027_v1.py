"""In-memory unittest only: no pytest fixture/cache/tmp, CUDA or physics."""
from copy import deepcopy
import hashlib
import math
import sys
import unittest
from unittest.mock import patch
import numpy as np
from experiments.spectral_reachability import p4_feedback_live_saved_audit_20261027_v1 as m


def fixture():
    c = dict(layout='states_with_between_state_actions_v1', context_length=3, latent_dim=192, action_dim=10,
        latent_history=np.zeros((3, 192), np.float32).tolist(), action_history=np.zeros((2, 10), np.float32).tolist())
    goal = np.full(192, .1); q = np.eye(192); j = np.zeros((192, 10), np.float32); j[:10] = np.eye(10)
    radius = math.sqrt(5)*.75; radii = (math.sqrt(5)*.25, radius)
    params = [np.zeros(10), np.full(10, .1/1.002)]; kinds = ['zero', 'root_dls_progress']
    for r in radii:
        for mode in range(2):
            for sign in (-1, 1):
                params.append(sign*r*np.eye(10)[mode]); kinds.append(f'spectral_mode_{mode}_radius_{r:.8f}_sign_{sign}')
    norm = dict(mean=[0., 0.], std=[1., 1.], source='metadata retained')
    def transition(commands):
        result = deepcopy(c); native = np.zeros(192, np.float32); native[:10] = commands.reshape(10)
        result['latent_history'][-1] = native.tolist(); result['action_history'][-1] = commands.reshape(10).tolist()
        return result
    nominal = transition(np.zeros((5, 2), np.float32)); candidates = []
    for i, logits in enumerate(params):
        commands = np.tanh(logits.astype(np.float32)).reshape(5, 2); context = transition(commands)
        d = np.asarray(context['latent_history'])[-1]-goal
        candidates.append(dict(index=i, kind=kinds[i], logits=logits.tolist(), commands=commands.tolist(), context=context, cost=float(np.mean(d*d))))
    winner = min(candidates, key=lambda row:(row['cost'], row['index']))
    s = dict(request=dict(context=deepcopy(c), context_sha256=m.algebra.content_sha(c)), fresh_context_specific_AD=True,
        nominal=nominal, jacobian_forward=deepcopy(nominal), bank=dict(params=np.stack(params).tolist(), kinds=kinds,
        singular_values=np.ones(10).tolist(), right_vectors=np.eye(10).tolist()), candidates=candidates,
        winner=deepcopy(winner), replay=deepcopy(winner['context']))
    return dict(goal=goal.tolist(), q=q.tolist(), normalization=norm), c, s, j


class AuditTests(unittest.TestCase):
    def test_no_neural_imports(self):
        self.assertFalse(any(name in sys.modules for name in ('torch', 'gymnasium', 'pymunk')))

    def test_exact_inventory(self):
        names = m.inventory(); self.assertEqual(len(names), 198)
        self.assertEqual(sum(name.endswith('.png') for name in names), 72)
        self.assertEqual(sum(name.endswith('.selection.json') for name in names), 8)

    def test_Q_metrics_independently(self):
        r = m.cost_metrics(np.full(192, 2.), np.ones(192), np.zeros(192), np.eye(192))
        self.assertEqual(r, dict(latent_MSE=1., Q_prediction_MSE=1., Q_goal_MSE=4.))
        q = np.zeros((192, 192)); q[0, 0] = 1
        r = m.cost_metrics(np.full(192, 2.), np.ones(192), np.zeros(192), q)
        self.assertEqual(r['Q_goal_MSE'], 4/192); self.assertEqual(r['Q_prediction_MSE'], 1/192)

    def test_nonfinite_metrics_refused(self):
        with self.assertRaises(ValueError): m.cost_metrics(np.full(192, np.nan), np.zeros(192), np.zeros(192), np.eye(192))

    def test_multiturn_wrapped_not_native_false_success(self):
        for delta in (0., .2, math.pi, 2*math.pi+.5, -4*math.pi-.5, 6*math.pi+.1):
            with self.subTest(delta=delta):
                state = np.zeros(7); state[4] = delta; r = m.physical_metrics(state, np.zeros(7))
                wrapped = abs((delta+math.pi)%(2*math.pi)-math.pi)
                self.assertAlmostEqual(r['wrapped_diagnostic']['wrapped_angle_error'], wrapped)
                self.assertEqual(r['wrapped_diagnostic']['wrapped_official_success'], wrapped < math.pi/9)
                if abs(delta) > 2*math.pi and wrapped > math.pi/9:
                    self.assertTrue(r['official']['official_success']); self.assertFalse(r['wrapped_diagnostic']['wrapped_official_success'])

    def test_position_success_thresholds(self):
        state = np.zeros(7); state[0] = 21
        r = m.physical_metrics(state, np.zeros(7)); self.assertFalse(r['official']['official_success']); self.assertTrue(r['official']['object_only_success'])
        state[2] = 21; self.assertFalse(m.physical_metrics(state, np.zeros(7))['official']['object_only_success'])

    def test_synthetic_ten_candidates_and_stable_selection(self):
        p, c, s, j = fixture(); winner = m.selection_check(p, c, s, j)
        self.assertEqual(winner['index'], 1)

    def test_saved_candidate_tampering_fails_closed(self):
        for fault in ('cost', 'winner', 'replay', 'action', 'history', 'context_sha', 'bank', 'metadata'):
            with self.subTest(fault=fault):
                p, c, s, j = fixture()
                if fault == 'cost': s['candidates'][0]['cost'] += .01
                elif fault == 'winner': s['winner'] = deepcopy(s['candidates'][0])
                elif fault == 'replay': s['replay']['latent_history'][-1][0] += .01
                elif fault == 'action': s['candidates'][0]['commands'][0][0] += .01
                elif fault == 'history': s['candidates'][0]['context']['action_history'][-1][0] += .01
                elif fault == 'context_sha': s['request']['context_sha256'] = 'wrong'
                elif fault == 'bank': s['bank']['params'][1][0] += .01
                else: s['candidates'][0]['context']['layout'] = 'outgoing-old-layout'
                with self.assertRaises(ValueError): m.selection_check(p, c, s, j)

    def test_actual_own_observed_and_imagined_persistent(self):
        _, c, _, _ = fixture(); observed = np.full((3, 192), 2., np.float32)
        self.assertEqual(m.planning_context('actual', c, observed)['latent_history'], observed.tolist())
        self.assertEqual(m.planning_context('imagined', c, observed), c)
        self.assertEqual(m.planning_context('actual', c, observed)['action_history'], c['action_history'])
        with self.assertRaises(ValueError): m.planning_context('other-arm', c, observed)

    def test_exact_journal_grammar_budget_and_selection_order(self):
        hashes = {name:hashlib.sha256(name.encode()).hexdigest() for name in m.inventory()}
        events, entries, rounds = m.expected_journal(hashes)
        self.assertEqual(len(events), 5291); self.assertEqual(len(entries), 2486); self.assertEqual(len(rounds), 6)
        for kind, count in m.COUNTS.items(): self.assertEqual(sum(e['kind'] == kind for e in entries), count)
        for arm in m.ARMS:
            for boundary in m.BOUNDARIES:
                seal = next(i for i, e in enumerate(events) if e['event'] == 'selection_sealed' and e['arm'] == arm and e['boundary'] == boundary)
                dispatch = next(i for i, e in enumerate(events) if e['event'] == 'dispatch_authorized' and e['arm'] == arm and e['boundary'] == boundary)
                self.assertLess(seal, dispatch)
        self.assertEqual({e['kind'] for e in entries if e['boundary'] == 20}, {'encoder', 'projector'})
        corrupt = deepcopy(events); corrupt[0], corrupt[1] = corrupt[1], corrupt[0]
        with self.assertRaises(ValueError): m.same(corrupt, events, 'order tamper')

    def test_guard_recount_in_memory_no_simulator(self):
        image = np.zeros((224, 224, 3), np.uint8); image[5:10, 5:10] = [65, 105, 225]
        stem = 'actual.from5'; names = {stem+f'.step{i}.png':'a'*64 for i in range(1, 6)}
        receipt = dict(status='completed', error=None, attempted_commands=5, completed_commands=5, planned_commands=5,
            remaining_commands_not_executed=0, terminal_pixel_sha256=m.rgb_sha(image), guard=dict(root_id='ROOT', abstained=False,
                P1_admission=False, primitive_commands_authorized=5, observations=[dict(primitive_step=i, canonical_agent_pixels=25) for i in range(6)]))
        data = {stem+'.physics.json':receipt}
        for i in range(1, 6):
            name = stem+f'.step{i}.png'; data[stem+f'.step{i}.json'] = dict(command=[0., 0.], terminated=False, truncated=False,
                rgb=dict(file=name, file_sha256=names[name], pixel_sha256=m.rgb_sha(image)))
        class Files:
            hashes = names
            def json(self, name): return data[name]
            def rgb(self, name): return image
        m.packet_check(Files(), stem, np.zeros((5, 2)), 'ROOT', image)
        receipt['guard']['observations'][3]['canonical_agent_pixels'] = 24
        with self.assertRaises(ValueError): m.packet_check(Files(), stem, np.zeros((5, 2)), 'ROOT', image)

    def test_external_receipt_SHA_checked_before_sources(self):
        with patch.object(m, 'sha', return_value='wrong'):
            with self.assertRaises(ValueError): m.provenance('/NO_SAVED_DATA', m.RECEIPT_SHA)


if __name__ == '__main__': unittest.main()
