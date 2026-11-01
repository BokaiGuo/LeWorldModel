"""Saved arithmetic and actual evidence, no simulator/model/tmp fixtures."""
from copy import deepcopy
import sys
import unittest
import numpy as np
from experiments.spectral_reachability import p1_pusht_task_feasibility_20261027_v1 as m


class FeasibilityTests(unittest.TestCase):
    def test_zero_support(self):
        r = m.action_support(np.zeros((10, 2)), .75*np.sqrt(5), [0,0], [1,1])
        self.assertEqual(r['packet_logit_norms'], [0,0]); self.assertEqual(r['nominal_target_offset_sum_px'], 0)

    def test_separate_packet_balls_not_full20_ball(self):
        cap = .75*np.sqrt(5); logits = np.zeros((2,10), np.float32); logits[:,0] = cap
        r = m.action_support(np.tanh(logits).reshape(10,2).astype(float), cap, [0,0], [1,1])
        self.assertTrue(all(abs(x-cap) < 2e-6 for x in r['packet_logit_norms']))
        self.assertGreater(np.linalg.norm(logits), cap)

    def test_raw_box_not_sufficient(self):
        a = np.full((10,2), np.float32(.8)).astype(float)
        with self.assertRaisesRegex(ValueError, 'ball'): m.action_support(a, .75*np.sqrt(5), [0,0], [1,1])

    def test_invalid_commands(self):
        for a in (np.zeros((5,2)), np.full((10,2), np.nan), np.ones((10,2)), np.full((10,2), .1)):
            with self.subTest(shape=a.shape), self.assertRaises(ValueError): m.action_support(a, 1, [0,0], [1,1])

    def test_invalid_normalizer(self):
        for std in ([0,1], [float('nan'),1], [1]):
            with self.assertRaises(ValueError): m.action_support(np.zeros((10,2)), 1, [0,0], std)

    def test_native_success_includes_agent(self):
        goal = np.zeros(7); state = goal.copy(); state[0] = 200
        d = m.saved.metrics(goal, state)
        self.assertFalse(d['native_formula']['success']); self.assertEqual(d['independent']['object_position_error'], 0)

    def test_wrapped_diagnostic_never_overwrites_native(self):
        goal = np.zeros(7); state = np.zeros(7); state[4] = 3*np.pi
        d = m.saved.metrics(goal, state)
        self.assertTrue(d['native_formula']['negative_angle_term']); self.assertTrue(d['native_formula']['success'])
        self.assertFalse(d['independent']['same_allxy_threshold_wrapped_success'])

    def test_velocity_reward_not_success(self):
        goal = np.zeros(7); state = goal.copy(); state[5] = 100
        d = m.saved.metrics(goal, state)
        self.assertTrue(d['native_formula']['success']); self.assertEqual(d['native_formula']['state_distance'], 100)

    def test_actual_complete_audit(self):
        report = m.audit()
        self.assertEqual(report['branches'], 262); self.assertEqual(len(report['roots']), 2)
        self.assertEqual(report['sameprefix_validation']['events'], 34196)
        self.assertFalse(report['physical_reach_witness_found']); self.assertFalse(report['global_unreachability_proven'])
        self.assertFalse(report['P1_admission']); self.assertFalse(report['training_distribution_support_verified'])
        for root in report['roots']:
            self.assertEqual(root['branches'], 131); self.assertEqual(root['success_branches'], 0)
            self.assertEqual(root['wrapped_success_branches'], 0); self.assertEqual(root['negative_native_angle_steps'], 0)

    def test_actual_wrong_root_and_command_rejected(self):
        reader = m.saved.Reader(); p,r,packets = m.saved.load_inputs(reader)
        row = r['rows'][0]; plan = p['inputs']['plans'][0]; packet = packets[row['root_sha256']]
        task = dict(packet_logit_cap=.75*np.sqrt(5), action_mean=[0,0], action_std=[1,1])
        bad = deepcopy(plan); bad['root_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'identity'): m.trajectory(row,bad,packet,task)
        bad = deepcopy(plan); bad['commands'][0][0] = .125
        with self.assertRaisesRegex(ValueError, 'command'): m.trajectory(row,bad,packet,task)

    def test_z_no_neural_or_physics_import(self):
        self.assertFalse(any(n in sys.modules for n in ('torch','gymnasium','pymunk','stable_worldmodel')))


if __name__ == '__main__': unittest.main()
