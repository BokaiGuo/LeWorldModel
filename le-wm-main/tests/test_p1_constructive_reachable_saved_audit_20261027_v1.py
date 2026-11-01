"""Pure arithmetic, in-memory mock journals, saved actual regression; no temp writes."""
import base64
from copy import deepcopy
import gzip
import io
import json
from pathlib import Path
import sys
import unittest
import numpy as np
from PIL import Image
from experiments.spectral_reachability import p1_constructive_reachable_saved_audit_20261027_v1 as m

ACTUAL_SHA = '9ad3a414fd2148258524d4a7e9d4f070f59532ec2e5943cfa61f9145b3a41bd9'
PREPARED = m.PROJECT/'results/p1_constructive_reachable_source_20261027_v1_prepare01.json'
DIRECTORY = m.PROJECT/'results/p1_constructive_reachable_source_20261027_v1_main01'


def raw(x): return (m.canonical(x)+'\n').encode()


def chain(events, receipt, files):
    previous = None
    for i,e in enumerate(events):
        e['index'] = i; e['previous_sha256'] = previous
        e['event_sha256'] = m.digest({k:v for k,v in e.items() if k != 'event_sha256'})
        previous = e['event_sha256']
    receipt['journal_chain_tail'] = previous
    files['journal.jsonl'] = b''.join(raw(e) for e in events)
    return raw(receipt)


def fixture(partial=False):
    slots = [dict(index=i,root_seed=10+i,reset_seed=20+i,sign=1 if i == 0 else -1) for i in range(2)]
    p = dict(slots=slots,caps={k:2*v for k,v in m.PER_SLOT.items()},exclusions=dict(root_hashes=[]))
    counts = {k:dict(attempted=0,completed=0) for k in m.PER_SLOT}
    rows = [dict(slot=s,status='not_started',reachable=False) for s in slots]
    r = dict(schema=m.SOURCE+'.sourcepack',prepare_id=m.digest(p),rows=rows,
        errors=[],first_error=None,status='fixed_slots_processed',P1_admission=False,new_neural_calls=0,training=False,
        new_independent_confirmation=False,direct_difficulty_required=False,global_freshness='unknown_calibration_only',
        counts=counts,sourcepack_complete=True,reachable_count=0)
    files = {}; events = [dict(kind='run_started',prepare=p)]
    def op(resource, children=(), **fields):
        counts[resource]['attempted'] += 1
        events.append(dict(kind='operation_attempted',resource=resource,**fields))
        for child in children:
            op(child, ('physics_step',) if child == 'set_state' else ())
        counts[resource]['completed'] += 1
        events.append(dict(kind='operation_completed',resource=resource,**fields))
    if partial:
        counts['reset']['attempted'] = 1
        events.append(dict(kind='operation_attempted',resource='reset',slot=0,phase='construction',kind_of_reset='seed'))
        rows[0]['status'] = 'engineering_failed'
        r.update(status='engineering_failed',sourcepack_complete=False,errors=[dict(phase='execution',type='RuntimeError',message='mock reset failed')])
        r['first_error'] = r['errors'][0]
    else:
        s = [100.,100.,150.,150.,0.,0.,0.]
        rgb = np.zeros((224,224,3),np.uint8); rgb[0,0] = [65,105,225]
        b = io.BytesIO(); Image.fromarray(rgb).save(b,format='PNG'); png = b.getvalue()
        for i,slot in enumerate(slots):
            op('reset', ('set_state','goal_render','set_state'),slot=i,phase='construction',kind_of_reset='seed')
            events.append(dict(kind='placement_recipe',slot=slot,phase='construction',original_goal=s,requested=s))
            op('reset', ('set_state','goal_render','set_state'),slot=i,phase='construction',kind_of_reset='placement')
            op('explicit_render')
            name = f's{i}_construction_00.png'; files[name] = png
            events.append(dict(kind='fresh_observation',slot=i,phase='construction',step=0,state=s,
                snapshot=dict(state=s,block_velocity=[0.,0.],block_angular_velocity=0.,block_angle=0.),
                png=name,file_sha256=m.sha(png),pixel_sha256=m.sha(rgb.tobytes()),canonical_agent_pixels=1))
            rows[i].update(status='rejected_fixed_candidate',reason='construction-recipe warmup already native terminal')
            events.append(dict(kind='slot_finished',receipt=deepcopy(rows[i])))
    events.append(dict(kind='run_finished',status=r['status'],counts=deepcopy(counts),errors=deepcopy(r['errors'])))
    return p,r,events,files


class ArithmeticTests(unittest.TestCase):
    def test_independent_witness_recipe(self):
        root = [100.,100.,150.,100.,0.,0.,0.]
        x,a = m.witness(root,1)
        np.testing.assert_array_equal(x[:5],np.tile([.5,0],(5,1)))
        np.testing.assert_array_equal(x[5:],np.tile([.25,.5],(5,1)))
        np.testing.assert_array_equal(a,np.tanh(x)); self.assertEqual(a.dtype,np.float32)
        self.assertTrue(np.all(np.linalg.norm(x.reshape(2,10),axis=1) <= .75*np.sqrt(5)))

    def test_native_agent_goal_and_wrapped_bug_separate(self):
        g = np.zeros(7); s = g.copy(); s[0] = 21
        self.assertFalse(m.native(g,s)['success'])
        s = g.copy(); s[4] = 2*np.pi+1
        d = m.native(g,s)
        self.assertTrue(d['success']); self.assertTrue(d['angle_bug']); self.assertFalse(d['wrapped_success'])

    def test_native_strict_threshold(self):
        g = np.zeros(7); s = g.copy(); s[0] = 20
        self.assertFalse(m.native(g,s)['success'])
        s[0] = 0; s[4] = np.pi/9
        self.assertFalse(m.native(g,s)['success'])

    def test_partial_attempt_not_refunded_or_admitted(self):
        p,r,e,f = fixture(True); b = chain(e,r,f)
        out = m.audit_saved(p,b,m.sha(b),f.__getitem__)
        self.assertEqual(out['status'],'saved_partial_consistent_not_admitted')
        self.assertEqual(out['fixed_denominator'],2); self.assertEqual(len(out['pending_operations']),1)
        self.assertEqual(out['counts']['reset'],dict(attempted=1,completed=0))
        self.assertFalse(out['P1_admission'])

    def test_rejected_fixed_denominator_retained(self):
        p,r,e,f = fixture(); b = chain(e,r,f)
        out = m.audit_saved(p,b,m.sha(b),f.__getitem__)
        self.assertEqual(out['fixed_denominator'],2); self.assertEqual(out['reachable_saved_evidence'],0)
        self.assertEqual(out['png_frames'],2)
        self.assertTrue(all(x['status'] == 'rejected_fixed_candidate' for x in out['rows']))

    def test_partial_false_complete_rejected(self):
        p,r,e,f = fixture(True); r['sourcepack_complete'] = True; b = chain(e,r,f)
        with self.assertRaisesRegex(ValueError,'complete flag'):m.audit_saved(p,b,m.sha(b),f.__getitem__)

    def test_survivor_only_denominator_rejected(self):
        p,r,e,f = fixture(); r['rows'].pop(); b = chain(e,r,f)
        with self.assertRaisesRegex(ValueError,'fixed slots'):m.audit_saved(p,b,m.sha(b),f.__getitem__)

    def test_duplicate_or_nonfinite_json_rejected(self):
        for b in (b'{"a":1,"a":2}',b'{"a":NaN}'):
            with self.assertRaises(ValueError):m.parse(b)

    def test_privileged_npz_field_rejected(self):
        b = io.BytesIO(); np.savez_compressed(b,rgb_history=np.zeros(1),raw_action_history=np.zeros(1),goal_rgb=np.zeros(1),state=np.zeros(7))
        with self.assertRaises(ValueError):m.arrays(b.getvalue(),('rgb_history','raw_action_history','goal_rgb'))

    def test_no_producer_or_native_import(self):
        self.assertFalse(any(k in sys.modules for k in ('torch','gymnasium','pymunk','stable_worldmodel',
            'experiments.spectral_reachability.'+m.SOURCE)))


@unittest.skipUnless(PREPARED.exists() and (DIRECTORY/'sourcepack.json').exists(),'actual saved sourcepack unavailable')
class ActualSavedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = m.decode_prepare(PREPARED.read_bytes(),m.PREPARE01_SHA)
        cls.b = (DIRECTORY/'sourcepack.json').read_bytes()
        if m.sha(cls.b) != ACTUAL_SHA:raise AssertionError('actual sourcepack external SHA')
        cls.r = m.parse(cls.b)
        cls.files = {p.name:p.read_bytes() for p in DIRECTORY.iterdir() if p.is_file()}
        cls.events = [m.parse(line) for line in cls.files['journal.jsonl'].splitlines()]

    def altered(self, edit):
        r,e,f = deepcopy(self.r),deepcopy(self.events),dict(self.files)
        edit(r,e,f); b = chain(e,r,f)
        return m.audit_saved(self.p,b,m.sha(b),f.__getitem__)

    def test_actual_complete_chain(self):
        r = m.audit_saved(self.p,self.b,ACTUAL_SHA,self.files.__getitem__)
        self.assertEqual(r['reachable_saved_evidence'],2); self.assertEqual(r['png_frames'],126)
        self.assertEqual(r['witness_commits_checked'],6); self.assertEqual(r['certificates_checked'],2)
        self.assertEqual(r['counts']['physics_step']['completed'],1224)
        self.assertFalse(any(d['angle_bug'] for row in r['rows'] for d in row['native_diagnostics']))
        self.assertFalse(r['durable_precommit_independently_proven']); self.assertFalse(r['P1_admission'])

    def test_prepare_envelope_hash_and_source_drift(self):
        b = PREPARED.read_bytes()
        with self.assertRaisesRegex(ValueError,'external prepare'):m.decode_prepare(b,'0'*64)
        with self.assertRaisesRegex(ValueError,'bound file SHA'):m.decode_prepare(b,m.PREPARE01_SHA,lambda p:'0'*64)
        e = m.parse(b); e['plan_sha256'] = '0'*64
        b = raw(e)
        with self.assertRaisesRegex(ValueError,'semantic SHA'):m.decode_prepare(b,m.sha(b))

    def test_rechained_reward_tamper_rejected(self):
        def edit(r,e,f):next(x for x in e if x['kind'] == 'native_return')['reward'] += 1
        with self.assertRaisesRegex(ValueError,'native reward'):self.altered(edit)

    def test_rechained_executed_witness_tamper_rejected(self):
        def edit(r,e,f):
            for x in e:
                if x['kind'] in ('operation_attempted','operation_completed') and x['resource'] == 'public_step' and x['step'] == 11:
                    x['raw_action'] = [0.,0.]
        with self.assertRaisesRegex(ValueError,'fixed witness'):self.altered(edit)

    def test_commit_content_tamper_rejected(self):
        def edit(r,e,f):
            c = m.parse(f['s0_construction_commit.json']); c['commands'][0] = [0.,0.]
            f['s0_construction_commit.json'] = raw(c)
        with self.assertRaisesRegex(ValueError,'tanh commands'):self.altered(edit)

    def test_png_bytes_tamper_rejected(self):
        def edit(r,e,f):f['s0_replay0_00.png'] += b'x'
        with self.assertRaisesRegex(ValueError,'PNG file SHA'):self.altered(edit)

    def test_array_entry_privilege_leak_rejected_with_new_sha(self):
        def edit(r,e,f):
            name = r['rows'][0]['blind_payload']; z = m.arrays(f[name],('rgb_history','raw_action_history','goal_rgb'))
            b = io.BytesIO(); np.savez_compressed(b,**z,witness_commands=np.zeros((10,2)))
            f[name] = b.getvalue(); r['rows'][0]['blind_payload_sha256'] = m.sha(f[name])
            for x in e:
                if x['kind'] == 'slot_finished' and x['receipt']['slot']['index'] == 0:x['receipt'] = deepcopy(r['rows'][0])
        with self.assertRaisesRegex(ValueError,'NPZ'):self.altered(edit)

    def test_broken_hashchain_rejected(self):
        f = dict(self.files); events = deepcopy(self.events); events[3]['event_sha256'] = '0'*64
        f['journal.jsonl'] = b''.join(raw(x) for x in events)
        with self.assertRaisesRegex(ValueError,'hashchain'):m.audit_saved(self.p,self.b,ACTUAL_SHA,f.__getitem__)

    def test_physics_ledger_tamper_rejected(self):
        def edit(r,e,f):
            r['counts']['physics_step']['completed'] -= 1
            e[-1]['counts'] = deepcopy(r['counts'])
        with self.assertRaisesRegex(ValueError,'API ledger'):self.altered(edit)

    def test_replay_prefix_tamper_even_rehashed_png_rejected(self):
        def edit(r,e,f):
            x = next(x for x in e if x['kind'] == 'fresh_observation' and x['phase'] == 'replay0' and x['step'] == 0)
            with Image.open(io.BytesIO(f[x['png']])) as im:rgb = np.array(im)
            rgb[0,0] = [1,2,3]; b = io.BytesIO(); Image.fromarray(rgb).save(b,format='PNG')
            f[x['png']] = b.getvalue(); x['file_sha256'] = m.sha(f[x['png']]); x['pixel_sha256'] = m.sha(rgb.tobytes())
        with self.assertRaisesRegex(ValueError,'RGB prefix'):self.altered(edit)


if __name__ == '__main__':unittest.main()
