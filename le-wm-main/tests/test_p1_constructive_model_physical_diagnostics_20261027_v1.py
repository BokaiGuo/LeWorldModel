"""Saved bytes/arithmetic only. No temp files, producer imports, fit or simulator."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
import numpy as np
from experiments.spectral_reachability import p1_constructive_model_physical_diagnostics_20261027_v1 as m

S=m.PROJECT/'results/p1_constructive_reachable_source_20261027_v1_main01/sourcepack.json'
P=m.PROJECT/'results/p1_constructive_blind_planners_cuda_20261027_v1_cuda_main01/report.json'
F=m.PROJECT/'results/p1_constructive_plan_physical_20261027_v1_main01/report.json'
E=m.PROJECT/'results/p1_constructive_selected_rgb_cuda_20261027_v1_cuda_main01/report.json'
E_SHA='c77a5d208acb813ad968afe24ef0c547f08a83ff9968a984b29502baf0dda770'


def raw(x):return (m.canonical(x)+'\n').encode()


def reseal(events,r,files):
    previous=None
    for i,e in enumerate(events):
        e['index']=i;e['previous_sha256']=previous
        e['event_sha256']=m.digest({k:v for k,v in e.items() if k != 'event_sha256'});previous=e['event_sha256']
    b=b''.join(raw(e) for e in events)
    files[str(F.parent/'journal.jsonl')]=b;r['journal_sha256']=m.sha(b)
    b=raw(r);files[str(F)]=b
    return m.sha(b)


@unittest.skipUnless(S.exists() and P.exists() and F.exists() and E.exists(),'actual selected inputs absent')
class SelectedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result=m.run_analysis(S,m.SOURCE_SHA,P,m.PLANNER_SHA,F,m.PHYSICAL_SHA,selected_path=E,selected_sha=E_SHA)
        if cls.result['errors']:raise AssertionError(cls.result['errors'])
        cls.files={p:Path(p).read_bytes() for p in cls.result['input_bindings_sha256']}

    def analyze(self,files,h=E_SHA):
        return m.run_analysis(S,m.SOURCE_SHA,P,m.PLANNER_SHA,F,m.PHYSICAL_SHA,m.Reader(files.__getitem__),E,h)

    def tamper(self,edit):
        files=dict(self.files);r=m.decode(files[str(E)]);edit(r,files);files[str(E)]=raw(r)
        out=self.analyze(files,m.sha(files[str(E)]))
        self.assertEqual(out['status'],'unverified_or_inconsistent');self.assertNotIn('selected_rgb',out)

    def test_actual_eight_latents_q_and_prediction_error(self):
        rows=self.result['selected_rgb']['rows'];self.assertEqual(len(rows),8)
        expected=[1.0826473758,.82554,.91204,.85647,1.41231,1.53792,.48967,.38515]
        for row,q in zip(rows,expected):self.assertAlmostEqual(row['metrics']['actual_Q'],q,places=4)
        self.assertTrue(rows[4]['metrics']['Q_improvement_sign_reversed'])
        self.assertGreater(rows[6]['metrics']['predicted_vs_real_mse'],0)
        self.assertFalse(self.result['P1_admission'])
        self.assertEqual(sum(Path(p).suffix == '.npy' for p in self.files),8)

    def test_wrong_report_sha(self):
        self.assertEqual(self.analyze(self.files,'0'*64)['status'],'unverified_or_inconsistent')

    def test_missing_image_denominator(self):
        self.tamper(lambda r,f:r['rows'].pop())

    def test_model_identity_tamper(self):
        self.tamper(lambda r,f:r.update(model_hash_after='0'*64))

    def test_array_bytes_tamper(self):
        def edit(r,f):
            p=str(E.parent/r['rows'][0]['latent']['file']);f[p]=f[p][:-1]+bytes([f[p][-1]^1])
        self.tamper(edit)

    def test_actual_metrics_tamper(self):
        self.tamper(lambda r,f:r['rows'][0]['metrics'].update(actual_Q=0))

    def test_own_checkpoint_binding_tamper(self):
        self.tamper(lambda r,f:r['rows'][0]['provenance']['checkpoint'].update(pixel_sha256='0'*64))

    def test_journal_module_count_tamper(self):
        def edit(r,f):
            p=str(E.parent/'events.jsonl');events=[m.decode(x) for x in f[p].splitlines()]
            events[1]['entry']['resource']='predict';f[p]=b''.join(raw(x) for x in events)
            r['artifact_sha256']['events.jsonl']=m.sha(f[p])
        self.tamper(edit)

    def test_narrow_array_name_and_manifest(self):
        reader=m.Reader(lambda p:self.fail('not authorized'))
        with self.assertRaises(ValueError):reader.native_latent(E,0,dict(file='../witness.npy',dtype='float32',shape=[192],sha256='a'*64),{})


class PureTests(unittest.TestCase):
    def test_metric_keeps_agent_in_native_and_object_separate(self):
        g=np.zeros(7);s=g.copy();s[0]=21
        d=m.metric(g,s)
        self.assertFalse(d['native_success']);self.assertEqual(d['object_position_error'],0)
        self.assertIsNone(d['object_success'])

    def test_rotation_is_wrapped_net_not_raw_or_path(self):
        r=np.zeros(7);r[4]=2*np.pi-.1;s=r.copy();s[4]=.1
        d=m.diagnostics(r,r,s)
        self.assertAlmostEqual(d['object_wrapped_rotation_signed_from_initial'],.2)
        self.assertEqual(d['rotation_scope'],'wrapped net change, not total path rotation')

    def test_native_negative_angle_bug_not_silently_fixed(self):
        g=np.zeros(7);s=g.copy();s[4]=2*np.pi+1
        d=m.metric(g,s)
        self.assertTrue(d['native_success']);self.assertTrue(d['negative_native_angle'])
        self.assertFalse(d['wrapped_allxy_success'])

    def test_native_threshold_strictness(self):
        g=np.zeros(7);s=g.copy();s[2]=20
        self.assertFalse(m.metric(g,s)['native_success'])
        s[2]=0;s[4]=np.pi/9
        self.assertFalse(m.metric(g,s)['native_success'])

    def test_no_new_non_greedy_scientific_threshold(self):
        r=np.array([100.,100.,150.,150.,0.,0.,0.]);g=r.copy();g[2]=200;s=r.copy();s[2]=160
        d=m.diagnostics(r,g,s)
        self.assertEqual(d['object_displacement_from_initial'],10)
        self.assertEqual(d['changes_from_initial']['object_position_error'],-10)
        self.assertFalse(d['native_nonprogress_descriptive'])

    def test_native_reward_distance_decomposition_not_success_redefinition(self):
        root=np.zeros(7);s=np.array([3.,4.,0.,0.,0.,12.,0.])
        d=m.diagnostics(root,root,s)
        self.assertEqual(d['native_squared_distance_components'],dict(agent_xy=25.,object_xy=0.,raw_angle=0.,agent_velocity=144.))
        self.assertEqual(d['metrics']['native_distance'],13.)
        self.assertTrue(d['metrics']['native_success'])  # Success excludes velocity.

    def test_reader_rejects_privileged_files_before_loader(self):
        def denied(path):self.fail('forbidden file opened: '+path)
        reader=m.Reader(denied)
        for name in ('certificate.npz','s0_construction_certificate.json','witness.json','latent.npy'):
            with self.assertRaises(ValueError):reader.raw('/mock/'+name,'a'*64)

    def test_sha_duplicate_key_and_flat_child_fail_closed(self):
        reader=m.Reader(lambda p:b'{}')
        with self.assertRaises(ValueError):reader.raw('/mock/report.json','0'*64)
        with self.assertRaises(ValueError):reader.child('/mock/report.json','../escape.json',m.sha(b'{}'))
        with self.assertRaises(ValueError):m.decode(b'{"a":1,"a":2}')
        with self.assertRaises(ValueError):m.decode(b'{"a":NaN}')

    def test_no_producer_environment_model_imports(self):
        self.assertFalse(any(k in sys.modules for k in ('torch','gymnasium','pymunk','stable_worldmodel',
            'experiments.spectral_reachability.'+m.SOURCE,'experiments.spectral_reachability.'+m.PLANNER,
            'experiments.spectral_reachability.'+m.PHYSICAL)))


@unittest.skipUnless(S.exists() and P.exists() and F.exists(),'actual saved inputs absent')
class ActualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result=m.run_analysis(S,m.SOURCE_SHA,P,m.PLANNER_SHA,F,m.PHYSICAL_SHA)
        if cls.result['status'] != 'saved_model_physical_diagnostics_complete':raise AssertionError(cls.result['errors'])
        cls.files={p:Path(p).read_bytes() for p in cls.result['input_bindings_sha256']}

    def analyze(self,files=None,physical_sha=m.PHYSICAL_SHA,physical=True):
        files=self.files if files is None else files
        return m.run_analysis(S,m.SOURCE_SHA,P,m.PLANNER_SHA,F if physical else None,physical_sha if physical else None,m.Reader(files.__getitem__))

    def altered(self,edit):
        f=dict(self.files);r=m.decode(f[str(F)]);events=[m.decode(x) for x in f[str(F.parent/'journal.jsonl')].splitlines()]
        edit(r,events,f);h=reseal(events,r,f)
        return self.analyze(f,h)

    def test_actual_complete_four_branches(self):
        r=self.result
        self.assertEqual(r['png_frames_checked'],88);self.assertEqual(r['physical_journal_events'],2314)
        self.assertEqual(len(r['physical_rows']),4)
        self.assertEqual(r['physical_counts']['physics_step'],dict(attempted=816,completed=816))
        self.assertEqual(r['physical_counts']['public_step'],dict(attempted=80,completed=80))
        self.assertTrue(all(x['full_horizon'] and not x['official_native_success_at_last_return'] for x in r['physical_rows']))
        self.assertFalse(r['P1_admission']);self.assertEqual(r['capability_change'],'not_measured')

    def test_actual_independent_endpoint_displacement(self):
        rows={(x['root_slot'],x['arm']):x for x in self.result['physical_rows']}
        expected={(0,m.ARMS[0]):(48.29448372956648,47.976995121251164),
                  (0,m.ARMS[1]):(30.70442080761503,64.25127865978467),
                  (1,m.ARMS[0]):(24.027056779866772,130.92915162616924),
                  (1,m.ARMS[1]):(132.97005656781644,45.39850942624763)}
        for k,(move,error) in expected.items():
            d=rows[k]['checkpoints']['10']
            self.assertAlmostEqual(d['object_displacement_from_initial'],move)
            self.assertAlmostEqual(d['metrics']['object_position_error'],error)
        self.assertEqual(rows[1,m.ARMS[0]]['checkpoints']['5']['object_displacement_from_initial'],0)

    def test_input_access_never_loads_witness_or_npz(self):
        for p in self.files:
            self.assertNotIn('certificate',Path(p).name)
            self.assertNotIn('witness',Path(p).name)
            self.assertNotIn(Path(p).suffix,('.npz','.npy'))

    def test_interface_waits_without_physical_not_fake_result(self):
        r=self.analyze(physical=False)
        self.assertEqual(r['status'],'awaiting_explicit_physical_report_and_SHA')
        self.assertEqual(len(r['models']),4);self.assertNotIn('physical_rows',r)

    def test_wrong_physical_SHA_not_promoted(self):
        r=self.analyze(physical_sha='0'*64)
        self.assertEqual(r['status'],'unverified_or_inconsistent');self.assertNotIn('comparisons',r)

    def test_survivor_only_physical_denominator_rejected(self):
        def edit(r,e,f):r['rows'].pop()
        r=self.altered(edit);self.assertIn('fixed4',r['errors'][0]['message'])

    def test_rechained_wrong_actual_action_rejected(self):
        def edit(r,e,f):
            x=next(x for x in e if x['kind'] == 'returned_transition' and x['phase'] == 'plan')
            x['command']=[0.,0.]
        r=self.altered(edit);self.assertIn('actual action',r['errors'][0]['message'])

    def test_rechained_wrong_native_label_rejected(self):
        def edit(r,e,f):
            x=next(x for x in e if x['kind'] == 'returned_transition' and x['phase'] == 'plan')
            x['actual_native_success']=True
        r=self.altered(edit);self.assertIn('native return',r['errors'][0]['message'])

    def test_rechained_goal_mismatch_rejected(self):
        def edit(r,e,f):r['rows'][0]['goal'][0]+=1
        r=self.altered(edit);self.assertEqual(r['status'],'unverified_or_inconsistent')

    def test_wrong_PNG_bytes_rejected(self):
        f=dict(self.files);name=str(F.parent/'b0_plan_05.png');f[name]+=b'X'
        r=self.analyze(f);self.assertIn('saved input SHA',r['errors'][0]['message'])

    def test_API_counts_rechained_report_still_recomputed(self):
        def edit(r,e,f):
            r['counts']['public_step']['completed']-=1;e[-1]['counts']=deepcopy(r['counts'])
        r=self.altered(edit);self.assertIn('physical ledger',r['errors'][0]['message'])

    def test_partial_no_endpoint_padding_or_survivor_comparison(self):
        def edit(r,e,f):
            cutoff=next(x['index'] for x in e if x.get('branch') == 3 and x['kind'] == 'operation_attempted' and
                x['resource'] == 'public_step' and x.get('phase') == 'plan' and x['step'] == 5)
            e[:]=[x for x in e if not (x.get('branch') == 3 and x['index'] >= cutoff)]
            row=r['rows'][3]
            last=next(x for x in e if x.get('branch') == 3 and x['kind'] == 'returned_transition' and x.get('phase') == 'plan' and x['step'] == 4)
            fresh=next(x['receipt'] for x in e if x.get('branch') == 3 and x['kind'] == 'fresh_observation' and x['receipt']['phase'] == 'plan' and x['receipt']['step'] == 4)
            row.update(status='failed',own_attempted=4,own_returned=4,PNG=16,full_horizon=False,endpoint_scope='partial',
                final_state=last['state'],final_metrics=last['metrics'],last_observation=fresh,
                checkpoints=row['checkpoints'][:1],first_error=dict(stage='mock observer',type='RuntimeError',message='mock partial only'))
            for x in e:
                if x['kind'] == 'branch_closed' and x['receipt']['root_slot'] == 1 and x['receipt']['arm'] == m.ARMS[1]:x['receipt']=deepcopy(row)
            counts={k:dict(attempted=0,completed=0) for k in r['counts']}
            for x in e:
                if x['kind'] in ('operation_attempted','operation_completed'):
                    counts[x['resource']]['attempted' if x['kind'] == 'operation_attempted' else 'completed']+=1
            r.update(status='partial_or_failed',counts=counts)
            e[-1].update(status=r['status'],counts=deepcopy(counts))
        r=self.altered(edit)
        self.assertEqual(r['status'],'saved_partial_physical_diagnostics',r['errors'])
        self.assertEqual(len(r['physical_rows']),4)
        row=r['physical_rows'][3]
        self.assertEqual(row['final_step'],4);self.assertEqual(row['checkpoints']['5']['availability'],'not_observed')
        self.assertFalse(r['comparisons'][1]['step10']['comparable'])
        self.assertFalse(r['P1_admission'])


if __name__ == '__main__':unittest.main()
