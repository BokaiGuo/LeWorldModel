"""In-memory fake dynamics, never a simulator, native model, or disk fixture."""
from copy import deepcopy
import hashlib
import io
from types import SimpleNamespace as NS
import sys
import unittest
from unittest.mock import patch
import numpy as np
from experiments.spectral_reachability import p1_constructive_moving_history_source_20261027_v1 as m


class MemoryStore:
    def __init__(self):self.files={};self.events=[];self.previous=None
    def put(self,name,value,final=False):
        if name in self.files:raise ValueError('exclusive memory file')
        self.files[name]=m.raw(value);return hashlib.sha256(self.files[name]).hexdigest()
    def emit(self,kind,**fields):
        e=dict(index=len(self.events),previous_sha256=self.previous,kind=kind,**deepcopy(fields));e['event_sha256']=m.source.digest(e)
        self.events.append(e);self.previous=e['event_sha256']
    def arrays(self,name,**arrays):
        b=io.BytesIO();np.savez_compressed(b,**arrays);self.files[name]=b.getvalue();return hashlib.sha256(b.getvalue()).hexdigest()
    def frame(self,name,value):
        self.files[name]=value.tobytes();return hashlib.sha256(self.files[name]).hexdigest()


class FakeEnv:
    def __init__(self,fail_step=False,close_error=False):
        self.unwrapped=self;self.relative=True;self.action_scale=100;self.dt=.01;self.control_hz=10;self.with_target=False
        self.agent=NS(position=(100.,100.),velocity=(0.,0.),angle=0.,angular_velocity=0.,force=(0.,0.),torque=0.,is_sleeping=False)
        self.block=NS(position=(150.,150.),velocity=(0.,0.),angular_velocity=0.,angle=0.,shapes=[],force=(0.,0.),torque=0.,is_sleeping=False)
        self.latest_action=None;self.commands=[];self.resets=[]
        self.fail_step=fail_step;self.close_error=close_error;self.goal_state=None;self.state=np.zeros(7)
    def _get_obs(self):return self.state.copy()
    def _set_state(self,state):self.state=np.array(state,dtype=float);self.block.angle=self.state[4];self.agent.position=tuple(self.state[:2]);self.block.position=tuple(self.state[2:4]);self.space.step(.01)
    def _set_goal_state(self,goal):self.goal_state=goal.copy()
    def info(self):return dict(goal_state=self.goal_state.copy(),goal=self._goal.copy())
    def reset(self,seed,options=None):
        self.resets.append((seed,deepcopy(options)));self.space=NS(step=lambda dt:None)
        goal=np.array([400.,400.,150.,150.,0.,0.,0.]) if options is None else options['goal_state']
        self._set_state(goal);self._set_goal_state(goal);self._goal=self.render()
        self._set_state(np.array([100.,100.,150.,150.,0.,0.,0.]) if options is None else options['state'])
        return dict(state=self._get_obs()),self.info()
    def render(self):
        rgb=np.full((224,224,3),int(self.state[0])%200,np.uint8);rgb[0,0]=[65,105,225];return rgb
    def eval_state(self,goal,state):
        a=abs(goal[4]-state[4]);return bool(np.linalg.norm(goal[:4]-state[:4])<20 and min(a,2*np.pi-a)<np.pi/9),float(np.linalg.norm(goal-state))
    def step(self,command):
        self.latest_action=command.copy();self.commands.append(command.copy())
        for _ in range(10):self.space.step(.01)
        if self.fail_step:raise RuntimeError('first fake physics failure')
        self.state[:2]+=20*np.asarray(command);self.state[2:4]+=2*np.asarray(command)
        self.agent.position=tuple(self.state[:2]);self.block.position=tuple(self.state[2:4])
        success,dist=self.eval_state(self.goal_state,self.state)
        return dict(state=self._get_obs()),-dist,success,False,self.info()
    def close(self):
        if self.close_error:raise RuntimeError('secondary fake close failure')


def plan():return m.build(60)


def execute(env=None,p=None):
    store=MemoryStore();env=FakeEnv() if env is None else env
    with patch.object(m,'validate_env',lambda body:None),patch.object(m.source,'sample_state',lambda *a:np.array([100.,100.,150.,150.,0.,0.,0.])):
        r=m.execute(plan() if p is None else p,store,lambda:env,authorization=m.AUTH)
    return r,store,env


class MovingHistoryTests(unittest.TestCase):
    def test_prepare_and_verify_no_native_import(self):
        p=plan();self.assertEqual(m.verify(p),p);self.assertEqual(len(p['slots']),2)
        self.assertEqual(p['slots'],m.old.read_prepared(m.RECIPE_PREPARED.read_bytes())['slots'])
        self.assertFalse(p['P1_admission']);self.assertEqual(p['caps']['physics_step'],1224)

    def test_tampered_caps_and_recipe_rejected(self):
        for key,value in (('caps',{'public_step':99999}),('witness','choose best outcome'),('goal_attachment','restore privileged poststate')):
            p=plan();p[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):m.verify(p)

    def test_real_recipe_and_envelope_bound(self):
        p=plan();self.assertEqual(p['original_recipe']['sha256'],m.RECIPE_SHA)
        encoded=m.prepared_bytes(p)
        self.assertLess(len(encoded),200_000);self.assertEqual(m.read_prepared(encoded),p)
        self.assertEqual(p['global_freshness'],'reused_recipe_calibration_not_fresh_not_same_task')

    def test_envelope_semantic_hash_tamper_rejected(self):
        import json
        e=json.loads(m.prepared_bytes(plan()));e['plan_sha256']='0'*64
        with self.assertRaises(ValueError):m.read_prepared(m.raw(e))

    def test_final_report_failure_never_complete(self):
        put=MemoryStore.put
        def failing_put(store,name,value,final=False):
            if final:raise OSError('mock final report write failure')
            return put(store,name,value,final)
        with patch.object(MemoryStore,'put',failing_put):r,_,_=execute()
        self.assertEqual(r['status'],'engineering_failed');self.assertFalse(r['sourcepack_complete'])
        self.assertIn('mock final report write failure',r['first_error']['message'])

    def test_wrong_recipe_sha_path_wall_and_protocol_rejected(self):
        for kwargs in (dict(wall_seconds=601),dict(wall_seconds=True),
                       dict(wall_seconds=60,recipe_sha='0'*64),
                       dict(wall_seconds=60,recipe_prepared=m.ROOT/'wrong.json')):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):m.build(**kwargs)
        for key,value in (('raw_warmup',np.zeros((10,2)).tolist()),('caps',{}),
                          ('slots',list(reversed(plan()['slots']))),('consumers','silently connected')):
            p=plan();p[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):m.verify(p)

    def test_fixed_witness_support_and_sign(self):
        s=np.array([100.,100.,150.,150.,0.,0.,0.]);cap=.75*np.sqrt(5)
        x,a=m.witness(s,1,cap);y,b=m.witness(s,-1,cap)
        self.assertEqual(a.dtype,np.float32);self.assertTrue((np.linalg.norm(x.reshape(2,10),axis=1)<=cap).all())
        self.assertTrue(np.array_equal(a[:5],b[:5]));self.assertFalse(np.array_equal(a[5:],b[5:]))

    def test_native_reach_and_payload_privilege_separation(self):
        r,store,_=execute();self.assertEqual(r['status'],'fixed_slots_processed');self.assertEqual(r['reachable_count'],2)
        for row in r['rows']:
            self.assertEqual(row['replay_native_success'],[True,True]);self.assertTrue(all(0<n<=10 for n in row['executed_replay_steps']))
            with np.load(io.BytesIO(store.files[row['blind_payload']]),allow_pickle=False) as z:
                self.assertEqual(set(z.files),{'rgb_history','raw_action_history','goal_rgb'})
                self.assertEqual(z['rgb_history'].shape,(3,224,224,3));self.assertTrue(m.exact(z['raw_action_history'],m.warmup_commands()))
        self.assertFalse(r['direct_difficulty_required']);self.assertFalse(r['P1_admission'])
        for k,c in r['counts'].items():self.assertLessEqual(c['attempted'],plan()['caps'][k]);self.assertEqual(c['attempted'],c['completed'])

    def test_goal_attachment_does_not_step_or_restore(self):
        env=FakeEnv();env.reset(seed=1);before=m.snapshot(env);goal=np.array([120.,120.,150.,150.,0.,0.,0.]);rgb=env.render()
        with patch.object(env,'_set_state',side_effect=AssertionError('no state restore')),patch.object(env.space,'step',side_effect=AssertionError('no physics')):
            m.attach_goal(env,goal,rgb)
        self.assertEqual(before,m.snapshot(env));self.assertTrue(np.array_equal(env.goal_state,goal))

    def test_no_authorization_no_factory(self):
        with self.assertRaises(ValueError):m.execute(plan(),MemoryStore(),lambda:self.fail('factory called'),authorization='no')

    def test_guard_rejected_no_fake_eligible_or_replacement(self):
        with patch.object(m,'guard',side_effect=m.Rejected('mock RGB guard')):
            r,store,_=execute()
        self.assertEqual(r['reachable_count'],0);self.assertEqual(len(r['rows']),2)
        self.assertTrue(all(x['status']=='rejected_fixed_candidate' for x in r['rows']))
        self.assertFalse(any(n.endswith('blind_inputs.npz') for n in store.files))

    def test_first_error_survives_close_failure(self):
        r,_,env=execute(FakeEnv(fail_step=True,close_error=True))
        self.assertEqual(r['status'],'engineering_failed');self.assertIn('first fake physics failure',r['first_error']['message'])
        self.assertIn('secondary fake close failure',str(r['errors']));self.assertEqual(r['counts']['public_step'],dict(attempted=1,completed=0))
        self.assertNotIn('reset',env.__dict__);self.assertNotIn('_set_state',env.__dict__)

    def test_journal_chain_and_commit_before_future(self):
        _,store,_=execute();previous=None
        for i,event in enumerate(store.events):
            self.assertEqual(event['index'],i);self.assertEqual(event['previous_sha256'],previous)
            self.assertEqual(event['event_sha256'],m.source.digest({k:v for k,v in event.items() if k!='event_sha256'}));previous=event['event_sha256']
        self.assertEqual(store.events[-1]['kind'],'run_finished')
        self.assertTrue(all(f's{i}_{phase}_commit.json' in store.files for i in range(2) for phase in ('construction','replay0','replay1')))

    def test_cap_and_wall_fail_before_operation(self):
        p=plan();p['caps']['public_step']=0;ops=m.Ops(p,MemoryStore())
        with self.assertRaises(ValueError):ops.call('public_step',lambda:self.fail('dispatched'))
        p=plan();ops=m.Ops(p,MemoryStore());ops.started-=1000
        with self.assertRaises(ValueError):ops.call('reset',lambda:self.fail('dispatched'))

    def test_z_import_no_neural_physics(self):
        self.assertFalse(any(n in sys.modules for n in ('torch','gymnasium','pymunk','stable_worldmodel')))

    def test_raw_not_logits_or_zero_and_really_dispatched(self):
        r,store,env=execute()
        self.assertEqual(r['reachable_count'],2)
        receipts=[e for e in store.events if e['kind']=='warmup_return']
        self.assertEqual(len(receipts),60)
        for slot in (0,1):
            for phase in ('construction','replay0','replay1'):
                a=[e for e in receipts if e['slot']==slot and e['phase']==phase]
                self.assertEqual([x['step'] for x in a],list(range(1,11)))
                self.assertTrue(m.exact(np.asarray([x['raw_action'] for x in a],np.float32),m.warmup_commands()))
                self.assertEqual([x['raw_float32_bits'] for x in a],[x.tobytes().hex() for x in m.warmup_commands()])
        self.assertTrue(m.exact(env.commands[:10],m.warmup_commands()))
        self.assertFalse(m.exact(m.warmup_commands(),np.tanh(m.warmup_commands())))
        self.assertEqual(len(env.resets),12)
        # Fake renderer intentionally observes x only: y motion need not change RGB.
        # This is descriptive, never an eligibility/selection condition.
        self.assertTrue(all(x['observed_history_changes']==[True,False] for x in r['rows']))

    def test_guard_before_selection_saves_failed_frame_and_full_denominator(self):
        with patch.object(m,'guard',side_effect=m.Rejected('no blue')):
            r,store,_=execute()
        self.assertEqual(r['fixed_denominator'],2);self.assertEqual(r['reachable_count'],0)
        for slot in (0,1):
            name=f's{slot}_construction_00.png'
            self.assertIn(name,store.files)
            kinds=[x['kind'] for x in store.events if x.get('slot')==slot]
            self.assertLess(kinds.index('fresh_observation'),kinds.index('fresh_guard_rejected'))
        self.assertEqual(r['counts']['public_step']['attempted'],0)

    def test_dynamics_drift_replay_failclosed(self):
        env=FakeEnv();step=env.step
        def changed(command):
            result=step(command)
            if len(env.resets)>=4:env.block.velocity=(1.,0.)
            return result
        env.step=changed
        r,store,_=execute(env)
        self.assertEqual(r['status'],'engineering_failed')
        self.assertIn('independent original recipe/history reconstruction',r['first_error']['message'])
        self.assertFalse(any(x.endswith('blind_inputs.npz') for x in store.files))
        self.assertEqual(r['rows'][1]['status'],'not_started')

    def test_native_latest_action_substitution_rejected(self):
        env=FakeEnv();step=env.step
        def changed(command):
            result=step(command);env.latest_action=np.zeros(2,np.float32);return result
        env.step=changed
        r,_,_=execute(env)
        self.assertEqual(r['status'],'engineering_failed')
        self.assertIn('actual native latest_action',r['first_error']['message'])
        self.assertEqual(r['counts']['public_step'],dict(attempted=1,completed=1))

    def test_plain_zero_payload_and_privileged_extra_rejected_in_ram(self):
        payload=dict(rgb_history=np.zeros((3,224,224,3),np.uint8),raw_action_history=m.warmup_commands(),
                     goal_rgb=np.zeros((224,224,3),np.uint8))
        def load(values):
            b=io.BytesIO();np.savez_compressed(b,**values);b.seek(0)
            with patch.object(m.source,'file_sha',return_value='mock'),patch.object(m.np,'load',return_value=np.load(b,allow_pickle=False)):
                return m.load_blind('not-a-real-file','mock')
        self.assertTrue(m.exact(load(payload)['raw_action_history'],m.warmup_commands()))
        for changed in (dict(payload,raw_action_history=np.zeros((10,2),np.float32)),
                        dict(payload,witness_commands=np.zeros((10,2))),
                        dict(payload,raw_action_history=m.warmup_commands().astype(float))):
            with self.assertRaises(ValueError):load(changed)

    def test_witness_computed_at_new_postwarmup_root_and_goal_new(self):
        r,store,_=execute()
        for row in r['rows']:
            with np.load(io.BytesIO(store.files[row['certificate']]),allow_pickle=False) as z:
                root=z['observed_states'][-1]
                self.assertFalse(np.array_equal(root,z['observed_states'][0]))
                x,a=m.witness(root,row['slot']['sign'],plan()['packet_logit_cap'])
                self.assertTrue(m.exact(x,z['witness_logits']))
                self.assertTrue(m.exact(a,z['witness_commands']))
                self.assertEqual(m.source.root_hash(root,z['goal']),row['root_sha256'])
        self.assertFalse(r['same_task_as_old']);self.assertFalse(r['consumers_connected'])

    def test_rejected_warmup_keeps_partial_raw_and_no_retry(self):
        env=FakeEnv();step=env.step
        def changed(command):
            obs,reward,_,truncated,info=step(command)
            return obs,reward,True,truncated,info
        env.step=changed
        r,store,_=execute(env)
        self.assertEqual(r['reachable_count'],0);self.assertEqual(len(env.commands),2)
        self.assertEqual(r['counts']['public_step']['completed'],2)
        self.assertEqual(len([e for e in store.events if e['kind']=='warmup_return']),2)
        self.assertTrue(all(x['status']=='rejected_fixed_candidate' for x in r['rows']))

    def test_snapshot_body_velocity_force_and_signed_bits(self):
        env=FakeEnv();env.reset(seed=1);a=m.snapshot(env)
        env.block.force=(1.,0.);self.assertNotEqual(a,m.snapshot(env))
        env.block.force=(-0.,0.);self.assertNotEqual(a,m.snapshot(env))


if __name__ == '__main__':unittest.main()
