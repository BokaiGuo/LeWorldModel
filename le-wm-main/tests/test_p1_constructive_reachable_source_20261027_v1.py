"""In-memory fake dynamics, never a simulator, native model, or disk fixture."""
from copy import deepcopy
import hashlib
import io
from types import SimpleNamespace as NS
import sys
import unittest
from unittest.mock import patch
import numpy as np
from experiments.spectral_reachability import p1_constructive_reachable_source_20261027_v1 as m


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
        self.block=NS(velocity=(0.,0.),angular_velocity=0.,angle=0.,shapes=[])
        self.fail_step=fail_step;self.close_error=close_error;self.goal_state=None;self.state=np.zeros(7)
    def _get_obs(self):return self.state.copy()
    def _set_state(self,state):self.state=np.array(state,dtype=float);self.block.angle=self.state[4];self.space.step(.01)
    def _set_goal_state(self,goal):self.goal_state=goal.copy()
    def info(self):return dict(goal_state=self.goal_state.copy(),goal=self._goal.copy())
    def reset(self,seed,options=None):
        self.space=NS(step=lambda dt:None)
        goal=np.array([400.,400.,150.,150.,0.,0.,0.]) if options is None else options['goal_state']
        self._set_state(goal);self._set_goal_state(goal);self._goal=self.render()
        self._set_state(np.array([100.,100.,150.,150.,0.,0.,0.]) if options is None else options['state'])
        return dict(state=self._get_obs()),self.info()
    def render(self):
        rgb=np.full((224,224,3),int(self.state[0])%200,np.uint8);rgb[0,0]=[65,105,225];return rgb
    def eval_state(self,goal,state):
        a=abs(goal[4]-state[4]);return bool(np.linalg.norm(goal[:4]-state[:4])<20 and min(a,2*np.pi-a)<np.pi/9),float(np.linalg.norm(goal-state))
    def step(self,command):
        for _ in range(10):self.space.step(.01)
        if self.fail_step:raise RuntimeError('first fake physics failure')
        self.state[:2]+=20*np.asarray(command);self.state[2:4]+=2*np.asarray(command)
        success,dist=self.eval_state(self.goal_state,self.state)
        return dict(state=self._get_obs()),-dist,success,False,self.info()
    def close(self):
        if self.close_error:raise RuntimeError('secondary fake close failure')


def plan():
    f=str(m.source.PROJECT/'experiments/spectral_reachability/p1_conditional_procedural_source_20261027_v1.py')
    e=dict(schema=m.source.NAME+'.exclusions',environment='swm/PushT-v1',files={f:m.source.file_sha(f)},root_hashes=[],seed_families=[],reset_spans=[],unknown_sources=[],coverage=dict(status='unknown',authority_sha256=None,scope='mock only'),physical_future_labels_read=False)
    return m.build([202798101,202798102],[202798201,202798202],60,history=e)


def execute(env=None,p=None):
    store=MemoryStore();env=FakeEnv() if env is None else env
    with patch.object(m,'validate_env',lambda body:None),patch.object(m.source,'sample_state',lambda *a:np.array([100.,100.,150.,150.,0.,0.,0.])):
        r=m.execute(plan() if p is None else p,store,lambda:env,authorization=m.AUTH)
    return r,store,env


class ConstructiveTests(unittest.TestCase):
    def test_prepare_and_verify_no_native_import(self):
        p=plan();self.assertEqual(m.verify(p),p);self.assertEqual(len(p['slots']),2)
        self.assertFalse(p['P1_admission']);self.assertEqual(p['caps']['physics_step'],1224)

    def test_tampered_caps_and_recipe_rejected(self):
        for key,value in (('caps',{'public_step':99999}),('witness','choose best outcome'),('goal_attachment','restore privileged poststate')):
            p=plan();p[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):m.verify(p)

    def test_real_metadata_exclusions_content_id_not_path(self):
        p=m.build([202798101,202798102],[202798201,202798202],300)
        self.assertIn('metadata_authority_content_id',p['exclusions']['files'])
        self.assertEqual(m.verify(p),p)
        encoded=m.prepared_bytes(p)
        self.assertLess(len(encoded),200_000);self.assertEqual(m.read_prepared(encoded),p)

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

    def test_explicit_seed_and_wall_bounds(self):
        for roots,resets,wall in (([1],[2],60),([1,1],[2,3],60),([1,2],[3,4],601)):
            with self.assertRaises(ValueError):m.build(roots,resets,wall,history=plan()['exclusions'])

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
                self.assertEqual(z['rgb_history'].shape,(3,224,224,3));self.assertEqual(z['raw_action_history'].dtype,np.float32)
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


if __name__ == '__main__':unittest.main()
