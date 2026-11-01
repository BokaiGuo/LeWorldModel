"""In-memory fake dynamics, never a simulator, native model, or disk fixture."""
from copy import deepcopy
import hashlib
import io
from types import SimpleNamespace as NS
import sys
import unittest
from unittest.mock import patch
import numpy as np
from experiments.spectral_reachability import p1_archive_native_alignment_probe_20261027_v1 as m


class MemoryStore:
    def __init__(self):self.files={};self.events=[];self.previous=None
    def put(self,name,value,final=False):
        if name in self.files:raise ValueError('exclusive memory file')
        self.files[name]=m.raw(value);return hashlib.sha256(self.files[name]).hexdigest()
    def emit(self,kind,**fields):
        e=dict(index=len(self.events),previous_sha256=self.previous,kind=kind,**deepcopy(fields));e['event_sha256']=m.digest(e)
        self.events.append(e);self.previous=e['event_sha256']
    def arrays(self,name,**arrays):
        b=io.BytesIO();np.savez_compressed(b,**arrays);self.files[name]=b.getvalue();return hashlib.sha256(b.getvalue()).hexdigest()
    def frame(self,name,value):
        self.files[name]=value.tobytes();return hashlib.sha256(self.files[name]).hexdigest()


class FakeEnv:
    def __init__(self,fail_step=False,close_error=False):
        self.unwrapped=self;self.relative=True;self.action_scale=100;self.dt=.01;self.control_hz=10;self.with_target=False;self.render_action=False;self.render_size=224
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



class Array:
    """Sparse logical H5 dataset: no big archive, disk or simulator."""
    def __init__(self,kind,shape,dtype,offsets,lengths):
        self.kind,self.shape,self.dtype=kind,shape,np.dtype(dtype)
        self.offsets,self.lengths=offsets,lengths
    def __getitem__(self,key):
        idx=np.arange(key.start or 0,key.stop,key.step or 1) if isinstance(key,slice) else np.asarray(key)
        ords=np.searchsorted(self.offsets,idx,side='right')-1
        local=idx-self.offsets[ords]
        if self.kind=='episode_idx':return ords.astype(self.dtype)
        if self.kind=='step_idx':return local.astype(self.dtype)
        if self.kind=='action':
            return np.stack((.01*(local+1),.005*(local+1)),axis=-1).astype(self.dtype)
        if self.kind=='state':
            out=np.tile(np.array([100.,100.,150.,150.,0.,0.,0.]),(len(idx),1))
            out[:,0]+=local;return out.astype(self.dtype)
        if self.kind=='pixels':
            return np.stack([np.full((224,224,3),int(k)%255,np.uint8) for k in local])
        raise AssertionError(self.kind)


class H5(dict):
    def __init__(self,short=False):
        n=m.EPISODES
        lengths=np.full(n,125,np.int64);lengths[:m.ROWS-125*n]+=1
        if short:lengths[2]+=int(lengths[1])-21;lengths[1]=21
        offsets=np.concatenate(([0],np.cumsum(lengths)[:-1]))
        super().__init__(ep_len=lengths,ep_offset=offsets);self.attrs={}
        for k,shape,dtype in [('action',(m.ROWS,2),'f4'),('state',(m.ROWS,7),'f8'),
                              ('proprio',(m.ROWS,4),'f4'),
                              ('pixels',(m.ROWS,224,224,3),'u1'),
                              ('episode_idx',(m.ROWS,),'i8'),('step_idx',(m.ROWS,),'i8')]:
            self[k]=Array(k,shape,dtype,offsets,lengths)
    def __enter__(self):return self
    def __exit__(self,*args):return False


def saved(short=False):
    f=H5(short)
    with patch.object(m,'file_stat',return_value=dict(path=str(m.DATA),size_bytes=123,mtime_ns=1,device=1,inode=1)),\
         patch('h5py.File',return_value=f):
        return m.read_archive()


def run(env=None,short=False,store=None):
    data,payload=saved(short);env=FakeEnv() if env is None else env
    store=MemoryStore() if store is None else store
    with patch.object(m,'read_archive',return_value=(data,payload)),patch.object(m,'validate_env',lambda body:None):
        plan=m.make_plan(data)
        report=m.execute(plan,store,lambda:env,authorization=m.AUTH)
    return report,store,env


class AlignmentTests(unittest.TestCase):
    def test_read_only_h5_slice_identity(self):
        data,payload=saved()
        self.assertEqual([e['episode_ordinal'] for e in data['episodes']],[1,6229])
        for r,p in zip(data['episodes'],payload):
            self.assertEqual(r['action_slice']['array'],m.array_id(p['actions']))
            self.assertEqual(p['actions'].shape,(21,2))
            self.assertEqual([x['step'] for x in r['pixels']],list(m.CHECKPOINTS))
            self.assertEqual(r['state_slice']['array'],m.array_id(p['states']))
        self.assertEqual(data['attrs'],{})

    def test_real_proprio_schema_bound_but_not_loaded(self):
        data,_=saved()
        self.assertEqual(data['schema']['proprio'],dict(shape=[m.ROWS,4],dtype=np.dtype('f4').str))
        # Array has no proprio read implementation: schema binding must not load it.
        for bad in ('missing','shape','dtype','extra'):
            f=H5()
            if bad=='missing':del f['proprio']
            elif bad=='shape':f['proprio'].shape=(m.ROWS,3)
            elif bad=='dtype':f['proprio'].dtype=np.dtype('f8')
            else:f['unexpected']=f['proprio']
            with self.subTest(bad=bad),patch.object(m,'file_stat',return_value={}),patch('h5py.File',return_value=f),self.assertRaises(ValueError):
                m.read_archive()

    def test_short_slot_no_substitution_and_no_slices(self):
        data,p=saved(True)
        self.assertIsNone(p[0]);self.assertEqual(data['episodes'][0]['status'],'rejected_short_fixed_slot')
        self.assertEqual(data['episodes'][0]['episode_ordinal'],1)
        r,store,env=run(short=True)
        self.assertEqual([x['status'] for x in r['rows'][:2]],['rejected_short_fixed_slot']*2)
        self.assertEqual(r['fixed_denominator'],4);self.assertEqual(len(env.resets),4)
        self.assertEqual(r['counts']['public_step']['completed'],40)

    def test_metadata_time_gap_rejected(self):
        f=H5();a=f['step_idx']
        class Bad:
            def __getitem__(self,key):
                v=a[key];v[-1]+=1;return v
        f['step_idx']=Bad()
        with self.assertRaisesRegex(ValueError,'time IDs'):m.selected_episode(f,1)
        f=H5();f['ep_offset'][1]+=1
        with self.assertRaisesRegex(ValueError,'boundaries'):m.selected_episode(f,1)

    def test_no_hidden_range_clipping_or_ball_gate(self):
        f=H5();original=f['action']
        class Big:
            def __getitem__(self,key):return original[key]*1000
        f['action']=Big()
        _,p=m.selected_episode(f,1)
        self.assertGreater(float(p['actions'].max()),1.)
        self.assertEqual(p['actions'].dtype,np.float32)

    def test_literal_lag_commands_same_labels_and_all_counts(self):
        r,store,env=run()
        self.assertEqual(r['status'],'alignment_probe_processed');self.assertEqual(r['errors'],[])
        self.assertEqual(r['horizon_completed'],4);self.assertEqual(r['completed_checkpoints'],20)
        self.assertEqual(r['counts'],{k:dict(attempted=v,completed=v) for k,v in m.contract()['caps'].items()})
        self.assertEqual(len(env.resets),8)
        self.assertTrue(all(seed==0 for seed,_ in env.resets))
        _,payload=saved()
        for i,row in enumerate(r['rows']):
            ordinal,lag=row['episode_ordinal'],row['lag']
            cmds=env.commands[20*i:20*(i+1)]
            self.assertTrue(m.base.exact(np.asarray(cmds),payload[i//2]['actions'][lag:lag+20]))
            self.assertEqual(row['missing_checkpoints'],[])
            self.assertEqual([c['step'] for c in row['checkpoints']],list(m.CHECKPOINTS))
            off=r['archive']['episodes'][i//2]['offset']
            self.assertEqual([c['archive_global_row'] for c in row['checkpoints']],[off+k for k in m.CHECKPOINTS])
        self.assertIsNone(r['scientific_winner']);self.assertFalse(r['P1_admission'])
        self.assertEqual(len([n for n in store.files if n.endswith('.png')]),84)
        self.assertEqual(len([e for e in store.events if e['kind']=='reset_return']),8)
        self.assertEqual(len([e for e in store.events if e['kind']=='render_return']),92)

    def test_initial_reset_drift_is_measured_not_rejected_or_repaired(self):
        env=FakeEnv();reset=env.reset
        def changed(seed,options=None):
            obs,info=reset(seed,options)
            if options is not None:
                env.state[0]+=.25;env.agent.position=tuple(env.state[:2])
                obs=dict(state=env._get_obs())
            return obs,info
        env.reset=changed
        r,_,_=run(env)
        self.assertEqual(r['errors'],[]);self.assertEqual(r['horizon_completed'],4)
        self.assertEqual(r['counts']['set_state']['completed'],16)
        self.assertTrue(all(not row['checkpoints'][0]['comparison']['state_exact'] for row in r['rows']))
        self.assertTrue(all(row['checkpoints'][0]['comparison']['state_difference'][0]==.25 for row in r['rows']))

    def test_initial_state_is_not_forced_exact_and_no_extra_setter(self):
        r,store,env=run()
        # Fake native reset honors state0, later archive labels intentionally differ.
        self.assertTrue(all(x['checkpoints'][0]['comparison']['state_exact'] for x in r['rows']))
        self.assertTrue(any(not x['checkpoints'][1]['comparison']['state_exact'] for x in r['rows']))
        self.assertEqual(r['counts']['set_state']['completed'],16)
        self.assertEqual(r['counts']['goal_setter']['completed'],8)
        goals=[e for e in store.events if e['kind']=='goal_setter_return']
        self.assertEqual(len(goals),8)
        self.assertTrue(all(e['before']==e['after'] for e in goals))
        for _,options in env.resets[1::2]:
            self.assertEqual(set(options),{'state','goal_state'})
            self.assertTrue(np.array_equal(options['state'],np.array([100.,100.,150.,150.,0.,0.,0.])))

    def test_no_goal_or_archive_match_selection(self):
        r,_,_=run()
        self.assertEqual([(x['episode_ordinal'],x['lag']) for x in r['rows']],[(1,0),(1,1),(6229,0),(6229,1)])
        self.assertTrue(all(x['status']=='horizon_completed' for x in r['rows']))
        self.assertEqual(m.contract()['comparison'],'observed0 vs archive[0]; after kth actual command vs archive[k], both lags same labels')

    def test_checkpoint_arithmetic_not_uint8_wraparound(self):
        s=np.array([1,2,3,4,5,6,7.],dtype=np.float64);a=s-2
        rgb=np.zeros((224,224,3),np.uint8);ref=rgb.copy();ref[0,0]=[255,1,2]
        c=m.compare(s,rgb,a,ref)
        self.assertEqual(c['state_difference'],[2.]*7)
        self.assertEqual(c['different_pixels'],1);self.assertEqual(c['different_channels'],3)
        self.assertEqual(c['pixel_max_absolute_error'],255)
        self.assertAlmostEqual(c['pixel_MAE'],258/(224*224*3))
        self.assertFalse(c['rgb_exact'])

    def test_returned_early_terminal_preserves_missing_denominator(self):
        env=FakeEnv()
        # Honest early termination at first action with a stable goal and evaluator.
        env.eval_state=lambda goal,state:(True,0.)
        r,store,_=run(env)
        self.assertEqual(r['native_stopped'],4);self.assertEqual(r['horizon_completed'],0)
        self.assertEqual(r['completed_checkpoints'],4)
        self.assertEqual(r['counts']['public_step']['completed'],4)
        self.assertTrue(all(x['missing_checkpoints']==[5,10,15,20] for x in r['rows']))
        self.assertEqual(len([n for n in store.files if n.endswith('.png')]),8)
        self.assertEqual(r['errors'],[])

    def test_actual_action_mutation_retains_return_frame_failclosed(self):
        env=FakeEnv();step=env.step
        def bad(c):
            v=step(c);env.latest_action=np.zeros(2,np.float32);return v
        env.step=bad
        r,store,_=run(env)
        self.assertEqual(r['status'],'engineering_failed')
        self.assertIn('latest_action',r['first_error']['message'])
        self.assertIn('e1_lag0_01.png',store.files)
        self.assertEqual(r['counts']['public_step'],dict(attempted=1,completed=1))
        self.assertEqual(r['rows'][1]['status'],'not_started')

    def test_partial_native_failure_retains_internal_substeps_and_primary(self):
        r,store,env=run(FakeEnv(fail_step=True,close_error=True))
        self.assertEqual(r['status'],'engineering_failed')
        self.assertIn('first fake physics failure',r['first_error']['message'])
        self.assertEqual(r['counts']['public_step'],dict(attempted=1,completed=0))
        self.assertEqual(r['counts']['physics_step']['completed'],14)
        self.assertIn('secondary fake close failure',str(r['errors']))
        self.assertTrue(any(e['kind']=='operation_failed' and e['resource']=='public_step' for e in store.events))
        self.assertNotIn('reset',env.__dict__);self.assertNotIn('_set_goal_state',env.__dict__)

    def test_failed_journal_does_not_overwrite_native_primary(self):
        store=MemoryStore();emit=store.emit
        def bad(kind,**fields):
            if kind=='operation_failed':raise OSError('secondary failed-event write')
            return emit(kind,**fields)
        store.emit=bad
        r,_,_=run(FakeEnv(fail_step=True),store=store)
        self.assertIn('first fake physics failure',r['first_error']['message'])
        self.assertIn('secondary failed-event write',str(r['errors']))

    def test_final_report_failure_not_processed(self):
        store=MemoryStore();put=store.put
        def bad(name,value,final=False):
            if final:raise OSError('report disk failure')
            return put(name,value,final)
        store.put=bad
        r,_,_=run(store=store)
        self.assertFalse(r['run_processed']);self.assertEqual(r['status'],'engineering_failed')

    def test_hashchain_and_raw_return_bits(self):
        r,store,_=run();previous=None
        for i,e in enumerate(store.events):
            self.assertEqual((e['index'],e['previous_sha256']),(i,previous))
            self.assertEqual(e['event_sha256'],m.digest({k:v for k,v in e.items() if k!='event_sha256'}))
            previous=e['event_sha256']
        self.assertEqual(previous,r['journal_chain_tail'])
        returns=[e for e in store.events if e['kind']=='native_return']
        self.assertEqual(len(returns),80)
        self.assertTrue(all(e['actual_raw_bits']==e['requested_float32_bits'] for e in returns))

    def test_tampered_contract_or_source_rejected(self):
        data,payload=saved()
        with patch.object(m,'read_archive',return_value=(data,payload)):
            for field in ('contract','archive','bindings'):
                p=m.make_plan(data)
                if field=='contract':p['contract']['caps']['goal_setter']=9
                elif field=='archive':p['archive']['episodes'][0]['offset']+=1
                else:p['bindings'][str(m.ENV)]='0'*64
                with self.subTest(field=field),self.assertRaises(ValueError):m.verify(p)

    def test_no_authorization_no_factory(self):
        with self.assertRaisesRegex(ValueError,'authorization'):
            m.execute({},MemoryStore(),lambda:self.fail('factory called'),authorization='no')

    def test_caps_and_wall_predispatch(self):
        p=m.contract();p['caps']['public_step']=0;meter=m.Meter(p,MemoryStore())
        with self.assertRaises(ValueError):meter.call('public_step',lambda:self.fail('dispatched'))
        meter=m.Meter(m.contract(),MemoryStore());meter.started-=301
        with self.assertRaises(ValueError):meter.call('reset',lambda:self.fail('dispatched'))

    def test_z_no_neural_simulator_imports(self):
        self.assertFalse(any(k in sys.modules for k in ('torch','gymnasium','pymunk','stable_worldmodel')))


if __name__=='__main__':unittest.main()
