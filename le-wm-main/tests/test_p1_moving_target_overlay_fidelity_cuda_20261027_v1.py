"""RAM array contracts only; no torch, simulator, research input or tmp directory."""
from copy import deepcopy
from io import BytesIO
import gzip
import zlib
from types import SimpleNamespace as NS
import numpy as np
import pytest
from experiments.spectral_reachability import p1_moving_target_overlay_fidelity_cuda_20261027_v1 as m


class RAMStore(m.Store):
    def __init__(self):
        self.files={};self.comp=zlib.compressobj(6,zlib.DEFLATED,31);self.closed=self.broken=False
        self._write('events.jsonl.gz',b'')
    def _write(self,name,raw,append=False,final=False):
        assert append or name not in self.files
        assert sum(map(len,self.files.values()))+len(raw)<=m.native.CAP-(0 if final else m.native.FINAL)
        self.files[name]=self.files.get(name,b'')+raw


def pixels(value):return np.full((224,224,3),value,np.uint8)


def roots():
    result=[]
    for slot in (0,1):
        off=dict(rgb_history=np.stack([pixels(slot+v) for v in (0,5,10)]),raw_action_history=m.moving.fixed_raw(),goal_rgb=pixels(70+slot))
        on=dict(rgb_history=np.stack([pixels(slot+v) for v in (30,35,40)]),raw_action_history=m.moving.fixed_raw(),goal_rgb=pixels(120+slot))
        root=dict(slot=slot,root_id=m.digest(np.array([slot,1],np.int32)),original_task_id=m.digest(np.array([slot,2],np.int32)),producer_report_sha256='a'*64,
            original_off_packet=deepcopy(off),packets=dict(off=off,on=on),plans_report_sha256='b'*64,
            q=np.eye(192,dtype=np.float32),goal_render_roles=dict(off='original_source_goal_bit_exact',on='same_body_true_goal_render'))
        def frame(c,raw,step,role,selection=None,arm=0):
            f=dict(slot=slot,root_id=root['root_id'],condition=c,step=step,commands_sha256=m.digest(raw),
                source_report_sha256='a'*64 if selection is None else 'b'*64,role=role,
                pixels=pixels(slot+step+arm+50+(30 if c=='on' else 0)))
            if selection is not None:f['selection_sha256']=selection
            return f
        raw=np.full((10,2),.03,np.float32)
        root['witness']=dict(commands=raw,labels={c:[frame(c,raw,s,'actual_witness_successor') for s in (5,10)] for c in m.CONDITIONS})
        plans=[]
        for i,a in enumerate(m.ARMS):
            raw=np.full((10,2),.05+.01*i,np.float32);selection=m.digest(np.array([slot,i],np.int32))
            plans.append(dict(arm=a,original_task_id=root['original_task_id'],commands=raw,selection_sha256=selection,planner_report_sha256='c'*64,
                labels={c:{role:frame(c,raw,step,'actual_fixed_plan_'+role,selection,i) for role,step in (('prefix',5),('terminal',10))} for c in m.CONDITIONS}))
        root['fixed_plans']=plans;result.append(root)
    return result


def load(store,entry):
    value=store.files[entry['file']];assert m.enc.digest(value)==entry['sha256'];raw=gzip.decompress(value)
    assert m.enc.digest(raw)==entry['npy_sha256'];a=np.load(BytesIO(raw),allow_pickle=False)
    assert m.digest(a)==entry['raw_sha256'];return a


def test_budget_fixed_new_moving_not_zero_or_archive():
    a,b=[m.contract(p) for p in m.PHASES]
    assert a['counts']==dict(encoder=20,projector=20,predict=16,predictor=16,action_encoder=16,pred_proj=16)
    assert b['counts']==dict(encoder=32,projector=32,predict=24,predictor=24,action_encoder=24,pred_proj=24)
    assert a['P1_admission'] is b['P1_admission'] is False
    assert set(m.PACKET_KEYS)=={'rgb_history','raw_action_history','goal_rgb'}


@pytest.mark.parametrize('bad',['state','witness','future','goal_latent','Q','identity','rawzero','rawdivide100','float64','pixelsize','changedoldoff'])
def test_strict_blind_packet_and_original_off_binding(bad):
    data=roots();p=data[0]['packets']['off']
    if bad=='rawzero':p['raw_action_history'][:]=0
    elif bad=='rawdivide100':p['raw_action_history']/=100
    elif bad=='float64':p['raw_action_history']=p['raw_action_history'].astype(float)
    elif bad=='pixelsize':p['rgb_history']=np.zeros((3,32,32,3),np.uint8)
    elif bad=='changedoldoff':p['goal_rgb'][0,0,0]+=1
    else:p[bad]='forbidden'
    with pytest.raises(ValueError):m.validate_roots(data)


@pytest.mark.parametrize('phase',m.PHASES)
def test_missing_actual_labels_rejected_before_any_encode_not_goal_substitute(phase):
    data=roots();key='witness' if phase==m.PHASES[0] else 'fixed_plans';data[0].pop(key)
    gate=m.prerequisites(data,phase);assert not gate['ready'] and gate['missing'] and not gate['P1_admission']
    with pytest.raises(ValueError,match='before any encoding'):m.census(data,phase)


@pytest.mark.parametrize('bad',['condition','root_id','slot','step','commands_sha256','source_report_sha256','selection_sha256','role'])
def test_real_labels_strict_identity_join(bad):
    data=roots();f=data[0]['fixed_plans'][0]['labels']['off']['prefix']
    f[bad]=7 if bad in ('slot','step') else 'changed'
    with pytest.raises(ValueError):m.prerequisites(data,m.PHASES[1])


def run_mock(phase,fail=None):
    data=roots();_,images,rows=m.census(data,phase);store=RAMStore();requests=[];encoded=[]
    def encode(rgb,meta,index):
        encoded.append((meta,index))
        if fail==('image',index):raise RuntimeError('encoding failed')
        return np.full(192,float(rgb[0,0,0]),np.float32)
    def predict(r,index):
        requests.append(deepcopy(r));assert set(r)==m.REQUEST_KEYS
        if fail==('predict',index):raise RuntimeError('prediction failed')
        value=r['latent_history'].copy();value[-1]+=np.float32(3)
        return value
    try:out=m.evaluate(data,phase,encode,predict,store,images,rows)
    except RuntimeError:out=None
    return data,images,rows,store,requests,encoded,out


@pytest.mark.parametrize('phase',m.PHASES)
def test_factual_and_fixed_old_plans_same_condition_chronology(phase):
    data,images,rows,s,requests,encoded,out=run_mock(phase);counts=m.contract(phase)['counts']
    assert len(encoded)==counts['encoder'] and len(requests)==counts['predict']
    assert all(i['status']=='encoded' for i in images) and all(r['status']=='saved' for r in rows)
    assert len(out['scores'])==counts['predict'] and not out['P1_admission'] and not out['selection']
    for r in requests:
        slot,c=r['slot'],r['condition'];offset=30 if c=='on' else 0
        if r['kind'] in ('H1','fresh_H2_first','fixed_H2_first'):
            assert np.array_equal(r['latent_history'],np.stack([np.full(192,slot+offset+t,np.float32) for t in (0,5,10)]))
            assert m.native.bits(r['raw_packets'][:2].reshape(10,2),m.moving.fixed_raw())
        else:
            assert np.array_equal(r['latent_history'][0],np.full(192,slot+offset+5,np.float32))
            assert np.array_equal(r['latent_history'][1],np.full(192,slot+offset+10,np.float32))
            if r['kind'] in ('fresh_H2_terminal','fixed_H2_terminal'):
                assert np.array_equal(r['latent_history'][2],np.full(192,slot+offset+13,np.float32))
            else:
                arm=0 if r['arm']=='source_factual_witness' else m.ARMS.index(r['arm'])
                assert np.array_equal(r['latent_history'][2],np.full(192,slot+offset+55+arm,np.float32))
            assert m.native.bits(r['raw_packets'][0],m.moving.fixed_raw()[5:])
        assert m.native.bits(r['tokens'],m.native.fidelity.normalized(r['raw_packets']))
    for row,r in zip(rows,requests,strict=True):
        assert m.native.bits(load(s,row['inputs']['latent_history']),r['latent_history'])
        assert m.native.bits(load(s,row['inputs']['raw_packets']),r['raw_packets'])
    if phase==m.PHASES[0]:
        assert not any(meta['role']=='goal' for meta,index in encoded) and not out['goal_scores']
    else:
        assert sum(meta['role']=='goal' for meta,index in encoded)==4
        assert len(out['goal_scores'])==16 and len(out['rankings'])==8
        for score in out['goal_scores']:
            slot,c,kind,arm=score['slot'],score['condition'],score['kind'],m.ARMS.index(score['arm'])
            goal=70+slot if c=='off' else 120+slot;offset=30 if c=='on' else 0
            predicted=slot+offset+(16 if kind=='fixed_H2_terminal' else 58+arm)
            actual=slot+offset+60+arm
            assert score['predicted_Q']==float((predicted-goal)**2)
            assert score['actual_Q']==float((actual-goal)**2)
        # Equal fixed-H2 predictions remain ties; do not arbitrarily name a winner.
        assert all(r['predicted_sign']==0 for r in out['rankings'] if r['kind']=='fixed_H2_terminal')
    assert sum(map(len,s.files.values()))<m.native.CAP


@pytest.mark.parametrize('phase',m.PHASES)
@pytest.mark.parametrize('fail',[('image',0),('image',19),('predict',0),('predict',13)])
def test_failed_partial_full_denominator_no_retry_or_padding(phase,fail):
    data,images,rows,s,requests,encoded,out=run_mock(phase,fail)
    assert out is None and len(images)==m.contract(phase)['counts']['encoder'] and len(rows)==m.contract(phase)['counts']['predict']
    kind,k=fail
    if kind=='image':
        assert len(encoded)==k+1 and not requests and images[k]['status']=='attempted' and 'pixels' in images[k]
        assert all(row['status']=='not_started' for row in rows)
    else:
        assert len(requests)==k+1 and rows[k]['status']=='attempted' and len(rows[k]['inputs'])==3
        assert all(row['status']=='not_started' for row in rows[k+1:])


def test_new_goal_fit_and_alternate_projection_rejected():
    data=roots();data[0]['q'][0,0]=2
    with pytest.raises(ValueError,match='identity192'):m.prerequisites(data,m.PHASES[1])
    data=roots();data[0]['goal_render_roles']['on']='goal_from_privileged_state'
    with pytest.raises(ValueError,match='same-body'):m.prerequisites(data,m.PHASES[1])


def test_cuda_callbacks_reject_cpu_before_any_model_or_tensor_call():
    with pytest.raises(ValueError,match='HOST CUDA only'):m.CudaCallbacks(NS(),NS(),NS(type='cpu'),RAMStore(),NS())


def source_fixture(triplets=(0,5,10,15,20)):
    """Literal producer v1 schema, RAM receipts only, no simulated dynamics."""
    name='p1_moving_target_overlay_source_20261027_v1';triplets=list(triplets)
    caps=dict(reset=12,public_step=120,physics_step=1224,set_state=24,goal_render=12,goal_setter=16,goal_bind=4,explicit_render=6*(21+2*len(triplets)))
    ct=dict(source_triplets=triplets,flags=[False,True,False],slots=2,warmup=10,horizon=10,caps=caps)
    p=dict(schema=name+'.prepare',contract=ct)
    r=dict(schema=name+'.report',status='paired_moving_overlay_recorded',errors=[],first_error=None,source_verified=True,
        full_protocol=True,contract=ct,prepare_id=m.enc.core.digest(p),rows=[],blind_exports=[],counts={k:dict(attempted=v,completed=v) for k,v in caps.items()})
    events=[dict(kind='run_started',prepare_id=r['prepare_id'],contract=ct)]
    data=dict(refs=[],references={});commits={};pictures={};packets={};interfaces=[]
    def picture(s,t,i):return pixels(s+t+(30 if i==1 else 0))
    for s in (0,1):
        rootsha=m.digest(np.array([s,7],np.int32));tasksha=m.digest(np.array([s,8],np.int32))
        data['refs'].append(dict(root_sha256=rootsha,history=np.stack([picture(s,t,0) for t in (0,5,10)]),goal_rgb=picture(s,20,0)))
        interfaces.append(dict(task_id=tasksha,q=np.eye(192,dtype=np.float32)))
        raw=np.full((10,2),.03+.01*s,np.float32)
        for ph in ('construction','replay0','replay1'):
            key=f's{s}_{ph}';data['references'][key]={};frames=[]
            for t in range(21):
                snap=dict(state=[s,t],body_velocity=[0,0],angular_velocity=0)
                off=picture(s,t,0);off_file=m.digest(off)  # fake PNG hash only in RAM boundary
                data['references'][key][str(t)]=dict(file_sha256=off_file,pixel_sha256=m.digest(off),snapshot=snap)
                invariant=dict(snapshot=snap,goal_state=[s,20],goal_pose=[10,20,30],goal_rgb_sha256='f'*64,renderer={'variation':1})
                for i in (range(3) if t in triplets else range(1)):
                    a=picture(s,t,i);png=f'{key}_{t}_{i}.png';pictures[png]=a
                    f=dict(reference_key=str(t),overlay_index=i,with_target=i==1,png=png,file_sha256=m.digest(a),pixel_sha256=m.digest(a),
                        pre=deepcopy(invariant),post=deepcopy(invariant),reference_error=None,
                        old_off_match=None if i==1 else dict(file=True,pixels=True,state=True,snapshot=True),off_replay_match=True if i==2 else None)
                    frames.append({k:f[k] for k in ('reference_key','overlay_index','png','file_sha256','pixel_sha256')})
                    events.append(dict(kind='fresh_overlay_frame',key=key,source_slot=s,receipt=f))
            commits[key]=dict(schema=name+'.witness_commit',commands=raw.tolist(),root_snapshot=data['references'][key]['10']['snapshot'])
            for phase,actions in (('warmup',m.moving.fixed_raw()),('witness',raw)):
                for t,a in enumerate(actions,1):events.append(dict(kind='native_return',key=key,source_slot=s,phase=phase,step=t,
                    requested_raw=a.tolist(),actual_raw=a.tolist(),requested_bits=a.tobytes().hex(),actual_bits=a.tobytes().hex(),actual_dtype=a.dtype.str))
            r['rows'].append(dict(key=key,source_slot=s,kind='source',phase=ph,status='source_episode_completed',public_returned=20,
                root_goal_sha256=rootsha,native_success=ph!='construction',frames=frames,missing_triplets=[]))
        artifacts={}
        for c,i in (('off',0),('on',1)):
            pk=dict(rgb_history=np.stack([picture(s,t,i) for t in (0,5,10)]),raw_action_history=m.moving.fixed_raw(),goal_rgb=picture(s,20,i))
            path=f's{s}_{c}_blind_inputs.npz';packets[path]=pk
            artifacts[c]=dict(path=path,sha256='d'*64,fields=['rgb_history','raw_action_history','goal_rgb'],
                rgb_history_pixel_sha256=[m.digest(a) for a in pk['rgb_history']],goal_pixel_sha256=m.digest(pk['goal_rgb']),raw_action_sha256=m.digest(m.moving.fixed_raw()))
        r['blind_exports'].append(dict(source_slot=s,old_root_sha256=rootsha,artifacts=artifacts))
    events.append(dict(kind='run_finished',status=r['status'],counts=deepcopy(r['counts']),errors=[]))
    return p,r,events,data,pictures,packets,commits,interfaces


def assemble_fixture(f):
    p,r,events,data,pictures,packets,commits,interfaces=f
    return m.assemble_source(p,r,events,data,lambda f:pictures[f['png']],lambda f:packets[f['path']],lambda k:commits[k],interfaces,'a'*64)


def test_literal_source186_all_receipts_blind_goal_actual15_same_condition():
    f=source_fixture();data=assemble_fixture(f)
    assert m.prerequisites(data,m.PHASES[0])['ready']
    assert sum(len(row['frames']) for row in f[1]['rows'])==186
    for root in data:
        s=root['slot']
        for c,offset in (('off',0),('on',30)):
            assert root['packets'][c]['goal_rgb'][0,0,0]==s+20+offset
            a,b=root['witness']['labels'][c]
            assert (a['step'],b['step'])==(5,10)
            assert a['pixels'][0,0,0]==s+15+offset and b['pixels'][0,0,0]==s+20+offset
    assert not m.prerequisites(data,m.PHASES[1])['ready']  # Jason is not required for factual.


def test_older_four_triplet_source_has_no_fabricated_on15_or_goal_substitution():
    f=source_fixture((0,5,10,20));data=assemble_fixture(f)
    assert all('witness' not in root for root in data)
    assert not m.prerequisites(data,m.PHASES[0])['ready']
    with pytest.raises(ValueError,match='before any encoding'):m.census(data,m.PHASES[0])


@pytest.mark.parametrize('bad',['partial','goal','history','raw','snapshot','offreplay','missing15','actualbits','root','commit','budget','prepareid'])
def test_source_adapter_rejects_forged_or_cross_condition_acquisition(bad):
    f=source_fixture();p,r,events,data,pictures,packets,commits,interfaces=f
    first=next(e['receipt'] for e in events if e['kind']=='fresh_overlay_frame' and e['receipt']['overlay_index']==2)
    if bad=='partial':r['status']='partial_native_stop'
    elif bad=='goal':packets['s0_on_blind_inputs.npz']['goal_rgb']=pixels(20)
    elif bad=='history':packets['s0_on_blind_inputs.npz']['rgb_history']=packets['s0_off_blind_inputs.npz']['rgb_history']
    elif bad=='raw':packets['s0_off_blind_inputs.npz']['raw_action_history']/=100
    elif bad=='snapshot':first['post']['snapshot']['state'][0]=99
    elif bad=='offreplay':first['off_replay_match']=False
    elif bad=='missing15':events[:]=[e for e in events if not(e['kind']=='fresh_overlay_frame' and e.get('key')=='s0_replay0' and e['receipt']['reference_key']=='15')]
    elif bad=='actualbits':next(e for e in events if e['kind']=='native_return')['actual_bits']='00'*8
    elif bad=='root':r['blind_exports'][0]['old_root_sha256']='e'*64
    elif bad=='commit':commits['s0_replay0']['commands'][0][0]=.7
    elif bad=='budget':r['counts']['explicit_render']['completed']=174
    elif bad=='prepareid':r['prepare_id']='e'*64
    with pytest.raises(ValueError):assemble_fixture(f)


def test_receipt_three_hashes_recover_original_precision_without_tmp():
    store=RAMStore();a=np.random.default_rng(37).normal(size=(197,192)).astype(np.float32)
    entry=m.save_array(store,'full_vit_patch_output',a)
    assert set(entry)=={'file','sha256','npy_sha256','raw_sha256','shape','dtype'}
    assert len({entry[k] for k in ('sha256','npy_sha256','raw_sha256')})==3
    assert m.native.bits(load(store,entry),a)


def test_actual_source_digest_matches_native_chain_not_new_prepare_canonical():
    from experiments.spectral_reachability import p1_conditional_procedural_source_20261027_v1 as source
    p=dict(prepare_id='f'*64,contract=dict(checkpoints=[0,5,10,15,20]))
    assert m.enc.core.digest(p)==source.digest(p)


def test_cli_definitions_are_complete_before_main_and_do_not_execute_on_import():
    import ast
    from pathlib import Path
    nodes=ast.parse(Path(m.__file__).read_text()).body
    assert isinstance(nodes[-1],ast.If) and '__name__' in ast.unparse(nodes[-1].test)
    assert callable(m.material) and callable(m.prepare) and callable(m.verify) and callable(m.run)


def plans_fixture():
    source=source_fixture();roots=assemble_fixture(source)
    name='p1_moving_target_overlay_plans_physical_20261027_v1';plans=[];rows=[];pictures={};saved=[];body=[]
    for b,(s,arm) in enumerate((s,a) for s in (0,1) for a in m.ARMS):
        raw=np.full((10,2),.04+.01*b,np.float32);selection=m.digest(np.array([s,b],np.int32))
        plan=dict(root_slot=s,arm=arm,commands=raw.tolist(),selection_digest=selection,compiler_report_SHA='c'*64);plans.append(plan)
        where=dict(branch=b,root_slot=s,arm=arm,selection_digest=selection)
        body.append(dict(kind='branch_started',plan=plan,**where));refs={};observations={};checkpoints=[]
        for phase in ('warmup','plan'):
            for t in range(11):
                modes=('off','on','off_after') if phase=='plan' and t in (0,5,10) else ('off',)
                off_value=s+t if phase=='warmup' else s+10+t+(0 if t==0 else b%2)
                snap=dict(state=[s,t],body_velocity=[0,0]);invariant=dict(snapshot=snap,goal_rgb_sha256='f'*64,renderer=dict(setting=1))
                old=dict(file_sha256=m.digest(pixels(off_value)),pixel_sha256=m.digest(pixels(off_value)),snapshot=snap)
                refs[phase,t]=old
                for mode in modes:
                    a=pixels(off_value+(30 if mode=='on' else 0));png=f'b{b}_{phase}_{t}_{mode}.png';pictures[png]=a
                    observation=dict(phase=phase,step=t,mode=mode,png=png,file_sha256=m.digest(a),pixel_sha256=m.digest(a),
                        before=deepcopy(invariant),after=deepcopy(invariant),snapshot=deepcopy(snap),with_target=mode=='on')
                    # Literal producer chronology: observation has NO old_off_match.
                    body.append(dict(kind='fresh_observation',receipt=deepcopy(observation),**where))
                    matches=None if mode=='on' else dict(file=True,pixels=True,state=True,snapshot=True,observation_state=True)
                    body.append(dict(kind='guard_receipt',phase=phase,step=t,mode=mode,png=png,accepted=True,
                        render_preserved_invariant=True,canonical_agent_pixels=7,old_off_match=matches,**where))
                    observations[phase,t,mode]=dict(**observation,old_off_match=matches)
                if phase=='plan' and t in (0,5,10):
                    checkpoints.append(dict(step=t,complete=True,off_before=observations[phase,t,'off'],on=observations[phase,t,'on'],off_after=observations[phase,t,'off_after']))
        for phase,actions in (('warmup',m.moving.fixed_raw()),('plan',raw)):
            for t,a in enumerate(actions,1):body.append(dict(kind='returned_transition',phase=phase,step=t,command=a.tolist(),actual_command=a.tolist(),actual_float32_bits=a.tobytes().hex(),**where))
        rows.append(dict(root_slot=s,arm=arm,selection_digest=selection,initial_exact=True,full_horizon=True,first_error=None,secondary_errors=[],
            own_attempted=10,own_returned=10,warmup_returned=10,missing_checkpoints=[],status='horizon_completed',checkpoints=checkpoints))
        saved.append(dict(off=refs))
    p=dict(schema=name+'.prepare',contract={},source=dict(plans=plans,original_report_sha256='d'*64))
    caps=dict(reset=8,public_step=80,physics_step=816,explicit_render=112,goal_render=8,set_state=16,goal_bind=4)
    r=dict(schema=name+'.report',status='moving_target_overlay_plans_recorded',prepare_id=m.enc.core.digest(p),contract=p['contract'],errors=[],
        inputs_final_verified=True,full_horizon_count=4,complete_triplets=12,counts={k:dict(attempted=v,completed=v) for k,v in caps.items()},rows=rows)
    events=[dict(kind='run_started',prepare=p)]+body+[dict(kind='run_finished',counts=deepcopy(r['counts']),errors=[])]
    return roots,p,r,events,saved,pictures


def assemble_plan_fixture(f):
    roots,p,r,events,saved,pictures=f
    return m.assemble_plans(roots,p,r,events,saved,lambda f:pictures[f['png']],'b'*64)


def test_jason_observation_before_guard_binding_has_no_old_off_key():
    f=plans_fixture();assert all('old_off_match' not in e['receipt'] for e in f[3] if e['kind']=='fresh_observation')
    roots=assemble_plan_fixture(f)
    assert m.prerequisites(roots,m.PHASES[1])['ready']
    assert len(m.image_jobs(roots,m.PHASES[1]))==32
    for root in roots:
        assert [x['arm'] for x in root['fixed_plans']]==list(m.ARMS)
        assert all(x['planner_report_sha256']=='c'*64 for x in root['fixed_plans'])


@pytest.mark.parametrize('bad',['guard_png','guard_rejected','guard_match','guard_duplicate','cp_shared','cp_match','oldfile','rootcontext','actualbits'])
def test_jason_join_rejects_actual_contract_mismatch(bad):
    f=plans_fixture();roots,p,r,events,saved,pictures=f
    guard=next(e for e in events if e['kind']=='guard_receipt' and e['phase']=='plan')
    if bad=='guard_png':guard['png']='someone_else.png'
    elif bad=='guard_rejected':guard['accepted']=False
    elif bad=='guard_match':guard['old_off_match']['pixels']=False
    elif bad=='guard_duplicate':events.append(deepcopy(guard))
    elif bad=='cp_shared':r['rows'][0]['checkpoints'][0]['on']['file_sha256']='e'*64
    elif bad=='cp_match':r['rows'][0]['checkpoints'][0]['on']['old_off_match']={}
    elif bad=='oldfile':saved[0]['off']['plan',5]['file_sha256']='e'*64
    elif bad=='rootcontext':roots[0]['packets']['on']['rgb_history'][-1]=pixels(99)
    elif bad=='actualbits':next(e for e in events if e['kind']=='returned_transition')['actual_float32_bits']='00'*8
    with pytest.raises(ValueError):assemble_plan_fixture(f)


def test_material_and_verify_end_to_end_ram_no_files_no_neural(monkeypatch):
    """Exercise actual material CLI boundary against literal saved schemas in RAM."""
    from pathlib import Path
    import json
    from experiments.spectral_reachability import p1_moving_target_overlay_source_20261027_v1 as producer
    p,r,events,data,pictures,packets,commits,interfaces=source_fixture()
    p['bindings']={};r['prepare_id']=m.enc.core.digest(p);events[0]['prepare_id']=r['prepare_id']
    previous=None
    for i,e in enumerate(events):
        e.update(index=i,previous_sha256=previous);e['event_sha256']=m.enc.core.digest(e);previous=e['event_sha256']
    r['journal_chain_tail']=previous
    report='/ram/source/report.json';prepared='/ram/source/prepare.json';binding='/ram/source/execution_binding.json'
    metadata={report:r,binding:dict(prepared=prepared,prepare_file_sha256='e'*64)}
    metadata.update({f'/ram/source/{k}_commands.json':v for k,v in commits.items()})
    bodies={'/ram/source/journal.jsonl':b''.join((json.dumps(e)+'\n').encode() for e in events)}
    seals={report:'d'*64,prepared:'e'*64}
    for file,a in pictures.items():seals['/ram/source/'+file]=m.digest(a)
    for file in packets:seals['/ram/source/'+file]='d'*64
    actual_sha=m.native.sha
    monkeypatch.setattr(m.native,'sha',lambda path:seals.get(str(path),'f'*64) if str(path).startswith('/ram/') else actual_sha(path))
    original_stat=Path.stat;original_file=Path.is_file;original_sym=Path.is_symlink;original_read=Path.read_bytes
    monkeypatch.setattr(Path,'stat',lambda path,*a,**kw:NS(st_size=len(bodies.get(str(path),b''))) if str(path).startswith('/ram/') else original_stat(path,*a,**kw))
    monkeypatch.setattr(Path,'is_file',lambda path:True if str(path).startswith('/ram/') else original_file(path))
    monkeypatch.setattr(Path,'is_symlink',lambda path:False if str(path).startswith('/ram/') else original_sym(path))
    monkeypatch.setattr(Path,'read_bytes',lambda path:bodies[str(path)] if str(path).startswith('/ram/') else original_read(path))
    def read(path,expected=None,cap=None):
        if expected is not None:assert m.native.sha(path)==expected
        return deepcopy(metadata[str(path)])
    monkeypatch.setattr(m.native,'read',read)
    monkeypatch.setattr(producer,'load_prepared',lambda path,expected:deepcopy(p))
    monkeypatch.setattr(producer,'verify',lambda value:data)
    monkeypatch.setattr(producer,'old_image',lambda value:pictures[Path(value['path']).name].copy())
    monkeypatch.setattr(producer,'files',lambda:[])
    monkeypatch.setattr(m.moving.moving,'load_blind',lambda path,expected:deepcopy(packets[Path(path).name]))
    monkeypatch.setattr(m.moving,'load_export',lambda path,h:interfaces[m.moving.EXPORTS.index(str(path))])
    def export(path,cap):
        s=m.moving.EXPORTS.index(str(path));off=packets[f's{s}_off_blind_inputs.npz']
        return dict(source_slot=s,blind_sha256=data['refs'][s]['blind_sha256'],task_contract=dict(source_slot=s,
            pixel_sha256=[m.digest(a) for a in (*off['rgb_history'],off['goal_rgb'])],raw_action_sha256=m.digest(off['raw_action_history'])))
    monkeypatch.setattr(m.moving,'read',export)
    for s in (0,1):data['refs'][s]['blind_sha256']=str(s+1)*64
    seals.update(dict(zip(m.moving.EXPORTS,m.moving.EXPORT_SHAS)))
    # Fixed exports are virtual too, even though their mandated paths are under results.
    old_sealed=m.native.sha
    monkeypatch.setattr(m.native,'sha',lambda path:seals[str(path)] if str(path) in seals else old_sealed(path))
    old_file=Path.is_file;old_stat=Path.stat;old_sym=Path.is_symlink
    monkeypatch.setattr(Path,'is_file',lambda path:True if str(path) in seals else old_file(path))
    monkeypatch.setattr(Path,'is_symlink',lambda path:False if str(path) in seals else old_sym(path))
    monkeypatch.setattr(Path,'stat',lambda path,*a,**kw:NS(st_size=100) if str(path) in seals else old_stat(path,*a,**kw))
    monkeypatch.setattr(m.native.sealed,'source_files',lambda:[])
    monkeypatch.setattr(m.moving.moving,'source_files',lambda:[])
    monkeypatch.setattr(m.enc,'MODELS',{})
    monkeypatch.setattr(m.enc,'native_metadata',lambda:dict(original_public_normalizer=True))
    args=dict(phase=m.PHASES[0],source_report=report,source_sha='d'*64,
        producer_source_sha=actual_sha(producer.__file__),producer_prepare_sha='e'*64)
    own,roots=m.material(**args)
    assert len(m.canonical(own))<256*1024 and m.prerequisites(roots,m.PHASES[0])['ready']
    metadata['/ram/own_prepare.json']=own;seals['/ram/own_prepare.json']='9'*64
    checked,again=m.verify('/ram/own_prepare.json','9'*64)
    assert checked==own and len(again)==2 and own['contract']['counts']['encoder']==20
    monkeypatch.setattr('sys.argv',[m.NAME,'verify','--prepared','/ram/own_prepare.json','--prepare-sha','9'*64])
    m.main()  # CLI dispatch reaches the real verify/material/producer adapters.
    from experiments.spectral_reachability import p1_moving_target_overlay_plans_physical_20261027_v1 as plans
    _,pp,rr,es,old,imgs=plans_fixture()
    pp['bindings']={};pp['source'].update(sourcepack_sha256=producer.PINNED[str(producer.SD/'sourcepack.json')],
        source_slots=[dict(root_sha256=root['root_id']) for root in roots])
    for pl in pp['source']['plans']:
        pl.update(compiler_report_SHA='52c475ed499b077f261517bc592252b208926919909674cebf68fa4029cc8310',
            compiler_report=str(m.moving.ROOT/'results/moving_history_planner_handoff_main01.json'),
            compiler_schema='moving_history_planner_handoff.v1')
    rr['prepare_id']=m.enc.core.digest(pp);es[0]['prepare']=pp;previous=None
    for i,e in enumerate(es):
        e.update(index=i,previous_sha256=previous);e['event_sha256']=m.enc.core.digest(e);previous=e['event_sha256']
    rr['journal_chain_tail']=previous
    pr='/ram/plans/report.json';pb='/ram/plans/execution_binding.json';prep='/ram/plans/prepare.json'
    metadata[pr]=rr;metadata[pb]=dict(prepared=prep,prepare_sha256='8'*64);seals[pr]='7'*64;seals[prep]='8'*64
    bodies['/ram/plans/journal.jsonl']=b''.join((json.dumps(e)+'\n').encode() for e in es)
    for file,a in imgs.items():pictures[file]=a;seals['/ram/plans/'+file]=m.digest(a)
    monkeypatch.setattr(plans,'verify',lambda *a:pp)
    monkeypatch.setattr(plans,'inputs',lambda:(pp['source'],None,old))
    args.update(phase=m.PHASES[1],plans_report=pr,plans_sha='7'*64,plans_source_sha=actual_sha(plans.__file__),plans_prepare_sha='8'*64)
    own,joined=m.material(**args)
    assert m.prerequisites(joined,m.PHASES[1])['ready'] and len(m.canonical(own))<256*1024
    assert len(m.image_jobs(joined,m.PHASES[1]))==32 and len(m.census(joined,m.PHASES[1])[2])==24
    pp['source']['plans'][0]['compiler_report_SHA']='cd8432c06288577d25d9d9b92c7eb3673de592499ee64cae93238156f3f19aa0'
    rr['prepare_id']=m.enc.core.digest(pp);es[0]['prepare']=pp;previous=None
    for i,e in enumerate(es):
        e.pop('event_sha256',None)
        e.update(index=i,previous_sha256=previous);e['event_sha256']=m.enc.core.digest(e);previous=e['event_sha256']
    rr['journal_chain_tail']=previous
    bodies['/ram/plans/journal.jsonl']=b''.join((json.dumps(e)+'\n').encode() for e in es)
    with pytest.raises(ValueError,match='sealed command handoff'):
        m.material(**args)


class ArrayTensor:
    """NativeTrace interface stand-in, NOT torch or a CPU neural network."""
    def __init__(self,a):self.a=np.asarray(a,np.float32);self.device=NS(type='cuda')
    def detach(self):return self
    def cpu(self):return self
    def numpy(self):return self.a


@pytest.mark.parametrize('phase',m.PHASES)
def test_full_native_module_inputs_patch_outputs_complete_size_under_8MiB_ram(phase):
    data=roots();store=RAMStore();_,images,rows=m.census(data,phase)
    trace=m.NativeTrace(NS(),store,lambda:None,lambda:True);rng=np.random.default_rng(20261027)
    def call(kind,args,out,kwargs=None):
        entry=trace.begin(kind,tuple(ArrayTensor(a) for a in args),kwargs or {});trace.end(entry,out)
    def encode(rgb,meta,index):
        trace.active=('image',index,None)
        x=(rgb.astype(np.float32)/np.float32(255)).transpose(2,0,1)[None].copy()
        patches=rng.normal(size=(1,197,192)).astype(np.float32);z=patches[:,0].copy()
        call('encoder',[x],dict(last_hidden_state=ArrayTensor(patches),pooler_output=ArrayTensor(patches.mean(1))),dict(interpolate_pos_encoding=True))
        call('projector',[patches[:,0]],ArrayTensor(z));trace.finish('image',index);return z[0].copy()
    def predict(r,index):
        trace.active=('prediction',index,r);z=r['latent_history'][None].copy();tokens=r['tokens'][None].copy()
        ae=np.repeat(tokens.mean(-1,keepdims=True),192,axis=-1).astype(np.float32)
        call('action_encoder',[tokens],ArrayTensor(ae));outer=trace.begin('predict',(ArrayTensor(z),ArrayTensor(ae)),{})
        out=z.copy();out[:,-1]+=np.float32(.125)
        call('predictor',[z,ae],ArrayTensor(out));call('pred_proj',[out.reshape(3,192)],ArrayTensor(out.reshape(3,192)))
        trace.end(outer,ArrayTensor(out));trace.finish('prediction',index);return out[0].copy()
    analysis=m.evaluate(data,phase,encode,predict,store,images,rows);m.full_counts(trace.counts(),m.contract(phase)['counts'])
    store.put('analysis.json',analysis);store.close()
    store.put('report.json',dict(images=images,requests=rows,module_ledger=trace.entries,
        artifact_sha256={k:m.enc.digest(v) for k,v in store.files.items()}),final=True)
    total=sum(map(len,store.files.values()));raw_bytes=sum(len(gzip.decompress(v)) for k,v in store.files.items() if k.endswith('.npy.gz'))
    assert raw_bytes>15*1024**2 and total<m.native.CAP
    for e in trace.entries:
        if e['resource']=='encoder':
            assert load(store,e['inputs'][0]['tensor']).shape==(1,3,224,224)
            assert load(store,e['output']['last_hidden_state']['tensor']).shape==(1,197,192)
            assert load(store,e['output']['pooler_output']['tensor']).shape==(1,192)
    print(phase,'synthetic compressed=',total,'original_NPY=',raw_bytes)


def test_run_verifies_before_creating_output_or_loading_model(monkeypatch):
    def fail(*args):raise ValueError('incomplete actual source')
    monkeypatch.setattr(m,'verify',fail)
    monkeypatch.setattr(m,'Store',lambda *args:pytest.fail('output before input acceptance'))
    with pytest.raises(ValueError,match='incomplete actual source'):m.run('/RAM/p','a'*64,'/RAM/output')
