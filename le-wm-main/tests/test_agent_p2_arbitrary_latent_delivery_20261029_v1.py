"""RAM/temporary saved metadata only. Never instantiate the actual two root optimizer."""
from copy import deepcopy
import io
import json
from pathlib import Path
import stat
import subprocess
import sys

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p2_arbitrary_latent_delivery_20261029_v1 as m


def fixture():
    d = dict(schema='agent_p2_target_only_delivery_20261029_v1.prepare',
             denominator=dict(roots=2,arms=3,total=6,assembled=4,missing=2,dispatched=0),rows=[])
    pm = dict(slots=[]); pr = dict(rows=[]); arrays = {}
    tp = dict(slots=[],expected_state={str(i):'fixture_state' for i in range(425)},
              protocol=dict(spec=dict(objective='identity192_terminal_meanMSE_float64')))
    for i in (0,1):
        z = np.arange(576,dtype=np.float32).reshape(1,3,192)/1024+i
        a = np.arange(20,dtype=np.float32).reshape(1,2,10)/128
        g = np.full((1,192),i+2,np.float32)
        names = dict(z=f'array_{3*i:05d}.npy',a=f'array_{3*i+1:05d}.npy',goal=f'array_{3*i+2:05d}.npy')
        arrays.update({names[k]:v for k,v in (('z',z),('a',a),('goal',g))})
        s = dict(root=i,condition='off',task_id=m.TASK_IDS[i],task_sha256='a'*64,blind_sha256='b'*64,
                 pixel_sha256=['c'*64]*4,z=z[0].tolist(),a=a[0].tolist())
        pm['slots'].append(s)
        pr['rows'].append(dict(root=i,status='returned',routes=dict(hi_whole=dict(z=names['z'],a=names['a'])),
            pixels=[dict(image=j,paths=dict(hi=dict(output=names['goal']))) for j in range(4)]))
        ts = {k:deepcopy(s[k]) for k in ('root','condition','task_id','task_sha256','blind_sha256','pixel_sha256')}
        ts.update(pointers=dict(history=names['z'],actions=names['a'],target=names['goal']),
                  input_sha256={k:m.bh(v.tobytes()) for k,v in (('z',z),('a',a),('target',g))})
        tp['slots'].append(ts)
        for arm in ('prior_FIRST','program_first_postcondition','arbitrary'):
            d['rows'].append(dict(root=i,condition='off',task_id=s['task_id'],task_sha256=s['task_sha256'],
                blind_sha256=s['blind_sha256'],arm=arm,status='missing' if arm=='arbitrary' else 'assembled_not_dispatched',
                history_sha256=m.bh(z.tobytes()),action_history_sha256=m.bh(a.tobytes()),
                controller_payload=None if arm=='arbitrary' else dict(z=z.tolist(),a=a.tolist(),target=np.full((1,192),99,np.float32).tolist())))
    return d,pm,pr,tp,arrays


def resolve(f):
    d,pm,pr,tp,a = f
    return m.resolve_inputs(d,pm,pr,tp,lambda n:a[n].copy())


def task(slot):
    return dict(schema=m.core.NAME+'.task',root_slot=slot['root'],condition='off',context=slot['context'],goal=slot['goal'],
                coordinate_refs={k:dict(path='/RAM/fixture.json',sha256='a'*64) for k in m.core.COMMON})


def saved_fixture(tmp_path,monkeypatch):
    d,pm,pr,tp,arrays = fixture()
    for n,a in arrays.items():
        s=io.BytesIO();np.save(s,a,allow_pickle=False);b=s.getvalue();(tmp_path/n).write_bytes(b)
        pr.setdefault('arrays',{})[n]=dict(data_sha256=m.bh(a.tobytes()),dtype=a.dtype.str,file_bytes=len(b),file_sha256=m.bh(b),shape=list(a.shape))
    def save(n,v):
        p=tmp_path/(n+'.json');p.write_bytes(m.raw(v)+b'\n');return m.ref(p)
    refs = dict(parity_prepare=save('pm',pm))
    pr.update(manifest=refs['parity_prepare']['path'],manifest_sha256=refs['parity_prepare']['sha256'])
    refs['parity_report']=save('pr',pr)
    pc=dict(report_sha256=refs['parity_report']['sha256'],all_values_exact=True)
    refs['parity_receipt']=save('pc',pc);tp['upstream_receipt']=pc
    refs['tracker_prepare']=save('tp',tp);refs['tracker_report']=save('tr',dict(fixture=True))
    tc=dict(status='all_known2_normalized_tracker_saved_verified',manifest_sha256=refs['tracker_prepare']['sha256'],report_sha256=refs['tracker_report']['sha256'])
    refs['tracker_receipt']=save('tc',tc)
    d.update(authorities=refs,full_authority_receipts=dict(parity_receipt=pc,tracker_receipt=tc),
             sources={r['path']:r['sha256'] for r in refs.values()},common_controller=dict(dispatch=False))
    dr=save('delivery',d);monkeypatch.setattr(m,'DELIVERY_SHA',dr['sha256'])
    return dr


def prepared_fixture(tmp_path,monkeypatch):
    r=saved_fixture(tmp_path,monkeypatch);p=tmp_path/'prepared.json'
    pr=m.prepare(p,r['path'],r['sha256']);return p,pr['sha256']


def run_fixture(tmp_path,monkeypatch):
    p,h=prepared_fixture(tmp_path,monkeypatch);out=tmp_path/'output'
    rr=m.run(p,h,out,'MAIN_NUMERICAL_ONLY')
    return p,h,out,rr['sha256']


def bind_completion_to_report(out):
    c=json.loads((out/'completion.json').read_bytes());c['report_sha256']=m.sha(out/'report.json')
    (out/'completion.json').write_bytes(m.raw(c)+b'\n')


def test_resolve_true_goal_not_prior_or_program_and_preserve_missing():
    f=fixture();original=deepcopy(f[0]);slots=resolve(f)
    assert len(slots)==2 and f[0]==original
    for i,s in enumerate(slots):
        assert s['goal']==[i+2.]*192
        assert s['goal'] != f[0]['rows'][3*i]['controller_payload']['target'][0]
        assert s['goal_pointer']==f'/rows/{i}/pixels/3/paths/hi/output'
        assert s['context']['latent_history']==s['z'][0]


@pytest.mark.parametrize('fault',['root','order','missing','payload_extra','history','past_action','goal_pointer','goalbytes',
    'shape','bool','nonfinite','notf32','goalimage','q','state','task','condition'])
def test_input_join_rejects_fault(fault):
    f=fixture();d,pm,pr,tp,a=f
    if fault=='root':pm['slots'][0]['root']=True
    elif fault=='order':d['rows'].reverse()
    elif fault=='missing':d['rows'][2]['controller_payload']={}
    elif fault=='payload_extra':d['rows'][0]['controller_payload']['witness']=[1]
    elif fault=='history':d['rows'][1]['controller_payload']['z'][0][0][0]+=1
    elif fault=='past_action':d['rows'][1]['controller_payload']['a'][0][0][0]+=1
    elif fault=='goal_pointer':tp['slots'][0]['pointers']['target']='prior_target.npy'
    elif fault=='goalbytes':a['array_00002.npy'][0,0]+=1
    elif fault=='shape':d['rows'][0]['controller_payload']['z']=[]
    elif fault=='bool':d['rows'][0]['controller_payload']['z'][0][0][0]=False
    elif fault=='nonfinite':d['rows'][0]['controller_payload']['a'][0][0][0]=float('nan')
    elif fault=='notf32':d['rows'][0]['controller_payload']['z'][0][0][0]=.1
    elif fault=='goalimage':pr['rows'][0]['pixels'][3]['image']=2
    elif fault=='q':tp['protocol']['spec']['objective']='fitted_metric'
    elif fault=='state':tp['expected_state'].pop('0')
    elif fault=='task':pm['slots'][0]['task_id']='a'*64
    else:tp['slots'][0]['condition']='on'
    with pytest.raises(ValueError):resolve(f)


def test_full64_matches_frozen_core_exactly_and_first_transport(tmp_path):
    s=resolve(fixture())[0];t=task(s);st=m.Store(tmp_path/'ops')
    r=m.numerical(t,st);old=m.core.optimize(t)
    assert r['history']==old['history'] and r['delivered_targets']==old['delivered_targets']
    assert r['resources']==dict(loss_attempted=65,loss_returned=65,gradient_attempted=64,gradient_returned=64,update_attempted=64,update_returned=64)
    delivered=m.target_delivery(s,r)
    assert delivered['target64']==r['history'][64]['targets'][0]
    assert delivered['target32']==np.asarray(delivered['target64'],np.float64).astype(np.float32)[None].tolist()
    assert set(delivered['controller_payload'])=={'z','a','target'}
    assert m.replay(t,m.journal(st.journal)[0],r)==r


def test_zero_goal_keeps_all64(tmp_path):
    s=resolve(fixture())[0];t=task(s);t['goal']=t['context']['latent_history'][-1].copy()
    r=m.numerical(t,m.Store(tmp_path/'ops'))
    assert len(r['history'])==65 and r['resources']['update_returned']==64
    assert all(x['score']['loss']==0 for x in r['history'])


def test_operation_failure_partial_prefix_and_no_refund(tmp_path,monkeypatch):
    s=resolve(fixture())[0];t=task(s);native=m.operation
    def fail(task,x,i,phase):
        if i==3 and phase=='loss':raise ValueError('fixture fail, no retry')
        return native(task,x,i,phase)
    monkeypatch.setattr(m,'operation',fail);st=m.Store(tmp_path/'ops');r=m.numerical(t,st)
    assert len(r['history'])==3 and r['delivered_targets'] is None
    assert r['resources']['loss_attempted']==4 and r['resources']['loss_returned']==3
    assert r['resources']['gradient_attempted']==4 and r['resources']['gradient_returned']==3
    assert r['resources']['update_returned']==3
    monkeypatch.setattr(m,'operation',native)
    assert m.replay(t,m.journal(st.journal)[0],r)==r
    assert m.target_delivery(s,r)['controller_payload'] is None


def test_timeout_without_intent_final_failure_is_durable(tmp_path):
    s=resolve(fixture())[0];t=task(s);clock=[0.]
    st=m.Store(tmp_path/'ops',clock=lambda:clock[0]);clock[0]=m.WALL+1
    r=m.numerical(t,st);events,_=m.journal(st.journal)
    assert len(events)==1 and events[0]['payload']['intent_recorded'] is False
    assert r['resources']==m.counters() and r['history']==[]
    assert m.replay(t,events,r)==r


def test_real_store_put_emit_reader_single_lf_and_fsync_order(tmp_path,monkeypatch):
    seen=[];real=m.os.fsync
    def sync(fd):
        # os.fstat fails if fsync runs after the stream was closed.
        seen.append(m.os.fstat(fd).st_mode);return real(fd)
    monkeypatch.setattr(m.os,'fsync',sync);st=m.Store(tmp_path/'disk')
    st.emit('one',dict(x=1));st.put('blob.json',m.raw(dict(scalar=1.))+b'\n');st.emit('two',dict(x=2))
    b=st.journal.read_bytes();assert b.count(b'\n')==2 and b'\n\n' not in b
    es,tail=m.journal(st.journal);assert len(es)==2 and tail==st.tail
    assert [stat.S_ISDIR(x) for x in seen]==[True,False,True,False,False,True,False]
    assert m.read(st.path/'blob.json',st.files['blob.json']['sha256'])==dict(scalar=1.)
    with pytest.raises(ValueError):st.put('blob.json',b'{}')
    with pytest.raises(FileExistsError):m.durable(st.path/'blob.json',b'{}')


def test_prepare_verify_run_consume_actual_disk_fixtures(tmp_path,monkeypatch):
    p,h,out,rh=run_fixture(tmp_path,monkeypatch)
    rc=m.consume(p,h,out,rh,tmp_path/'receipt.json');r=m.read(rc['path'],rc['sha256'])
    assert len(r['plans'])==2 and r['denominator']==2 and 'all2' in r['status']
    assert r['actual_controller_calls']==r['actual_new_neural_calls']==0 and r['verification_extra_cpu_arithmetic']
    assert m.read(p,h)['contract']['optimizer']['updates']==64
    with pytest.raises(FileExistsError):m.run(p,h,out,'MAIN_NUMERICAL_ONLY')
    with pytest.raises(ValueError):m.run(p,h,tmp_path/'unauthorized','HOST_CUDA')
    assert not (tmp_path/'unauthorized').exists()


def test_cli_relative_paths_fixture_only(tmp_path,monkeypatch,capsys):
    dr=saved_fixture(tmp_path,monkeypatch);monkeypatch.chdir(tmp_path)
    m.main(['prepare','--delivery',dr['path'],'--delivery-sha',dr['sha256'],'--output','prepared.json'])
    pr=json.loads(capsys.readouterr().out)
    m.main(['verify','--prepared','prepared.json','--prepare-sha',pr['sha256']])
    assert json.loads(capsys.readouterr().out)['verified']
    m.main(['optimize','--prepared','prepared.json','--prepare-sha',pr['sha256'],'--output','output','--authorization','MAIN_NUMERICAL_ONLY'])
    rr=json.loads(capsys.readouterr().out)
    m.main(['consume','--prepared','prepared.json','--prepare-sha',pr['sha256'],'--output','output','--report-sha',rr['sha256'],'--receipt','receipt.json'])
    assert json.loads(capsys.readouterr().out)['path']==str(tmp_path/'receipt.json')


def rewrite_journal(out,events):
    tail='0'*64;b=b''
    for i,e in enumerate(events):
        e['index']=i;e['previous']=tail;tail=m.digest(e);b+=m.raw(dict(event=e,sha256=tail))+b'\n'
    (out/'events.jsonl').write_bytes(b)
    r=json.loads((out/'report.json').read_bytes());r.update(journal_tail=tail,event_count=len(events));return r


@pytest.mark.parametrize('fault',['FIRST','context','goal','extra_payload','gradient','history','counter','unknown_event','splice','bool_event','blank','extra_file','sha'])
def test_saved_consumer_rejects_mutation(tmp_path,monkeypatch,fault):
    p,h,out,rh=run_fixture(tmp_path,monkeypatch);r=json.loads((out/'report.json').read_bytes())
    if fault=='FIRST':r['rows'][0]['target64'][0]+=1
    elif fault=='context':r['rows'][0]['controller_payload']['z'][0][0][0]+=1
    elif fault=='goal':r['rows'][0]['goal_sha256']='0'*64
    elif fault=='extra_payload':r['rows'][0]['controller_payload']['commands']=[1]
    elif fault in ('gradient','history','counter'):
        n='root0_numerical.json';v=json.loads((out/n).read_bytes())
        if fault=='gradient':v['history'][0]['gradient'][0][0]+=.1
        elif fault=='history':v['history'][3]['targets'][0][0]+=.1
        else:v['resources']['gradient_attempted']=0
        b=m.raw(v)+b'\n';(out/n).write_bytes(b);r['files'][n]=dict(bytes=len(b),sha256=m.bh(b));r['rows'][0]['optimizer_result_sha256']=m.bh(b)
    elif fault in ('unknown_event','splice','bool_event'):
        es,_=m.journal(out/'events.jsonl')
        if fault=='unknown_event':es[3]['kind']='unknown'
        elif fault=='bool_event':es[2]['payload']['root']=False
        else:es[3],es[5]=es[5],es[3]
        r=rewrite_journal(out,es)
    elif fault=='blank':(out/'events.jsonl').write_bytes((out/'events.jsonl').read_bytes()+b'\n')
    elif fault=='extra_file':(out/'extra').write_bytes(b'')
    else:(out/'root0_numerical.json').write_bytes(b'{}')
    (out/'report.json').write_bytes(m.raw(r)+b'\n');rh=m.sha(out/'report.json')
    bind_completion_to_report(out)
    with pytest.raises(ValueError):m.consume(p,h,out,rh,tmp_path/'receipt.json')


def test_source_and_input_binding_tamper_fail_before_output(tmp_path,monkeypatch):
    p,h=prepared_fixture(tmp_path,monkeypatch);v=m.read(p,h)
    # A temporary inherited metadata authority, never a frozen source.
    path=next(Path(x) for x in v['bindings'] if Path(x).parent==tmp_path and x!=str(p))
    path.write_bytes(b'{}')
    with pytest.raises(ValueError):m.run(p,h,tmp_path/'never','MAIN_NUMERICAL_ONLY')
    assert not (tmp_path/'never').exists()


@pytest.mark.parametrize('path',list(m.PINS))
def test_literal_upstream_source_protocol_pin_drift_rejected(path,monkeypatch):
    bad=dict(m.PINS);bad[path]='0'*64;monkeypatch.setattr(m,'PINS',bad)
    with pytest.raises(ValueError,match='binding changed'):m.source_seals()


@pytest.mark.parametrize('field',['future_state','outcome','program_endpoint','prior_target','commands'])
def test_optimizer_whitelist_rejects_privileged_fields(tmp_path,field):
    t=task(resolve(fixture())[0]);t[field]=[1]
    with pytest.raises(ValueError):m.numerical(t,m.Store(tmp_path/'unused'))


def test_all2_partial_first_root_does_not_skip_second(tmp_path,monkeypatch):
    p,h=prepared_fixture(tmp_path,monkeypatch);native=m.operation
    def fail(task,x,i,phase):
        if task['root_slot']==0 and i==0 and phase=='loss':raise ValueError('fixture first-root failure')
        return native(task,x,i,phase)
    monkeypatch.setattr(m,'operation',fail);out=tmp_path/'partial';rr=m.run(p,h,out,'MAIN_NUMERICAL_ONLY')
    monkeypatch.setattr(m,'operation',native)
    rc=m.consume(p,h,out,rr['sha256'],tmp_path/'receipt.json');v=m.read(rc['path'],rc['sha256'])
    assert [x['status'] for x in v['plans']]==['partial','delivered'] and v['denominator']==2
    assert v['plans'][0]['controller_payload'] is None and v['plans'][0]['resources']['loss_attempted']==1


def test_verify_time_exhaustion_no_arithmetic_dispatch_all2_partial(tmp_path,monkeypatch):
    p,h=prepared_fixture(tmp_path,monkeypatch);clock=[0.];native=m.verify
    def slow_verify(*args):
        result=native(*args);clock[0]=m.WALL+1.;return result
    monkeypatch.setattr(m,'verify',slow_verify)
    def forbidden(*args):raise AssertionError('no arithmetic after verification consumes stage budget')
    monkeypatch.setattr(m,'operation',forbidden);out=tmp_path/'timeout'
    rr=m.run(p,h,out,'MAIN_NUMERICAL_ONLY',clock=lambda:clock[0]);r=m.read(rr['path'],rr['sha256'])
    assert [x['status'] for x in r['rows']]==['partial','partial']
    assert all(x['resources']==m.counters() for x in r['rows'])
    c=m.consume(p,h,out,rr['sha256'],tmp_path/'receipt.json');rc=m.read(c['path'],c['sha256'])
    assert rc['complete'] is False and rc['whole_stage_wall_through_report_fsync']>m.WALL
    assert 'partial' in rc['status'] and len(rc['plans'])==2


def test_final_report_fsync_timeout_cannot_certify_complete(tmp_path,monkeypatch):
    p,h=prepared_fixture(tmp_path,monkeypatch);clock=[0.];native=m.durable
    def slow_report(path,b):
        native(path,b)
        if Path(path).name=='report.json':clock[0]=m.WALL+1.
    monkeypatch.setattr(m,'durable',slow_report);out=tmp_path/'late_report'
    rr=m.run(p,h,out,'MAIN_NUMERICAL_ONLY',clock=lambda:clock[0])
    c=m.consume(p,h,out,rr['sha256'],tmp_path/'receipt.json');rc=m.read(c['path'],c['sha256'])
    assert all(x['status']=='delivered' for x in rc['plans'])  # Arithmetic captured, stage NOT complete.
    assert rc['complete'] is False and rc['status'].startswith('partial_')


def test_forged_overwall_report_success_rejected(tmp_path,monkeypatch):
    p,h,out,rh=run_fixture(tmp_path,monkeypatch);r=m.read(out/'report.json',rh)
    r['whole_numerical_wall']=m.WALL+1.;(out/'report.json').write_bytes(m.raw(r)+b'\n')
    c=json.loads((out/'completion.json').read_bytes());c.update(report_sha256=m.sha(out/'report.json'),
        whole_stage_wall_through_report_fsync=m.WALL+2.,complete=True,status='complete_numerical_within_budget')
    (out/'completion.json').write_bytes(m.raw(c)+b'\n')
    with pytest.raises(ValueError,match='wall exceeded'):m.consume(p,h,out,m.sha(out/'report.json'),tmp_path/'receipt.json')


def test_npy_exact_hash_shape_dtype_trailing_rejection(tmp_path):
    a=np.ones((1,192),np.float32);s=io.BytesIO();np.save(s,a);b=s.getvalue();p=tmp_path/'array_00000.npy';p.write_bytes(b)
    d=dict(data_sha256=m.bh(a.tobytes()),dtype='<f4',file_bytes=len(b),file_sha256=m.bh(b),shape=[1,192])
    assert m.same(a,m.load_array(tmp_path,p.name,d))
    b+=b'\0';p.write_bytes(b);d.update(file_bytes=len(b),file_sha256=m.bh(b))
    with pytest.raises(ValueError):m.load_array(tmp_path,p.name,d)


def test_json_duplicate_nonfinite_and_symlink(tmp_path):
    for b in (b'{"a":1,"a":2}',b'{"a":NaN}'):
        with pytest.raises(ValueError):m.decode(b)
    p=tmp_path/'a';p.write_bytes(b'{}');link=tmp_path/'link';link.symlink_to(p)
    with pytest.raises(ValueError):m.read(link,m.sha(p))


def test_pins_and_cold_import_without_neural_or_old_runtimes():
    assert m.source_seals()[str(m.ROOT/'experiments/spectral_reachability'/ (m.core.NAME+'.py'))]==m.PINS['experiments/spectral_reachability/'+m.core.NAME+'.py']
    script='import sys; from experiments.spectral_reachability import '+m.NAME+' as m; m.main(["contract"]); assert "torch" not in sys.modules; assert not any("target_only_delivery_" in k or "history3_tracker_cuda_" in k for k in sys.modules)'
    p=subprocess.run([sys.executable,'-B','-c',script],cwd=m.ROOT,capture_output=True,text=True,check=True)
    assert json.loads(p.stdout)['neural_calls']==0
