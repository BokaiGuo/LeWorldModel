"""RAM/native disk ABI only. No torch import, model loading, real prepare or fullstrict."""
from copy import deepcopy
import gzip
import io
import json
from pathlib import Path
import time
import shutil
import os
import stat
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p2_full30_control15_feedback_20261029_v1 as R

K, N = R.K, R.native


def ctx(root=0):
    z = np.random.default_rng(100+root).normal(0, .05, (3, 192)).astype(np.float32)
    return dict(layout='states_with_between_state_actions_v1', context_length=3, latent_dim=192,
                action_dim=10, latent_history=z.tolist(), action_history=np.zeros((2, 10), np.float32).tolist())


def packet(c, a):
    """Synthetic arithmetic callbacks with all original native shapes, not a neural model."""
    tokens = np.concatenate((np.asarray(c['action_history'], np.float32), N.kernel.normalized_history(a)))
    ae = np.tile(tokens, (1, 20))[:, :192].astype(np.float32)[None]
    z = np.asarray(c['latent_history'], np.float32)[None]
    pred = (z + np.float32(.03125)*ae).astype(np.float32)
    proj = pred.reshape(3, 192).copy()
    out = deepcopy(c)
    out.update(latent_history=np.concatenate((z[0, 1:], proj[-1:])).tolist(), action_history=tokens[1:].tolist())
    modules = [N.module_receipt(0, 'action_encoder', [N.blob(tokens[None])], N.blob(ae)),
               N.module_receipt(1, 'predictor', [N.blob(z), N.blob(ae)], N.blob(pred)),
               N.module_receipt(2, 'pred_proj', [N.blob(pred.reshape(3, 192))], N.blob(proj))]
    return out, modules


def make_job(base=0, policy='I'):
    u = np.random.default_rng(base+19).normal(0, .07, 60)
    actions = K.parent.chart(u).reshape(30, 2)
    cap = K.parent.C15 if base%2==0 else float(np.sqrt(K.parent.C30**2-float(u[:30]@u[:30])))
    return dict(base_index=base, root_slot=base//2, arm=K.parent.ARMS[base%2], policy=policy,
                input_context=ctx(base), goal=np.zeros(192).tolist(), parent_logits=u.tolist(),
                incumbent_logits=u[30:].tolist(), suffix_cap=cap, prefix_raw=actions[:15].tolist(),
                parent_raw_bits=actions.tobytes().hex(), task_id='c'*64, task_sha256='d'*64,
                parent_witness_sha256='e'*64)


class FixtureBackend:
    def __init__(self, row, store):
        self.row, self.store = row, store
        self.event = lambda e: store.event(dict(slot=row['slot'], event=e))

    def rollout(self, c, actions, tag):
        cs = []
        for k, a in enumerate(actions):
            i = len(self.row['forwards'])
            request = N.packet_request(i, c, a, tag+f'/{k}')
            out, ms = packet(c, a)
            for j, m in enumerate(ms):
                m['index'] = 3*i+j
            f = dict(index=i, tag=tag+f'/{k}', request=request, input=deepcopy(c), raw=N.blob(a),
                     modules=ms, output=out, completed=True)
            self.event(dict(kind='forward_intent', request=request))
            self.event(dict(kind='forward_inputs', index=i, context=c, raw=f['raw']))
            for e in N.module_events(f):
                self.event(e)
            N.validate_packet(f)
            self.event(dict(kind='forward_return', index=i, context=out, completed=True))
            self.row['forwards'].append(f)
            cs.append(deepcopy(out))
            c = out
        return cs

    def jacobian(self, c, u, tag):
        i = len(self.row['AD'])
        request = R.ad_request(i, c, u, tag)
        self.event(dict(kind='AD_intent', request=request))
        paths = self.rollout(c, K.parent.chart(u), tag+'/forward')
        # The synthetic J is not claimed as true neural or finite-difference evidence.
        j = np.zeros((192, 30), np.float32)
        buf = io.BytesIO()
        np.save(buf, j, allow_pickle=False)
        name = f'slot{self.row["slot"]}_J{i}.npy'
        self.store._write(name, buf.getvalue())
        d = dict(file=name, sha256=R.sha(self.store.output/name), shape=[192, 30], dtype='float32')
        self.row['AD'].append(dict(request=request, contexts=paths, array=d, completed=True))
        self.last_ad_contexts = deepcopy(paths)
        self.event(dict(kind='AD_return', index=i, array=d, contexts=paths, completed=True))
        return j.astype(float)


@pytest.fixture(scope='module')
def bundle(tmp_path_factory):
    base = tmp_path_factory.mktemp('full30-fixture')
    output = base/(K.NAME+'_fixture')
    store = R.Store(output, time.monotonic())
    bindings = {}
    policy = base/'policy'
    policy.mkdir()
    loadfiles = []
    for name in ('config.json', 'weights.pt'):
        pth = policy/name
        with pth.open('xb') as f:
            f.write(b'fixture, not a model')
        h = R.sha(pth)
        bindings[str(pth)] = h
        d = dict(path=str(pth), sha256=h, bytes=pth.stat().st_size, hash_buffer_sha256=h,
                 deserialize_buffer_sha256=h, same_buffer=True)
        loadfiles.append(d)
        store.event(dict(kind='load_intent', path=str(pth), expected_sha256=h))
        store.event(dict(kind='consumed_buffer_frozen', receipt=d))
    source = str(Path(__file__).resolve())
    bindings[source] = R.sha(source)
    model_hash, prepared_sha = 'a'*64, 'b'*64
    prepared = str(base/'synthetic_prepare.json')
    p = dict(jobs=[make_job(i,pol) for i in range(4) for pol in ('I','F')], bases=[dict(base_index=i) for i in range(4)], policy=str(policy), expected_model_hash=model_hash, bindings_sha256=bindings,
             inherited_resources={'fixture': True, 'charged_again_as_new': False})
    load = dict(schema=N.loader.NAME+'.loader_receipt', prepare_sha256=prepared_sha,
                adapter_source_sha256=R.sha(N.loader.__file__),
                legacy_loader_source_sha256=N.loader.native.SOURCES['experiments/checkpoint_compat.py'],
                files=loadfiles, device='cuda:0', remapped_keys=[], missing=[], unexpected=[],
                tensor_hash_before=model_hash, frozen_eval=True)
    store.put('loader.json', load)
    store.put('classes.json', {source: bindings[source]})
    store.put('binding.json', dict(prepared=prepared, prepare_sha256=prepared_sha))
    summaries, rows = [], []
    for i, job in enumerate(p['jobs']):
        root, arm = job['root_slot'], job['arm']
        row = dict(slot=i, index=root, base_index=job['base_index'], policy=job['policy'],
                   arm=arm, status='sealed', forwards=[], AD=[])
        store.event(dict(kind='slot_started', slot=i, root_slot=root, arm=arm, policy=job['policy'], base_index=job['base_index']))
        row['result'] = K.execute(job, FixtureBackend(row, store))
        row['plan'] = R.row_plan(p, job, row)
        row['resources'] = K.resources(row['forwards'], row['AD'])
        row['memory'] = dict(allocated_peak=1, reserved_peak=1)
        store.put(f'slot{i}.json.gz', row)
        store.event(dict(kind='slot_sealed', slot=i, witness=row['plan']['new_witness_sha256']))
        summaries.append(dict(slot=i, root_slot=root, arm=arm, policy=job['policy'], base_index=job['base_index'],
                              status='sealed', file=f'slot{i}.json.gz', wall_seconds=.1,
                              new_suffix_Q=row['result']['new_suffix_Q'], contexts=row['result']['contexts'], resources=row['resources']))
        rows.append(row)
    store.close()
    store.put('seal.json', dict(schema=K.NAME+'.seal', complete=True, through_durable_data_seconds=.5,
              data_artifacts_sha256={f.name: R.sha(f) for f in output.iterdir()}), final=True)
    r = dict(schema=K.NAME+'.report', status='all8_feedback_sealed_NOT_admitted', prepared=prepared,
             prepare_sha256=prepared_sha, contract=K.contract(), rows=summaries, errors=[], first_error=None,
             model_before=model_hash, model_after=model_hash, source_verified=True, host={'fixture': True},
             wall_seconds=1., actual_resources=R.census(R.journal(output)), inherited_resources=p['inherited_resources'],
             loader_memory=dict(allocated_peak=1, reserved_peak=1),
             precision_before=R.precision_expected(), precision_after=R.precision_expected(),
             policy_roster=R.roster(p,summaries),
             artifact_sha256={f.name: R.sha(f) for f in output.iterdir()}, **K.FLAGS)
    store.put('report.json', r, final=True)
    store.put('completion.json',dict(schema=K.NAME+'.completion',report_sha256=R.sha(output/'report.json'),
              boundary='after_report_fsync_and_hash',elapsed_seconds=1.1),final=True)
    return dict(output=output, p=p, report=r, rows=rows)


def test_full_shape_budget_before_delivery(bundle):
    budget = R.size_budget(bundle['rows'], R.journal(bundle['output']))
    print('FULL216_8J_TRANSPORT', budget)
    assert budget['full_J_NPY_bytes'] == 8*(192*30*4+128)
    assert budget['conservative_uncompressed_total'] < R.CAP-R.RESERVE
    assert sum(len(row['forwards']) for row in bundle['rows']) == 216
    assert sum(len(row['AD']) for row in bundle['rows']) == 8


def test_complete_saved_native_disk_consumer(bundle, monkeypatch):
    monkeypatch.setattr(R, 'verify', lambda *_: bundle['p'])
    result = R.consume(bundle['output'], R.sha(bundle['output']/'report.json'), R.sha(bundle['output']/'completion.json'))
    assert len(result['plans']) == 8 and len(result['policy_slots']) == 12
    assert result['actual_new'] == K.contract()['total']


@pytest.mark.parametrize('field,value', [('goal',[True]*192),('suffix_cap',True),('root_slot',True),
    ('incumbent_logits',[float('nan')]*30),('prefix_raw',[[0,0]]*15)])
def test_bad_job_rejected(field, value):
    job=make_job();job[field]=value
    with pytest.raises(ValueError): K.validate(job)


def test_float64_original_witness_and_chart():
    job=make_job(1);u,cap=K.validate(job)
    assert cap > K.parent.C15
    assert K.parent.chart(u).tobytes() == np.asarray(np.tanh(u),'<f4').reshape(3,5,2).tobytes()
    prop=K.proposals(job,np.ones(192),np.zeros((192,30)))
    assert len(prop['logits']) == 5
    assert all(v==u.tolist() for v in prop['logits'])


def test_nonzero_continuous_chain():
    job=make_job();rng=np.random.default_rng(15)
    j=rng.normal(size=(192,30)).astype(np.float32);z=rng.normal(size=192)
    out=K.proposals(job,z,j);u=np.asarray(job['incumbent_logits'])
    want=(2/192)*(j.astype(float)*(1-np.tanh(u)**2)[None,:]).T@z
    np.testing.assert_array_equal(out['gradient'],want)
    for v in out['logits'][1:]: K.legal_new(v,job)


@pytest.mark.parametrize('mutation', ['completed','input','module','request','Jshape'])
def test_saved_native_mutations(bundle, mutation):
    row=deepcopy(bundle['rows'][0]);job=bundle['p']['jobs'][0]
    if mutation=='completed': row['AD'][0]['completed']=False
    if mutation=='input': row['forwards'][0]['input']['latent_history'][0][0]+=.1
    if mutation=='module': row['forwards'][0]['modules'][0]['completed']=False
    if mutation=='request': row['AD'][0]['request']['coordinates']='logits'
    def load(d): return np.zeros((192,10) if mutation=='Jshape' else (192,30),np.float32)
    backend=R.SavedBackend(row,lambda _:None,load)
    with pytest.raises(ValueError): K.execute(job,backend)


def test_real_store_single_lf_and_exclusive(tmp_path):
    s=R.Store(tmp_path/(K.NAME+'_disk'),time.monotonic())
    s.put('metadata.json',dict(a=1));s.event(dict(kind='one'));s.event(dict(kind='two'));s.close()
    assert R.journal(s.output)==[dict(kind='one'),dict(kind='two')]
    assert b'\n\n' not in gzip.decompress((s.output/'events.jsonl.gz').read_bytes())
    with pytest.raises(FileExistsError):s.put('metadata.json',dict(a=2))


def test_unknown_authority_before_read(monkeypatch):
    monkeypatch.setattr(R,'read',lambda *_:pytest.fail('must reject before reading'))
    with pytest.raises(ValueError):R.build({})


def test_wrong_auth_no_io(tmp_path):
    with pytest.raises(ValueError):R.run_cuda('no','0'*64,tmp_path/'nothing',None)
    assert not (tmp_path/'nothing').exists()


def test_expired_verify_never_dispatch(tmp_path,monkeypatch):
    ticks=iter((0.,1801.,1801.,1801.,1801.,1801.,1801.,1801.,1801.,1801.))
    monkeypatch.setattr(R.time,'monotonic',lambda:next(ticks,1801.))
    monkeypatch.setattr(R,'verify',lambda *_:dict(bindings_sha256={},expected_model_hash='x',jobs=[make_job(i,p) for i in range(4) for p in ('I','F')],bases=[dict(base_index=i) for i in range(4)],inherited_resources={}))
    with pytest.raises(ValueError):R.run_cuda('fake','0'*64,tmp_path/(K.NAME+'_expired'),R.AUTH)
    report=R.linker.decode((tmp_path/(K.NAME+'_expired')/'report.json').read_bytes())
    assert report['status']=='failed_partial' and report['actual_resources']['forward_attempted']==0


def origin_fixture():
    ep=dict(schema=R.E.NAME+'.prepare',expected_model_hash='a'*64,plans=[],rows=[])
    ec=dict(schema=R.E.NAME+'.consume',status='all28_full30_RGB_verified_NOT_admitted',
            fixed_denominator=28,plan_denominator=4,actual_context_chains=[],rows=[],
            P1_admission=False,P2_admission=False,P2_mechanism=False,C3=False,physical_release=False)
    for i in range(4):
        job=make_job(i);u=np.asarray(job['parent_logits']);actions=K.parent.chart(u).reshape(30,2)
        c=ctx(i);warm=np.zeros((10,2),np.float32)
        contexts=[]
        for a in actions.reshape(6,5,2):
            c,_=packet(c,a);contexts.append(deepcopy(c))
        plan=dict(root_slot=i//2,arm=job['arm'],task_id=job['task_id'],task_sha256=job['task_sha256'],
                  logits=u.tolist(),commands=actions.tolist(),raw_float32_bits=actions.tobytes().hex(),
                  model_hash=ep['expected_model_hash'],normalization=dict(mean=N.kernel.ACTION_MEAN.tolist(),std=N.kernel.ACTION_STD.tolist()),
                  root_warm_raw10=warm.tolist(),goal=job['goal'],contexts=contexts)
        plan['witness_sha256']=K.digest(plan)
        wrapper=dict(index=i,root_slot=i//2,arm=job['arm'],task_id=job['task_id'],
                     planner_plan=plan,commands=actions.tolist(),selection_digest=K.digest(i))
        ep['plans'].append(wrapper)
        checks=[]
        for j,cp in enumerate(R.E.CP):
            a=np.concatenate((warm,actions[:cp]))[-10:]
            normalized=N.kernel.normalized_history(a[:5]).tolist()+N.kernel.normalized_history(a[5:]).tolist()
            common=dict(plan_index=i,checkpoint=cp,task_id=job['task_id'],root_slot=i//2,
                        arm=job['arm'],selection_digest=wrapper['selection_digest'],goal192=job['goal'],
                        source=dict(step=cp,path='fixture.png'),causal_raw10_bits=a.tobytes().hex())
            ep['rows'].append(dict(common,absolute_step=cp+10,normalized_action_history=normalized))
            z=ctx(i)['latent_history'][-1] if cp==0 else contexts[j-1]['latent_history'][-1]
            ec['rows'].append(dict(common,latent_missing=False,actual192=z))
            checks.append(dict(step=cp,context_missing=False,context=deepcopy(contexts[j-1] if j else ctx(i))))
        ec['actual_context_chains'].append(dict(plan_index=i,root_slot=i//2,arm=job['arm'],checkpoints=checks))
    return ep,ec


def test_original_schema_warm_cp5_and_future_label_firewall():
    ep,ec=origin_fixture();bases,jobs=R.material(ep,ec)
    assert len(bases)==4 and len(jobs)==8
    for i in range(4):
        for j in (4,5,6):
            ec['rows'][7*i+j]['actual192']=['FUTURE_OUTCOME_FORBIDDEN']
            ep['rows'][7*i+j]['native_metrics']={'success':'do not read'}
            ec['actual_context_chains'][i]['checkpoints'][j]['context']='future hidden'
    assert R.material(ep,ec)==(bases,jobs)
    assert all(set(j)==K.JOB_FIELDS for j in jobs)
    assert jobs[1]['input_context']==ec['actual_context_chains'][0]['checkpoints'][3]['context']
    bad=deepcopy(jobs[0]);bad['actual_cp30']=0
    with pytest.raises(ValueError):K.validate(bad)


@pytest.mark.parametrize('mutation',['cpbool','absolute','raw','normalizer','latent','missing','chain','goal','parentbits'])
def test_origin_join_rejects_mutations(mutation):
    ep,ec=origin_fixture()
    if mutation=='cpbool':ec['rows'][1]['checkpoint']=True
    if mutation=='absolute':ep['rows'][1]['absolute_step']=5
    if mutation=='raw':ep['rows'][1]['causal_raw10_bits']='00'
    if mutation=='normalizer':ep['rows'][1]['normalized_action_history'][0][0]+=.1
    if mutation=='latent':ec['rows'][1]['actual192'][0]=float('nan')
    if mutation=='missing':ec['actual_context_chains'][0]['checkpoints'][3]['context_missing']=True
    if mutation=='chain':ec['actual_context_chains'][0]['checkpoints'][3]['context']['latent_history'][0][0]+=.1
    if mutation=='goal':ec['rows'][1]['goal192'][0]+=.1
    if mutation=='parentbits':ep['plans'][0]['planner_plan']['raw_float32_bits']='00'
    with pytest.raises(ValueError):R.material(ep,ec)


def copied(bundle, tmp_path):
    output=tmp_path/(K.NAME+'_copy');shutil.copytree(bundle['output'],output)
    return output


def reseal(output):
    report=R.linker.decode((output/'report.json').read_bytes())
    report['artifact_sha256']={n:R.sha(output/n) for n in report['artifact_sha256']}
    seal=R.linker.decode((output/'seal.json').read_bytes())
    seal['data_artifacts_sha256']={n:h for n,h in report['artifact_sha256'].items() if n!='seal.json'}
    (output/'seal.json').write_bytes(K.raw(seal))
    report['artifact_sha256']['seal.json']=R.sha(output/'seal.json')
    (output/'report.json').write_bytes(K.raw(report))
    c=R.linker.decode((output/'completion.json').read_bytes());c['report_sha256']=R.sha(output/'report.json')
    (output/'completion.json').write_bytes(K.raw(c))


@pytest.mark.parametrize('mutation',['unknown','move','selection_before_returns','doubleLF'])
def test_whole_journal_chronology_coherent_reseal(bundle,tmp_path,monkeypatch,mutation):
    output=copied(bundle,tmp_path);events=R.journal(output)
    if mutation=='unknown':events.insert(4,dict(kind='unknown_event'))
    if mutation=='move':events[8],events[9]=events[9],events[8]
    if mutation=='selection_before_returns':
        i=next(i for i,e in enumerate(events) if e.get('event',{}).get('kind')=='suffix_selection')
        events.insert(i-1,events.pop(i))
    previous='0'*64;data=[]
    for i,e in enumerate(events):
        v=dict(index=i,previous_sha256=previous,payload=e);previous=K.digest(v)
        data.append(K.raw(dict(v,event_sha256=previous)))
    (output/'events.jsonl.gz').write_bytes(gzip.compress(b''.join(data)+(b'\n' if mutation=='doubleLF' else b''),mtime=0))
    reseal(output);monkeypatch.setattr(R,'verify',lambda *_:bundle['p'])
    with pytest.raises(ValueError):R.consume(output,R.sha(output/'report.json'),R.sha(output/'completion.json'))


@pytest.mark.parametrize('value',[True,-1.,0.5,1800.01,float('inf')])
def test_completion_boundary_rejected(bundle,tmp_path,monkeypatch,value):
    output=copied(bundle,tmp_path)
    c=R.linker.decode((output/'completion.json').read_bytes());c['elapsed_seconds']=value
    if np.isfinite(value):data=K.raw(c)
    else:data=json.dumps(c).encode()  # Decoder must refuse nonfinite too.
    (output/'completion.json').write_bytes(data)
    monkeypatch.setattr(R,'verify',lambda *_:bundle['p'])
    with pytest.raises(ValueError):R.consume(output,R.sha(output/'report.json'),R.sha(output/'completion.json'))


def test_completion_external_sha_and_report_binding(bundle,tmp_path,monkeypatch):
    output=copied(bundle,tmp_path);monkeypatch.setattr(R,'verify',lambda *_:bundle['p'])
    with pytest.raises(ValueError):R.consume(output,R.sha(output/'report.json'),'0'*64)
    c=R.linker.decode((output/'completion.json').read_bytes());c['report_sha256']='0'*64
    (output/'completion.json').write_bytes(K.raw(c))
    with pytest.raises(ValueError):R.consume(output,R.sha(output/'report.json'),R.sha(output/'completion.json'))


@pytest.mark.parametrize('field,value',[('wall_seconds',1801.),('wall_seconds',True),('P2_mechanism',True),
    ('precision_after',{}),('model_after','corrupt')])
def test_report_gate_mutations(bundle,tmp_path,monkeypatch,field,value):
    output=copied(bundle,tmp_path);r=R.linker.decode((output/'report.json').read_bytes());r[field]=value
    (output/'report.json').write_bytes(K.raw(r));reseal(output)
    monkeypatch.setattr(R,'verify',lambda *_:bundle['p'])
    with pytest.raises(ValueError):R.consume(output,R.sha(output/'report.json'),R.sha(output/'completion.json'))


def test_per_record_sync_flush_budget_measures_real_writer(tmp_path):
    s=R.Store(tmp_path/(K.NAME+'_entropy'),time.monotonic());rng=np.random.default_rng(25)
    events=[dict(kind='transport_only',data=rng.bytes(1024+i).hex()) for i in range(128)]
    for e in events:s.event(e)
    s.close();budget=R.size_budget([],events)
    assert budget['sync_flush_records']==128 and budget['sync_flush_bytes_per_record']==64
    assert (s.output/'events.jsonl.gz').stat().st_size<=budget['journal_compressed_bound']
    assert R.journal(s.output)==events


def test_put_fsync_open_regular_then_dir(tmp_path,monkeypatch):
    s=R.Store(tmp_path/(K.NAME+'_fsync'),time.monotonic());seen=[]
    real=os.fsync
    def checked(fd):
        seen.append('file' if stat.S_ISREG(os.fstat(fd).st_mode) else 'dir');return real(fd)
    monkeypatch.setattr(os,'fsync',checked)
    s.put('meta.json',{'x':1});assert seen==['file','dir']
    s.close()


def test_failed_nested_attempts_not_refunded():
    e=[dict(slot=0,event=dict(kind='AD_intent')),dict(slot=0,event=dict(kind='forward_intent')),
       dict(slot=0,event=dict(kind='predict_API_intent')),dict(slot=0,event=dict(kind='module_intent',resource='predictor'))]
    c=R.census(e)
    assert c['jacobian_attempted']==c['forward_attempted']==c['predict_API_attempted']==1
    assert c['jacobian_returned']==c['forward_returned']==c['predict_API_returned']==0


def test_source_binding_tamper(tmp_path):
    p=tmp_path/'source.py';p.write_bytes(b'fixture')
    pins={str(p):R.sha(p)};R.rehash(pins);p.write_bytes(b'changed')
    with pytest.raises(ValueError):R.rehash(pins)



