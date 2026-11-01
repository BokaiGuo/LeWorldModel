"""RAM-only input-statistics experiment tests. No temp, models, environment or fit."""
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image
from experiments.spectral_reachability import p1_constructive_archive_live_input_summary_20261027_v1 as m


def frame(v): return np.full((224,224,3),v,np.uint8)


def payload(archive=True):
    a=np.arange(40,dtype=np.float32).reshape(20,2)/10-2
    if archive: return dict(frames=[frame(v) for v in (0,16,32,48,64)],actions=a)
    return dict(plan_frames=[frame(v) for v in (3,8,13)],history_frames=[frame(3) for _ in range(3)],actions=a[:10],warmup=np.zeros((10,2),np.float32))


def test_pixel_math_fixed_hist_and_temporal_denominators():
    r=m.pixels([frame(0),frame(16),frame(255)],[0,5,10])
    assert r['frame_count']==3 and r['pixels_per_channel']==3*224*224
    np.testing.assert_allclose(r['mean'],np.full(3,271/3))
    np.testing.assert_allclose(r['std_population'],np.full(3,np.std([0,16,255])))
    counts=np.asarray(r['histogram_counts']); assert (counts.sum(axis=1)==3*224*224).all()
    assert np.array_equal(counts[0][[0,1,15]],np.full(3,224*224))
    assert [x['mean_abs_rgb_delta'] for x in r['adjacent_changes']]==[16.,239.]
    assert all(x['gap_public_steps']==5 and x['changed_pixel_fraction']==1 for x in r['adjacent_changes'])


def test_action_magnitude_delta_packet_norm_raw_not_clipped():
    a=np.tile(np.array([3.,4.],np.float32),(10,1)); before=a.copy(); r=m.actions(a)
    assert r['magnitude_l2']==dict(mean=5.,rms=5.,maximum=5.)
    assert r['command_change_per_public_step']==dict(count=9,mean_l2=0.,maximum_l2=0.)
    np.testing.assert_allclose(r['packet_raw_l2'],[np.sqrt(125)]*2)
    assert r['coordinates_abs_gt_one']==20 and r['physical_speed'] is None
    np.testing.assert_array_equal(before,a)


@pytest.mark.parametrize('fault',['nan','dtype','shape','packet'])
def test_bad_action_input_not_repaired(fault):
    a=np.zeros((10,2),np.float32)
    if fault=='nan': a[0,0]=np.nan
    if fault=='dtype': a=a.astype(float)
    if fault=='shape': a=a[:,0]
    if fault=='packet': a=a[:9]
    with pytest.raises(ValueError): m.actions(a)


def test_exact_repeats_fixed_pairs_and_whole_contexts():
    r=m.repeats([frame(1),frame(1),frame(2)])
    assert r['pair_denominator']==3 and r['equal_pairs']==[[0,1]] and r['equal_pair_fraction']==1/3
    assert r['adjacent_denominator']==2 and r['adjacent_equal']==1 and r['redundant_positions']==1
    archive=[payload() for _ in range(4)]; live=[payload(False) for _ in range(4)]
    result=m.cross_repetition(archive,live)
    assert result['live_root_whole_rgb_history']['positions']==4
    assert result['live_root_whole_rgb_history']['equal_pair_fraction']==1
    assert result['live_root_history']['positions']==12


def test_separate_live_deployment_history_from_future_evaluator_rgb():
    d=payload(False); r=m.summarize_live(d)
    assert r['root_observed_context']['rgb']['unique_entries']==1
    assert r['postroot_rgb_triplet_repetition']['unique_entries']==3
    assert r['postroot_context_consumed_by_planner'] is False
    assert r['root_observed_context']['packet']['positions']==3
    assert r['root_observed_context']['packet']['equal_pairs']==[[0,1]]


def test_archive_extra_frames_not_pooled_with_common_steps():
    d=payload(); r=m.summarize_archive(d)
    assert r['archive_only_rgb_0_to_20']['frame_count']==5
    assert r['common_relative_rgb_0_5_10']['frame_count']==3
    assert r['archive_only_raw_0_to_20']['public_action_count']==20
    assert r['common_raw_0_to_10']['public_action_count']==10
    assert r['W10']['rgb_arrays'][-1]==m.array_id(d['frames'][2])
    assert r['W15']['rgb_arrays'][-1]==m.array_id(d['frames'][3])


def test_pooling_no_cross_branch_pseudotime_or_subset():
    a=[payload() for _ in range(4)]; r=m.pooled(a,'archive')
    assert r['frame_denominator']==12 and r['action_denominator']==40
    assert r['actions']['within_sequence_command_change']['count']==36
    assert 'command_change_per_public_step' not in r['actions'] and 'adjacent_changes' not in r['pixels']
    assert r['independent_sample_count'] is None
    a[-1]=None
    with pytest.raises(ValueError): m.pooled(a,'archive')


class Column:
    def __init__(self,kind):
        self.kind=kind; self.dtype=np.dtype('uint8' if kind=='pixels' else 'float32' if kind=='action' else 'int64')
        self.shape=(18685*25,224,224,3) if kind=='pixels' else (18685*25,2) if kind=='action' else (18685*25,)
    def __getitem__(self,key):
        ids=np.arange(key.start,key.stop) if isinstance(key,slice) else np.asarray(key)
        if self.kind=='pixels': return np.stack([frame(i%251) for i in ids])
        if self.kind=='action': return np.column_stack((ids%29,ids%13)).astype(np.float32)/5
        return ids//25 if self.kind=='episode_idx' else ids%25


class H5(dict):
    def __enter__(self): return self
    def __exit__(self,*args): return False


def fake_h5():
    f=H5({k:Column(k) for k in ('pixels','action','episode_idx','step_idx')})
    f.update(ep_len=np.full(18685,25,np.int64),ep_offset=np.arange(18685,dtype=np.int64)*25)
    return f


def archive_record(f,i):
    off=i*25; a=f['action'][off:off+20]; ids=f['episode_idx'][off:off+21]; steps=f['step_idx'][off:off+21]
    rgb=f['pixels'][[off+s for s in m.STEPS]]
    return dict(episode_index=i,offset=off,length=25,next_or_end_offset=off+25,frame_local_indices=list(m.STEPS),
        frame_global_indices=[off+s for s in m.STEPS],action_local_slice=[0,20],action_global_slice=[off,off+20],
        episode_idx=m.array_id(ids),step_idx=m.array_id(steps),actions=m.array_id(a),raw_actions=a.astype(float).tolist(),
        frames=[dict(step=s,pixels=m.array_id(a)) for s,a in zip(m.STEPS,rgb,strict=True)])


@pytest.mark.parametrize('i',m.EPISODES)
def test_archive_bound_exact_slices_last_endpoint_unbounded_factual(i):
    f=fake_h5(); r=archive_record(f,i); d=m.archive_slice(f,r,i)
    assert d['actions'].shape==(20,2) and d['frames'].shape==(5,224,224,3)
    assert abs(d['actions']).max()>1


@pytest.mark.parametrize('fault',['offset','array','pixels','time','short'])
def test_archive_slice_drift_refused(fault):
    f=fake_h5(); r=archive_record(f,0)
    if fault=='offset': r['offset']=1
    if fault=='array': r['actions']['sha256']='0'*64
    if fault=='pixels': r['frames'][0]['pixels']['sha256']='0'*64
    if fault=='time': r['step_idx']['sha256']='0'*64
    if fault=='short': f['ep_len'][0]=20
    with pytest.raises(ValueError): m.archive_slice(f,r,0)


def test_archive_collect_missing_input_retains_fixed_four(monkeypatch):
    f=fake_h5(); version={'mock_version':1}
    prepared=dict(schema='p1_constructive_offline_fidelity_cuda_20261027_v2.prepare',episode_indices=list(m.EPISODES),
        dataset=str(m.DATA),file_version=version,episodes=[archive_record(f,i) for i in m.EPISODES])
    prepared['episodes'][1]['actions']['sha256']='0'*64
    monkeypatch.setattr(m,'file_version',lambda p:version)
    rows,data=m.collect_archive(prepared,read_h5=lambda:f)
    assert [r['episode'] for r in rows]==list(m.EPISODES)
    assert rows[1]['status']=='missing_or_invalid' and rows[1]['statistics'] is None and data[1] is None
    assert all(rows[i]['status']=='described' for i in (0,2,3))


def test_archive_dataset_version_drift_refuses_before_read(monkeypatch):
    prepared=dict(schema='p1_constructive_offline_fidelity_cuda_20261027_v2.prepare',episode_indices=list(m.EPISODES),
        dataset=str(m.DATA),file_version={'version':1},episodes=[{}]*4)
    monkeypatch.setattr(m,'file_version',lambda p:{'version':2})
    def no_read(): raise AssertionError('must not open altered dataset')
    with pytest.raises(ValueError,match='stat version'): m.collect_archive(prepared,read_h5=no_read)


def fake_live():
    files={}; events=[dict(kind='run_started',prepare={})]; rows=[]
    def add(**event): events.append(event)
    for b,(root,arm) in enumerate(m.LIVE_SLOTS):
        sel=f'{b+1:064x}'; common=dict(branch=b,root_slot=root,arm=arm,selection_digest=sel)
        checkpoints=[]
        for phase in ('warmup','plan'):
            for step in (0,5,10):
                rgb=frame(3 if phase=='warmup' or step==0 else 3+step)
                stream=BytesIO(); Image.fromarray(rgb).save(stream,format='PNG'); raw=stream.getvalue()
                name=f'b{b}_{phase}_{step:02}.png'; files['/RAM/'+name]=raw
                receipt=dict(phase=phase,step=step,png=name,file_sha256=m.digest(raw),pixel_sha256=m.digest(rgb.tobytes()))
                add(kind='fresh_observation',receipt=receipt,**common)
                if phase=='plan': checkpoints.append(deepcopy(receipt))
            for step in range(1,11):
                command=[0.,0.] if phase=='warmup' else [float(np.float32(step/20)),0.]
                add(kind='operation_attempted',resource='public_step',phase=phase,step=step,raw_action=command,**common)
                add(kind='returned_transition',phase=phase,step=step,command=command,**common)
        rows.append(dict(root_slot=root,arm=arm,selection_digest=sel,initial_exact=True,warmup_returned=10,
            own_returned=10,full_horizon=True,first_error=None,secondary_errors=[],checkpoints=checkpoints))
    add(kind='run_finished')
    for i,e in enumerate(events): e['index']=i
    raw=b''.join(m.canonical(e) for e in events); files['/RAM/journal.jsonl']=raw
    report=dict(schema='p1_constructive_plan_physical_20261027_v1.report',errors=[],rows=rows,
        contract=dict(with_target=False,branches=4,own_horizon=10,warmup=10),journal_sha256=m.digest(raw))
    return report,events,files


def test_all_four_saved_live_payloads_and_input_only_fields():
    r,e,files=fake_live(); reader=m.Reader(files.__getitem__)
    # Accessing hidden-state/outcome data is neither needed nor allowed by this interface.
    for row in r['rows']: row.update(state='DO NOT ACCESS',goal='DO NOT ACCESS',native_success='DO NOT ACCESS')
    rows,data=m.collect_live(reader,'/RAM/report.json',r)
    assert len(rows)==4 and all(row['status']=='described' for row in rows)
    assert all(len(d['plan_frames'])==3 and d['actions'].shape==(10,2) for d in data)
    assert len(reader.bindings)==25  #24 PNG + single journal, no source/cert/task files
    reader.finish()


@pytest.mark.parametrize('fault',['missing_frame','wrong_pixels','wrong_action','wrong_arm','report_binding','warmup_notzero'])
def test_one_branch_fault_retains_four_rows_not_replaced(fault):
    r,e,files=fake_live(); branch=[x for x in e if x.get('branch')==0]
    if fault=='missing_frame': e.remove(next(x for x in branch if x['kind']=='fresh_observation' and x['receipt']['phase']=='plan' and x['receipt']['step']==5))
    if fault=='wrong_pixels': next(x for x in branch if x['kind']=='fresh_observation')['receipt']['pixel_sha256']='0'*64
    if fault=='wrong_action': next(x for x in branch if x['kind']=='returned_transition' and x['phase']=='plan')['command']=[9.,9.]
    if fault=='wrong_arm': next(x for x in branch if x['kind']=='fresh_observation')['arm']='other'
    if fault=='report_binding': r['rows'][0]['checkpoints'][0]['file_sha256']='0'*64
    if fault=='warmup_notzero': next(x for x in branch if x['kind']=='operation_attempted' and x.get('phase')=='warmup')['raw_action']=[1.,0.]
    for i,x in enumerate(e): x['index']=i
    raw=b''.join(m.canonical(x) for x in e); files['/RAM/journal.jsonl']=raw; r['journal_sha256']=m.digest(raw)
    rows,data=m.collect_live(m.Reader(files.__getitem__),'/RAM/report.json',r)
    assert len(rows)==4 and rows[0]['status']=='missing_or_invalid' and data[0] is None
    assert all(row['status']=='described' for row in rows[1:])


@pytest.mark.parametrize('fault',['reorder','duplicate','missing'])
def test_no_live_role_selection_or_replacement(fault):
    r,e,files=fake_live()
    if fault=='reorder': r['rows'].reverse()
    if fault=='duplicate': r['rows'][1]=deepcopy(r['rows'][0])
    if fault=='missing': r['rows'].pop()
    with pytest.raises(ValueError): m.collect_live(m.Reader(files.__getitem__),'/RAM/report.json',r)


def test_reader_hash_change_and_privileged_suffix_refusal():
    files={'/RAM/report.json':b'{}'}; reader=m.Reader(files.__getitem__)
    reader.json('/RAM/report.json',m.digest(b'{}')); files['/RAM/report.json']=b'{"changed":true}'
    with pytest.raises(ValueError): reader.finish()
    for name in ('witness.json','certificate.json','sourcepack.json','payload.npz'):
        with pytest.raises(ValueError): reader.raw('/RAM/'+name,'0'*64)


def test_complete_build_in_ram_and_partial_suppresses_comparison(monkeypatch):
    archive_rows=[dict(status='described') for _ in range(4)]; live_rows=[dict(status='described') for _ in range(4)]
    ad=[payload() for _ in range(4)]; ld=[payload(False) for _ in range(4)]
    class Reader:
        bindings={}
        def json(self,path,sha): return dict(file_version={'version':1})
        def finish(self): return None
    monkeypatch.setattr(m,'collect_archive',lambda p:(archive_rows,ad))
    monkeypatch.setattr(m,'collect_live',lambda *args:(live_rows,ld))
    monkeypatch.setattr(m,'file_version',lambda p:{'version':1})
    r=m.build('archive',m.ARCHIVE_SHA,'physical',m.PHYSICAL_SHA,Reader())
    assert r['status']=='fixed_input_statistics_described_not_admitted' and r['comparison'] is not None
    assert r['denominators']['archive_expected']==r['denominators']['live_expected']==4
    assert not r['P1_admission'] and not r['training_provenance_verified']
    ld[2]=None; live_rows[2]=dict(status='missing_or_invalid',error='missing selected RGB')
    r=m.build('archive',m.ARCHIVE_SHA,'physical',m.PHYSICAL_SHA,Reader())
    assert r['status']=='partial_or_failed' and r['pooled']['live'] is None and r['comparison'] is None
    assert r['denominators']['live_described']==3 and r['denominators']['live_expected']==4


def test_wrong_fixed_source_sha_refused_before_read():
    class Reader:
        def json(self,*args): raise AssertionError('must not read')
    with pytest.raises(ValueError): m.build('archive','0'*64,'physical',m.PHYSICAL_SHA,Reader())


def test_writer_floor_no_disk_writes(monkeypatch):
    monkeypatch.setattr(m.shutil,'disk_usage',lambda p:SimpleNamespace(free=m.FLOOR-1))
    with pytest.raises(ValueError): m.write_new('/RAM/not_written.json',{})


def test_no_model_env_producer_imports_or_fits():
    import ast
    tree=ast.parse(Path(m.__file__).read_text())
    names=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    names += [x.name for n in ast.walk(tree) if isinstance(n,ast.Import) for x in n.names]
    assert not any(k in name for name in names for k in ('torch','gym','sklearn','spectral_reachability','stable_worldmodel'))
    c=m.protocol(); assert c['neural_calls']==c['physics_calls']==0
    assert c['no_fit'] and c['no_selection'] and not c['overlay_intervention'] and not c['P1_admission']
