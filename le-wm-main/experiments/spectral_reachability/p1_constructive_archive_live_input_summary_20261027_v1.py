"""Fixed archived4/live4 input-description experiment. No learning or model/env imports.

Only explicit SHA-bound saved inputs. No outcomes, privileged payloads, fitting,
selection, overlay intervention, training-provenance inference or P1 admission.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import numpy as np
from PIL import Image

NAME='p1_constructive_archive_live_input_summary_20261027_v1'
ROOT=Path(__file__).resolve().parents[2]
DATA=Path('/home/gbk/.stable_worldmodel/datasets/pusht_expert_train.h5')
ARCHIVE_SHA='580dd9e35431423885041da76bbdbca3a033808dae7db40aabbaff25e1c34adb'
PHYSICAL_SHA='d205850e95f639b649fba9b5c6ee8059f2507431753a538b1ce30eccbd234300'
EPISODES=(0,6228,12456,18684)
ARMS=('line_search_GN','full10_conditional_D2P')
LIVE_SLOTS=tuple((i,a) for i in range(2) for a in ARMS)
STEPS=(0,5,10,15,20)
BINS=np.arange(0,257,16,dtype=np.int64)
CAP,FLOOR=512*1024,200*1024**2


def need(ok,message):
    if not ok: raise ValueError(message)


def canonical(x): return (json.dumps(x,sort_keys=True,allow_nan=False)+'\n').encode()
def digest(b): return hashlib.sha256(b).hexdigest()
def array_id(a):
    a=np.ascontiguousarray(a)
    return dict(dtype=a.dtype.str,shape=list(a.shape),sha256=digest(a.tobytes()))


def decode(b):
    def unique(items):
        d={}
        for k,v in items: need(k not in d,'duplicate JSON key'); d[k]=v
        return d
    value=json.loads(b,object_pairs_hook=unique); canonical(value); return value


def file_version(path):
    path=Path(path); need(path.is_file() and not path.is_symlink(),'regular nonsymlink dataset')
    s=path.stat()
    return {k:int(getattr(s,k)) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')}


class Reader:
    """Bounded explicit small files only; never opens NPZ, witness or sourcepack."""
    def __init__(self,load=None): self.load=load or self.disk; self.bindings={}; self.bytes=0
    @staticmethod
    def disk(path):
        p=Path(path)
        need(p.is_file() and not p.is_symlink() and p.stat().st_size<=8*1024**2,'bounded nonsymlink input')
        return p.read_bytes()
    def raw(self,path,expected):
        path=Path(path).absolute()
        need(path.suffix in ('.json','.jsonl','.png') and not any(k in path.name.lower() for k in ('witness','certificate','sourcepack')),'only declared report/journal/PNG')
        need(isinstance(expected,str) and len(expected)==64 and all(c in '0123456789abcdef' for c in expected),'explicit SHA256')
        b=self.load(str(path)); self.bytes+=len(b)
        need(len(b)<=8*1024**2 and self.bytes<=64*1024**2 and digest(b)==expected,'bounded saved SHA '+str(path))
        need(str(path) not in self.bindings or self.bindings[str(path)]==expected,'same path identity')
        self.bindings[str(path)]=expected; return b
    def json(self,path,expected): return decode(self.raw(path,expected))
    def png(self,parent,receipt):
        name=receipt['png']; need(Path(name).name==name and name not in ('','.','..'),'flat PNG child')
        data=self.raw(Path(parent).absolute().parent/name,receipt['file_sha256'])
        need(len(data)<=1024**2,'PNG1MiB cap')
        with Image.open(BytesIO(data)) as im:
            need(im.format=='PNG' and im.mode=='RGB' and im.size==(224,224),'unaltered RGB224 PNG')
            rgb=np.array(im)
        image(rgb); need(digest(rgb.tobytes())==receipt['pixel_sha256'],'decoded pixel SHA')
        return rgb
    def finish(self):
        for p,h in self.bindings.items(): need(digest(self.load(p))==h,'saved input changed '+p)


def image(a): need(isinstance(a,np.ndarray) and a.dtype==np.uint8 and a.shape==(224,224,3),'uint8 RGB224')


def pixels(frames,steps):
    need(len(frames)==len(steps) and len(frames)>0,'explicit frame denominator')
    for a in frames: image(a)
    flat=np.stack(frames).reshape(-1,3); f=flat.astype(np.float64)
    counts=np.stack([np.histogram(flat[:,c],bins=BINS)[0] for c in range(3)])
    return dict(frame_count=len(frames),pixels_per_channel=len(flat),steps=list(steps),
        channels='RGB',mean=f.mean(axis=0).tolist(),std_population=f.std(axis=0).tolist(),
        histogram_edges=BINS.tolist(),histogram_counts=counts.tolist(),histogram_fraction=(counts/len(flat)).tolist(),
        frames=[dict(step=s,array=array_id(a)) for s,a in zip(steps,frames,strict=True)],
        adjacent_changes=[dict(from_step=s,to_step=t,gap_public_steps=t-s,
            mean_abs_rgb_delta=float(np.mean(np.abs(a.astype(float)-b.astype(float)))),
            changed_pixel_fraction=float(np.mean(np.any(a!=b,axis=-1))))
            for a,b,s,t in zip(frames[:-1],frames[1:],steps[:-1],steps[1:],strict=True)])


def actions(raw):
    need(isinstance(raw,np.ndarray) and raw.dtype==np.float32 and raw.ndim==2 and raw.shape[1]==2 and len(raw)>1 and len(raw)%5==0 and np.isfinite(raw).all(),'finite raw Nx2 float32, five-step packets')
    a=raw.astype(float); norms=np.linalg.norm(a,axis=1); changes=np.linalg.norm(np.diff(a,axis=0),axis=1)
    return dict(array=array_id(raw),public_action_count=len(raw),packet_count=len(raw)//5,
        mean=a.mean(axis=0).tolist(),std_population=a.std(axis=0).tolist(),coordinate_min=a.min(axis=0).tolist(),coordinate_max=a.max(axis=0).tolist(),
        magnitude_l2=dict(mean=float(norms.mean()),rms=float(np.sqrt(np.mean(norms**2))),maximum=float(norms.max())),
        command_change_per_public_step=dict(count=len(changes),mean_l2=float(changes.mean()),maximum_l2=float(changes.max())),
        packet_raw_l2=np.linalg.norm(a.reshape(-1,10),axis=1).tolist(),coordinates_abs_gt_one=int(np.count_nonzero(np.abs(a)>1)),
        physical_speed=None,units='stored raw action; command delta is not measured object/agent velocity',clip=False,fit=False)


def repeats(entries):
    need(len(entries)>0,'nonempty repeated-position denominator')
    keys=[(a.dtype.str,tuple(a.shape),digest(np.ascontiguousarray(a).tobytes())) for a in entries]
    pairs=[(i,j) for i in range(len(keys)) for j in range(i+1,len(keys))]
    equal=[(i,j) for i,j in pairs if keys[i]==keys[j]]
    return dict(positions=len(keys),unique_entries=len(set(keys)),redundant_positions=len(keys)-len(set(keys)),
        pair_denominator=len(pairs),equal_pairs=[list(x) for x in equal],
        equal_pair_fraction=len(equal)/len(pairs) if pairs else None,
        adjacent_denominator=len(keys)-1,adjacent_equal=sum(a==b for a,b in zip(keys[:-1],keys[1:],strict=True)),
        semantics='exact dtype/shape/bytes equality only; NOT physical state or independent-lineage identity')


def context(frames,packets,scope):
    need(len(frames)==3 and packets.shape==(3,5,2),'states3 / historical2+outgoing packet3')
    return dict(scope=scope,rgb=repeats(frames),packet=repeats(list(packets)),
        rgb_arrays=[array_id(a) for a in frames],packet_arrays=[array_id(a) for a in packets])


def summarize_archive(d):
    f,a=d['frames'],d['actions']; need(len(f)==5 and a.shape==(20,2),'archive fixed five frames/raw20')
    return dict(common_relative_rgb_0_5_10=pixels(f[:3],STEPS[:3]),archive_only_rgb_0_to_20=pixels(f,STEPS),
        common_raw_0_to_10=actions(a[:10]),archive_only_raw_0_to_20=actions(a),
        W10=context(f[:3],a[:15].reshape(3,5,2),'actual observed W10 input; outgoing10:15'),
        W15=context(f[1:4],a[5:20].reshape(3,5,2),'actual observed W15 input; outgoing15:20'))


def summarize_live(d):
    f,h,a,w=d['plan_frames'],d['history_frames'],d['actions'],d['warmup']
    need(len(f)==len(h)==3 and a.shape==w.shape==(10,2),'live fixed history/plan triplets + raw10 each')
    return dict(common_relative_rgb_0_5_10=pixels(f,(0,5,10)),common_raw_0_to_10=actions(a),warmup_raw=actions(w),
        root_observed_context=context(h,np.concatenate([w,a[:5]]).reshape(3,5,2),'real root history warmup0/5/10 + first outgoing plan0:5'),
        postroot_rgb_triplet_repetition=repeats(f),postroot_context_consumed_by_planner=False,
        postroot_context_scope='observed evaluation frames; original planner was open-loop; NOT feedback model inputs')


def archive_slice(f,record,index):
    need(type(index) is int and index in EPISODES,'fixed archive ordinal')
    need(len(f['ep_len'])==18685 and f['ep_offset'].shape==f['ep_len'].shape==(18685,),'known source metadata')
    for k in ('ep_len','ep_offset','episode_idx','step_idx'): need(f[k].dtype.kind in 'iu','integer time metadata')
    n=f['pixels'].shape[0]; off=int(f['ep_offset'][index]); length=int(f['ep_len'][index])
    end=int(f['ep_offset'][index+1]) if index<18684 else n
    need(off>=0 and length>=21 and off+length==end<=n,'complete fixed window/end boundary')
    need((index==0 and off==0) or (index>0 and int(f['ep_offset'][index-1])+int(f['ep_len'][index-1])==off),'previous boundary')
    need(f['pixels'].shape[1:]==(224,224,3) and f['pixels'].dtype==np.uint8 and f['action'].shape==(n,2) and f['action'].dtype==np.float32,'native storage')
    for k in ('episode_idx','step_idx'): need(f[k].shape==(n,),'same row alignment')
    ids=np.asarray(f['episode_idx'][off:off+21]); steps=np.asarray(f['step_idx'][off:off+21])
    need(np.array_equal(ids,np.full(21,index)) and np.array_equal(steps,np.arange(21)),'same episode/time')
    a=np.asarray(f['action'][off:off+20]); frames=np.asarray(f['pixels'][[off+s for s in STEPS]])
    need(a.shape==(20,2) and np.isfinite(a).all(),'literal finite expert raw; no ball rejection')
    need(record['episode_index']==index and record['offset']==off and record['length']==length and record['next_or_end_offset']==end and
        record['frame_local_indices']==list(STEPS) and record['frame_global_indices']==[off+s for s in STEPS] and
        record['action_local_slice']==[0,20] and record['action_global_slice']==[off,off+20],'sealed exact slices')
    need(record['episode_idx']==array_id(ids) and record['step_idx']==array_id(steps) and record['actions']==array_id(a),'sealed selected array bytes')
    need(np.array_equal(np.asarray(record['raw_actions']),a),'literal raw recipe, no clipping')
    need(len(record['frames'])==5,'all five sealed frame entries')
    for s,r,rgb in zip(STEPS,record['frames'],frames,strict=True):
        image(rgb); need(r['step']==s and r['pixels']==array_id(rgb),'sealed decoded pixels')
    return dict(actions=a,frames=frames)


def collect_archive(prepared,read_h5=None):
    need(prepared['schema']=='p1_constructive_offline_fidelity_cuda_20261027_v2.prepare' and prepared['episode_indices']==list(EPISODES), 'fixed archived prepare')
    need(Path(prepared['dataset'])==DATA and len(prepared['episodes'])==4,'fixed existing dataset')
    before=file_version(DATA); need(before==prepared['file_version'],'same archived dataset stat version')
    if read_h5 is None:
        import h5py
        import hdf5plugin  # installed local codec only
        read_h5=lambda:h5py.File(DATA,'r',rdcc_nbytes=4*1024**2)
    rows=[]; data=[]
    with read_h5() as f:
        for i,record in zip(EPISODES,prepared['episodes'],strict=True):
            try:
                d=archive_slice(f,record,i); stats=summarize_archive(d)
                rows.append(dict(episode=i,status='described',statistics=stats)); data.append(d)
            except (Exception,KeyboardInterrupt) as e:
                rows.append(dict(episode=i,status='missing_or_invalid',statistics=None,error=str(e))); data.append(None)
    need(file_version(DATA)==before,'dataset stat stable during fixed slice reads')
    return rows,data


def journal(data):
    need(data.endswith(b'\n'),'complete journal lines')
    events=[decode(b) for b in data.splitlines()]
    need(events and events[0]['kind']=='run_started' and events[-1]['kind']=='run_finished','terminal saved journal')
    need([e['index'] for e in events]==list(range(len(events))),'ordered journal identities')
    return events


def event_actions(events,branch,root,arm,selection,phase):
    attempts=[e for e in events if e['kind']=='operation_attempted' and e.get('resource')=='public_step' and e.get('branch')==branch and e.get('phase')==phase]
    returned=[e for e in events if e['kind']=='returned_transition' and e.get('branch')==branch and e.get('phase')==phase]
    need([e['step'] for e in attempts]==list(range(1,11)) and [e['step'] for e in returned]==list(range(1,11)),'all10 real action/return positions; no replacements')
    values=[]
    for attempt,result in zip(attempts,returned,strict=True):
        for e in (attempt,result): need((e['root_slot'],e['arm'],e['selection_digest'])==(root,arm,selection),'own branch input identity')
        raw=np.asarray(attempt['raw_action']); a=raw.astype(np.float32)
        need(raw.shape==(2,) and np.isfinite(raw).all() and np.array_equal(raw,a),'literal float32 executed command')
        if phase=='plan': need(np.array_equal(result['command'],a),'attempt/return action agreement')
        values.append(a)
    return np.stack(values)


def live_branch(reader,report_path,row,events,branch):
    root,arm=LIVE_SLOTS[branch]; selection=row['selection_digest']
    need((row['root_slot'],row['arm'])==(root,arm) and row['initial_exact'] is True and row['warmup_returned']==row['own_returned']==10 and row['full_horizon'] is True,'complete original fixed branch')
    need(row['first_error'] is None and row['secondary_errors']==[],'no branch input fault')
    plan_events=[e for e in events if e['kind']=='fresh_observation' and e.get('branch')==branch and e['receipt']['phase']=='plan' and e['receipt']['step'] in (0,5,10)]
    warm_events=[e for e in events if e['kind']=='fresh_observation' and e.get('branch')==branch and e['receipt']['phase']=='warmup' and e['receipt']['step'] in (0,5,10)]
    for selected in (plan_events,warm_events):
        need([e['receipt']['step'] for e in selected]==[0,5,10],'exact fixed PNG steps')
        for e in selected: need((e['root_slot'],e['arm'],e['selection_digest'])==(root,arm,selection),'own RGB branch identity')
    fields=('phase','step','png','file_sha256','pixel_sha256')
    need([{k:r[k] for k in fields} for r in row['checkpoints']]==[{k:e['receipt'][k] for k in fields} for e in plan_events],'report/journal PNG agreement')
    frames=[reader.png(report_path,e['receipt']) for e in plan_events]
    history=[reader.png(report_path,e['receipt']) for e in warm_events]
    need(np.array_equal(history[-1],frames[0]),'post-bind plan0 equals real history end, not future reuse')
    warmup=event_actions(events,branch,root,arm,selection,'warmup')
    need(not warmup.any(),'original real zero warmup, NOT synthetic padding')
    commands=event_actions(events,branch,root,arm,selection,'plan')
    return dict(plan_frames=frames,history_frames=history,actions=commands,warmup=warmup)


def collect_live(reader,path,report):
    need(report['schema']=='p1_constructive_plan_physical_20261027_v1.report' and report['errors']==[],'original physical report interface')
    need(report['contract']['with_target'] is False and report['contract']['branches']==4 and report['contract']['own_horizon']==10 and report['contract']['warmup']==10,'original target-free four-branch protocol')
    need([(r['root_slot'],r['arm']) for r in report['rows']]==list(LIVE_SLOTS),'fixed four branches; no outcome sorting/filtering')
    ev=journal(reader.raw(Path(path).absolute().parent/'journal.jsonl',report['journal_sha256']))
    rows=[]; data=[]
    for b,((root,arm),row) in enumerate(zip(LIVE_SLOTS,report['rows'],strict=True)):
        try:
            d=live_branch(reader,path,row,ev,b)
            rows.append(dict(root_slot=root,arm=arm,selection_digest=row['selection_digest'],status='described',statistics=summarize_live(d))); data.append(d)
        except (Exception,KeyboardInterrupt) as e:
            rows.append(dict(root_slot=root,arm=arm,status='missing_or_invalid',statistics=None,error=str(e))); data.append(None)
    return rows,data


def pooled(data,key):
    need(len(data)==4 and all(d is not None for d in data),'no complete-case or favorable-subset pooling')
    frames=[frame for d in data for frame in (d['frames'][:3] if key=='archive' else d['plan_frames'])]
    p=pixels(frames,list(range(12)))
    # Pooled frames have no temporal adjacency across episodes or branches.
    p.pop('adjacent_changes'); p.pop('steps'); p.pop('frames')
    raw=np.concatenate([d['actions'][:10] for d in data])
    a=actions(raw); a.pop('command_change_per_public_step')  # never invent cross-branch time transitions
    deltas=np.concatenate([np.linalg.norm(np.diff(d['actions'][:10].astype(float),axis=0),axis=1) for d in data])
    a['within_sequence_command_change']=dict(count=len(deltas),mean_l2=float(deltas.mean()),maximum_l2=float(deltas.max()))
    return dict(pixels=p,actions=a,sequence_count=4,frame_denominator=12,action_denominator=40,
        independent_sample_count=None,weighting='four fixed sequences equally; each3 frames/10 actions; not iid')


def cross_repetition(archive,live):
    need(all(d is not None for d in archive+live),'all fixed rows for cross repetition')
    sets=dict(archive_common=[a for d in archive for a in d['frames'][:3]],
        live_common=[a for d in live for a in d['plan_frames']],live_root_history=[a for d in live for a in d['history_frames']],
        archive_W10_whole_rgb_history=[np.stack(d['frames'][:3]) for d in archive],
        live_root_whole_rgb_history=[np.stack(d['history_frames']) for d in live],
        archive_W10_whole_raw_history=[d['actions'][:10] for d in archive],
        live_root_whole_raw_history=[d['warmup'] for d in live])
    return {k:repeats(v) for k,v in sets.items()}


def protocol():
    return dict(archive_episodes=list(EPISODES),live_slots=[list(x) for x in LIVE_SLOTS],
        common_frames=[0,5,10],archive_only_extra_frames=[15,20],common_raw_slice=[0,10],archive_only_extra_actions=[10,20],
        histogram_edges=BINS.tolist(),pixel_units='raw uint8 0..255, no model preprocessing or overlay removal',
        timeline='archive relative episode steps vs live relative plan steps; not paired physical states or matched control phases',
        context='compare actual archiveW10 with live real warmup history; postroot live RGB is evaluator-only, not planner feedback',
        lineage='live4 branches from TWO roots, not4 independent demonstrations; archive4 ordinal lineage unknown; no dedup or iid inference',
        provenance='selected-source descriptive comparison only; NOT checkpoint training membership/provenance',
        no_fit=True,no_selection=True,no_training=True,neural_calls=0,physics_calls=0,overlay_intervention=False,
        P1_admission=False,causal_claim=False,independent_confirmation=False,output_cap_bytes=CAP,disk_floor_bytes=FLOOR)


def build(archive_prepared,archive_sha,physical_report,physical_sha,reader=None):
    need(archive_sha==ARCHIVE_SHA and physical_sha==PHYSICAL_SHA,'this experiment fixes existing archive/live report identities')
    reader=reader or Reader(); prepared=reader.json(archive_prepared,archive_sha); report=reader.json(physical_report,physical_sha)
    archive,ad=collect_archive(prepared); live,ld=collect_live(reader,physical_report,report)
    complete=all(r['status']=='described' for r in archive+live)
    groups={k:pooled(d,k) if all(x is not None for x in d) else None for k,d in (('archive',ad),('live',ld))}
    comparison=None
    if complete:
        ap,lp=groups['archive']['pixels'],groups['live']['pixels']
        comparison=dict(live_minus_archive_rgb_mean=(np.asarray(lp['mean'])-ap['mean']).tolist(),
            fixed_histogram_total_variation_per_channel=(.5*np.abs(np.asarray(lp['histogram_fraction'])-ap['histogram_fraction']).sum(axis=1)).tolist(),
            live_minus_archive_raw_magnitude_mean=groups['live']['actions']['magnitude_l2']['mean']-groups['archive']['actions']['magnitude_l2']['mean'],
            interpretation='descriptive marginal differences; no fitted decision rule, threshold, p-value or causal attribution')
    reader.finish()
    need(file_version(DATA)==prepared['file_version'],'final archived dataset version stable')
    paths=[Path(__file__),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]
    return dict(schema=NAME+'.report',status='fixed_input_statistics_described_not_admitted' if complete else 'partial_or_failed',
        protocol=protocol(),archive=archive,live=live,pooled=groups,comparison=comparison,
        repetition=cross_repetition(ad,ld) if complete else None,
        denominators=dict(archive_expected=4,archive_described=sum(d is not None for d in ad),live_expected=4,live_described=sum(d is not None for d in ld),
            archive_common_frames_expected=12,live_common_frames_expected=12,archive_all_frames_expected=20),
        bindings=reader.bindings,dataset=dict(path=str(DATA),version=prepared['file_version'],identity_scope='stat + exact selected slices, NOT whole HDF5 SHA'),
        source_sha256={str(p):digest(p.read_bytes()) for p in paths},
        versions={k:importlib.metadata.version(k) for k in ('numpy','pillow')},errors=[dict(group=g,row=i,error=r['error']) for g,rs in (('archive',archive),('live',live)) for i,r in enumerate(rs) if 'error' in r],
        P1_admission=False,training_provenance_verified=False,new_model_or_physics_run=False)


def write_new(output,report):
    raw=canonical(report); output=Path(output).absolute()
    need(not output.exists() and len(raw)<=CAP and shutil.disk_usage(output.parent).free>=FLOOR+len(raw),'NEW small JSON and200MiB reserve')
    with output.open('xb') as f: f.write(raw); f.flush(); os.fsync(f.fileno())
    fd=os.open(output.parent,os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)
    return dict(output=str(output),sha256=digest(raw),bytes=len(raw),status=report['status'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('archive-prepared','archive-sha','physical-report','physical-sha','output'): p.add_argument('--'+key,required=True)
    a=p.parse_args(); r=build(a.archive_prepared,a.archive_sha,a.physical_report,a.physical_sha)
    print(canonical(write_new(a.output,r)).decode(),end='')
    if r['status']=='partial_or_failed': raise SystemExit(2)


if __name__=='__main__': main()
