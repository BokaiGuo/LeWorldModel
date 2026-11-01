"""Saved consumer v2: correct baseline/anchor clock, no new native execution."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib
import json
from pathlib import Path
import shutil

import numpy as np

TAG='agent_p5_live_raster_saved_consumer_20261028_v2'
ROOT=Path(__file__).resolve().parents[2]
RENDERER='agent_p5_live_raster_intervention_20261028_v1'
RENDERER_SHA='34876d0624373416b415dc97b4938364de4eb3156985d5df851461e665b73e84'
_renderer_path=ROOT/'experiments/spectral_reachability'/f'{RENDERER}.py'
if hashlib.sha256(_renderer_path.read_bytes()).hexdigest()!=RENDERER_SHA:
    raise ValueError('sealed renderer source')
renderer=importlib.import_module('experiments.spectral_reachability.'+RENDERER)
NAME=renderer.NAME
h=renderer.h
STATES,COLORS,SHOTS=renderer.STATES,renderer.COLORS,renderer.SHOTS
EXPECTED,DELTAS=renderer.EXPECTED,renderer.DELTAS
require,canonical,digest=renderer.require,renderer.canonical,renderer.digest
protocol,equal,differences=renderer.protocol,renderer.equal,renderer.differences


def read(*args):return renderer.read(*args)
def sha(*args):return renderer.sha(*args)
def verify(*args):return renderer.verify(*args)
def files_set(r):return set(r['artifacts'])


def source_seals():
    paths=(Path(__file__).resolve(),ROOT/f'tests/test_{TAG}.py',ROOT/f'innerdocs/{TAG}.md',_renderer_path)
    return {str(p):sha(p) for p in paths}


def constructor_rng_receipt(before,after):
    """Replay only the documented original seed(None) integer draw in a local RNG."""
    require(isinstance(before,list) and len(before)==5 and before[0]=='MT19937','complete original MT19937 state')
    words=before[1]
    require(set(words)=={'dtype','shape','values'} and words['dtype']=='<u4' and words['shape']==[624]
        and len(words['values'])==624 and all(type(x) is int and 0<=x<2**32 for x in words['values']), 'exact624 uint32 words')
    require(type(before[2]) is int and 0<=before[2]<=624 and type(before[3]) is int and before[3] in (0,1)
        and type(before[4]) in (float,int) and np.isfinite(before[4]),'complete RNG position/cache flags')
    local=np.random.RandomState(0)
    local.set_state((before[0],np.asarray(words['values'],np.uint32),before[2],before[3],before[4]))
    draw=int(local.randint(0,25536))
    s=local.get_state()
    predicted=[s[0],dict(dtype=s[1].dtype.str,shape=list(s[1].shape),values=s[1].tolist()),int(s[2]),int(s[3]),float(s[4])]
    require(equal(predicted,after),'one exact ctor randint must explain full624 words/position/cache')
    return dict(operation='np.random.randint(0,25536)',draw=draw,full_state_exact=True,
        before_digest=digest(before),predicted_after_digest=digest(predicted),actual_after_digest=digest(after),
        production_rng_modified=False)


def validate_anchor(baseline,current,state_id):
    fields={'public_state','global_rng','topology','goal_color'}
    require(set(baseline)==set(current)==fields,'complete baseline/current snapshots')
    for k in ('public_state','topology','goal_color'):
        require(equal(baseline[k],current[k]),'baseline/current '+k+' must remain exact')
    require(set(baseline['global_rng'])==set(current['global_rng'])=={'python','numpy'},'full global RNG schema')
    require(equal(baseline['global_rng']['python'],current['global_rng']['python']),'global Python RNG stays exact')
    if state_id=='goal':
        require(equal(baseline,current),'goal baseline/current full RNG exact')
        rng_receipt=dict(operation='none_after_goal_baseline',full_state_exact=True,production_rng_modified=False)
    else:
        require(state_id in STATES[:4],'fixed original word id')
        rng_receipt=constructor_rng_receipt(baseline['global_rng']['numpy'],current['global_rng']['numpy'])
    diff=differences(baseline,current)
    require(all(x['pointer'].startswith('/global_rng/') for x in diff),'only inter-phase global RNG differences')
    return dict(exact_differences=diff,constructor_rng=rng_receipt)


def consume(path,expected,report_path,report_sha):
    """Saved-only independent byte/PNG/dispatch/state rederivation; no physics."""
    from PIL import Image
    p=verify(path,expected);r=read(report_path,report_sha);out=Path(report_path).absolute().parent
    require(r['schema']==NAME+'.report' and r['prepare_sha256']==expected and r['prepare_digest']==digest(p) and r['protocol']==protocol(),'execution identity')
    require(r['status']=='live_raster_recorded' and not r['errors'] and r['all_states_attained'] and r['profiler_released'],'complete fresh raster only')
    require(r['old_failure']==p['old_failure'] and r['old_renderer_release'] is False and r['historical_full_public_equivalence'] is None and r['scientific_admission'] is False,'old failure and claim boundaries')
    files={str(f.relative_to(out)) for f in out.rglob('*') if f.is_file()}
    require(all(not f.is_symlink() for f in out.rglob('*')) and files==set(r['artifacts'])|{'report.json'},'exclusive complete artifact census')
    require(sum((out/n).stat().st_size for n in files)<=h.CAP and 0<=r['wall_seconds']<=h.WALL,'actual job caps')
    for name,rec in r['artifacts'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts and Path(name).as_posix()==name,'local artifact path')
        require(rec==dict(path=name,sha256=sha(out/name),bytes=(out/name).stat().st_size),'actual artifact SHA/size')
    require(read(out/'execution_binding.json')==dict(schema=NAME+'.execution',prepare_sha256=expected,prepare_digest=digest(p),protocol=protocol()),'saved execution edge')
    v3=h.load_bound(h.V3,h.V3_SHA);v=read(p['v3_prepare'],p['v3_sha256'])
    v3.check_origins(v,*[v['origins'][k] for k in ('environment','pymunk','pygame')],read(out/'runtime_probe.json'))
    events=[];previous=None
    for i,line in enumerate((out/'journal.jsonl').read_bytes().splitlines()):
        x=json.loads(line);hh=x.pop('entry_sha256')
        require(x['sequence']==i and x['previous_sha256']==previous and digest(x)==hh,'journal hashchain')
        previous=hh;events.append(x['event'])
    require(len(events)==r['journal_events'] and previous==r['journal_terminal_sha256'] and events[-1]==dict(phase='final',errors=[]),'journal terminal')
    allowed={'intent','returned','failed','native_call','native_return','frame_saved','all44_prefixes_exact','all5_states_attained','final'}
    require(all(e['phase'] in allowed for e in events),'known journal phases')
    for phase in ('native_call','native_return'):
        require(dict(Counter(e['name'] for e in events if e['phase']==phase))==EXPECTED,'native actual entry/return counts')
    require(r['meter']==dict(entries=EXPECTED,return_events=EXPECTED,errors=[]),'native receipt')
    require(all(e['dt']==.01 for e in events if e['phase']=='native_call' and e['name']=='space_step'),'605 dt exact')
    pending=None;ni=Counter();no=Counter();dispatch=[];frames=[];prefix=False;attained=False
    for e in events:
        phase=e['phase']
        if phase=='intent':
            require(pending is None,'nonoverlap public dispatch');pending=(e['state_id'],e['operation']);ni.clear();no.clear()
            s,op=pending;details=e['details'];dispatch.append((s,op,details.get('frame'),details.get('shot')))
            require(s in STATES,'known state id');config=p['goal_protocol'] if s=='goal' else p['plans'][STATES.index(s)]
            if op=='construct':require(details==dict(constructor=config['constructor']),'original constructor')
            elif op=='seed':require(details==dict(seed=config['seed']),'original explicit seed')
            elif op=='step':
                require(details==read(out/s/f"command_{details['frame']:02d}.json"),'durable command dispatch')
                require(details['frame']<=10 or prefix,'all prefix gate before suffix')
            elif op in ('color','restore_render'):require(attained,'all5 before color')
        elif phase in ('native_call','native_return'):
            require(pending==(e['context']['state_id'],e['context']['operation']),'native public operation context')
            (ni if phase=='native_call' else no)[e['name']]+=1
        elif phase=='returned':
            require(pending==(e['state_id'],e['operation']),'paired actual return')
            require(dict(ni)==DELTAS[pending[1]] and dict(no)==DELTAS[pending[1]],'per-dispatch native delta');pending=None
        elif phase=='failed':require(False,'failed public operation')
        elif phase=='frame_saved':frames.append((e['state_id'],e['frame'],e['exact']))
        elif phase=='all44_prefixes_exact':
            require(not prefix and [(s,n) for s,n,_ in frames]==[(s,n) for s in STATES[:4] for n in range(11)]
                and all(all(x.values()) for _,_,x in frames),'44 exact prefix frames');prefix=True
        elif phase=='all5_states_attained':
            require(not attained and len(frames)==65 and prefix,'all65 before intervention');attained=True
    wanted=[]
    for s in STATES[:4]:
        wanted.extend((s,k,None,None) for k in ('construct','seed','reset'));wanted.extend((s,'step',n,None) for n in range(1,11))
    wanted.extend((s,'step',n,None) for s in STATES[:4] for n in range(11,16))
    wanted.extend(('goal',k,None,None) for k in ('construct','seed','reset'))
    wanted.extend((s,k,None,shot) for s in STATES for k,shot in (('color','visible'),('restore_render','visible_restored'),('color','white'),('restore_render','white_restored')))
    wanted.extend((s,'close',None,None) for s in STATES)
    require(pending is None and dispatch==wanted and prefix and attained,'fixed entire operation chronology')
    require(Counter(x[1] for x in dispatch)==Counter(protocol()['public_operations']),'public attempt/return census')
    def file(rec):
        require(rec==r['artifacts'][rec['path']],'artifact edge')
        return out/rec['path']
    def image(rec):
        arr=np.load(file(rec['npy']),allow_pickle=False)
        require(arr.shape==(224,224,3) and arr.dtype==np.uint8,'native RGB224')
        with Image.open(file(rec['png'])) as im:require(im.mode=='RGB' and np.array_equal(np.asarray(im),arr),'PNG decoded pixels')
        require(rec['pixel_sha256']==hashlib.sha256(arr.tobytes()).hexdigest(),'pixel SHA')
        return arr
    require([x['state_id'] for x in r['rows']]==list(STATES),'five fixed rows')
    encoded=[];allframes=[];comparisons=[];baseline_diffs=[]
    for row,target in zip(r['rows'],p['targets']):
        s=row['state_id'];nmax=target['frame'];ns=list(range(nmax+1))
        require(row['status']=='completed' and [x['frame'] for x in row['records']]==ns,'full planned frames')
        for rec in row['records']:
            n=rec['frame'];image(rec['image']);require(rec['image']['npy']==rec['files']['rgb.npy'],'native image edge')
            delta={};exact={}
            for name in h.NAMES:
                fr=rec['files'][name];require(fr['path']==s+f'/frame_{n:02d}/'+name,'exact original frame identity')
                src=p['frames'][s][n][name];exact[name]=file(fr).read_bytes()==Path(src['path']).read_bytes()
                if name.endswith('.json'):delta[name]=differences(read(src['path'],src['sha256']),read(file(fr)))
            require(exact==rec['exact'] and equal(delta,read(file(rec['differences']))),'durable complete exact source differences')
            require(all(exact[k] for k in (h.NAMES if n<=10 else h.NAMES[:3])),'original prefix/suffix byte gates')
            allframes.append((s,n,exact))
            comparisons.append(dict(state_id=s,frame=n,exact=exact,differences=rec['differences']))
            if n:
                raw=np.asarray((p['plans'][STATES.index(s)]['prefix_raw_u']+p['plans'][STATES.index(s)]['reference_suffix_raw_u'])[n-1],np.float64)
                before=read(out/s/f'frame_{n-1:02d}/full_public_state.json');pos=np.asarray(before['bodies']['agent']['position'],np.float64)
                require(equal(read(out/s/f'command_{n:02d}.json'),dict(frame=n,raw_env_argument=raw.tolist(),dtype='<f8',agent_xy_before=pos.tolist(),absolute_target=(pos+100*raw).tolist(),dt=.01,expected_substeps=10)),'original raw commands and absolute targets')
        require(row['baseline']==row['records'][-1] and [x['shot'] for x in row['shots']]==list(SHOTS),'fresh predetermined baseline/shots')
        require([x['color'] for x in row['cells']]==list(COLORS),'two fixed colors')
        baseline=None;anchor=read(file(row['cells'][0]['anchor']))
        for shot in row['shots']:
            require(shot['status']=='captured','all triplet slots present');files=shot['files']
            arr=image(files['rgb_image']);before=read(file(files['before']));after=read(file(files['after']))
            require(equal(before,after),'actual render before/after unchanged')
            if shot['shot']=='baseline':
                require(files['rgb_image']==row['baseline']['image'],'baseline native image reuse')
                baseline=arr
                baseline_diffs.append(dict(state_id=s,baseline_after=files['after'],current_anchor=row['cells'][0]['anchor'],
                    **validate_anchor(after,anchor,s),baseline_snapshot_digest=digest(after),current_anchor_digest=digest(anchor)))
                require(equal(after['public_state'],read(file(row['baseline']['files']['full_public_state.json']))),'baseline actual public state')
                require(after['public_state']['space']['current_time_step']==.01,'actual postreset timestep')
            else:
                require(equal(after['public_state'],anchor['public_state']) and equal(after['global_rng'],anchor['global_rng'])
                    and equal(after['topology'],anchor['topology']),'color only invariant')
                if shot['shot'] in COLORS:
                    require(after['goal_color']['rgba']==h.RGBA[shot['shot']],'actual requested native color')
                    encoded.append(dict(image_key=s+'/'+shot['shot'],state_id=s,color=shot['shot'],
                        rgb=files['rgb_image']['npy'],png=files['rgb_image']['png'],pixel_sha256=files['rgb_image']['pixel_sha256'],
                        before=files['before'],after=files['after'],fresh_snapshot_digest=digest(after),
                        fresh_public_state_digest=digest(after['public_state'])))
                else:require(equal(after,anchor),'original color object fully restored')
            if shot['shot']!='white':require(np.array_equal(arr,baseline),'original-color/contact pixels byte exact')
        for j,cell in enumerate(row['cells']):
            color=COLORS[j];require(cell['status']=='completed' and cell['requested_rgba']==h.RGBA[color],'complete color cell')
            require(cell['off_before_shot']==('baseline' if j==0 else 'visible_restored'),'shared fresh off triplets')
            require(equal(read(file(cell['anchor'])),anchor),'unchanged intervention anchor')
            require(cell['original_rgba']==anchor['goal_color']['rgba'] and cell['original_color_id']==anchor['goal_color']['object_id'],'original color identity')
            require(equal(read(file(cell['restoration'])),dict(snapshot=anchor,original_color_id=anchor['goal_color']['object_id'],same_color_object=True)),'actual restoration receipt')
    # Frame order in journal differs from row-major saved layout (all prefixes first).
    expected_frames=[(s,n) for s in STATES[:4] for n in range(11)]+[(s,n) for s in STATES[:4] for n in range(11,16)]+[('goal',0)]
    by_frame={(s,n):x for s,n,x in allframes}
    require(frames==[(s,n,by_frame[(s,n)]) for s,n in expected_frames],'all65 journal/saved frame joins')
    require(len(encoded)==10 and len([x for x in files_set(r) if x.endswith('.png')])==85,'10 fixed encoder inputs/85 PNG')
    return dict(schema=TAG+'.receipt',status='strict_live_raster_complete_v2',prepare_sha256=expected,report_sha256=report_sha,
        renderer_schema=NAME,consumer_bindings=source_seals(),baseline_to_current_anchor=baseline_diffs,
        states=5,cells=10,physics_steps=605,native_entries=EXPECTED,encoder_inputs=encoded,source_comparisons=comparisons,
        old_failure=p['old_failure'],old_renderer_release=False,historical_full_public_equivalence=None,scientific_admission=False)



def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=('consume','verify'))
    for key in ('prepare','prepare-sha','report','report-sha','output','receipt','receipt-sha'):
        ap.add_argument('--'+key)
    a=ap.parse_args()
    result=consume(a.prepare,a.prepare_sha,a.report,a.report_sha)
    if a.mode=='verify':
        require(equal(read(a.receipt,a.receipt_sha),result),'complete deterministic v2 receipt')
        result=dict(schema=TAG+'.verify',status='saved_v2_receipt_verified',
            receipt_sha256=a.receipt_sha,scientific_admission=False,old_renderer_release=False)
    if a.output:
        data=canonical(result)+b'\n'
        require(len(data)<=2*1024**2,'2MiB receipt cap')
        require(shutil.disk_usage(Path(a.output).absolute().parent).free>=200*1024**2+len(data),'200MiB free floor')
        h.new(a.output,data)
    print(json.dumps(dict(status=result['status'],scientific_admission=False)))
    return 0


if __name__=='__main__':raise SystemExit(main())

