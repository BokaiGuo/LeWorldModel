"""Four native words: saved-only full trajectory/body/callback/RGB analysis.

MAIN actual bounded strict consumer source and persisted exact return required.
No neural/physics execution, fitting, threshold, selection or causal attribution.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np

TAG='agent_p5_contact_word_response_analysis_20261028_v1'
ROOT=Path(__file__).resolve().parents[2]
PHYSICAL='agent_p5_contact_action_words_physical_20261028_v1'
MODEL='agent_p5_contact_action_response_20261028_v1'
ARITHMETIC='agent_p5_word_physics_analysis_20261028_v1'
WORDS=('xy','yx','x_minus_x','zero_zero')
PAIRS=(('xy','yx'),('x_minus_x','zero_zero'))
REPORT_SHA='a07c2583c5d5dd553324d0e5c0b740ffbde3fa7a806798b3a8dcb0fe77ab6633'
MANIFEST_SHA='46f8cf63d045123edfa61b257ad11bd94b3741b47dfe2244463c5c5096df0ab7'
CONSUMER_SHA='8bb51bddb3b2836d43ce5a51e0162b05d3f98ad2aa7f6cb4de23739c8b4fe7d5'
RECEIPT_SHA='f6cd1f53d7537a6292c49647a3e20cfd61caf8a4b5e5e9d395223fc5845a39ee'
MODEL_REPORT_SHA='4fd032a4e5fb365f5a11eecbaba416c1ace7352e7afd092aa0adf6b972bded7f'
MODEL_RECEIPT_SHA='247a8a6328053f34b8367f39dca63f81fb009ca2a41734db0f618d1ea6eab2c3'
MODEL_MANIFEST_SHA='572e2fded1c8927302c4ffe66921ce05e58f5954b43a9bc44bc022c19c6bbfb6'
PINS={ARITHMETIC:'b73f0d43fbc3536fd2027c292c0a94c03f3c74f40f4effc1a7a5efde0d687c95',
      MODEL:'bd5b85b308fcb621adda1f4452911f58e233b9ba0812e8d569d0dbd540940930',
      PHYSICAL:'0a00d0f7864f03fc8e9ddf1fa7cef749a1d57f6c977357ce78c309262aa4556f'}
CAP=8*1024**2


def require(ok,message):
    if not ok: raise ValueError(message)


def canonical(v):
    return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def sha(path):
    p=Path(path)
    require(p.is_file() and not any(x.is_symlink() for x in (p,*p.parents)),'regular nonsymlink input')
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''): h.update(b)
    return h.hexdigest()


def read_json(path,expected=None):
    actual=sha(path)
    require(expected is None or actual==expected,'actual input FILE SHA')
    require(Path(path).stat().st_size<=CAP,'JSON cap')
    def pairs(items):
        out={}
        for k,v in items:
            require(k not in out,'duplicate JSON key')
            out[k]=v
        return out
    return json.loads(Path(path).read_text(),object_pairs_hook=pairs,
        parse_constant=lambda v:require(False,'nonfinite JSON '+v))


def load_bound(path,expected):
    require(sha(path)==expected,'actual consumer/helper source SHA')
    spec=importlib.util.spec_from_file_location(TAG+'_'+Path(path).stem,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def helper(name):
    return load_bound(ROOT/'experiments/spectral_reachability'/(name+'.py'),PINS[name])


def policy():
    return dict(words=list(WORDS),denominator=4,prefix_frame=10,suffix_frames=list(range(11,21)),
        baseline_rows=4,suffix_rows=40,pair_rows=20,pairs=[list(p) for p in PAIRS],
        sign='left minus right, same saved frame',
        units=dict(position='native pixels',velocity='native pixels/s',angle='radians',
            angular_velocity='rad/s',coverage='native fraction',raw_action='relative native u; pixel displacement100*u',
            rgb='raw uint8 intensity; squared intensity for RGB MSE',patch='model-only squared feature units',
            callback_count='actual original callback entries',callback_points='actual accumulated points, not distinct contacts',
            impulse='native Pymunk impulse, not force/work'),
        scientific_admission=False,task_success=None,object_noncommutativity_claim=False,
        latent_to_physical_fit=False,physical_release=False,
        inference='one constructed OOD source, four paired words, not four independent roots',
        interpretation='agent motion and full-image RGB differences alone do not establish object response mechanism')


def record_loader(out,row):
    records={r['frame']:r for r in row['records']}
    require(len(row['records'])==len(records)==21 and set(records)==set(range(21)),'complete actual frame census')
    def load(frame):
        p=Path(out)/row['arm']/f'frame_{frame:02d}'
        rec=records[frame]
        for name in ('rgb.npy','full_public_state.json','transition.json'):
            require(rec['files'][name]==dict(sha256=sha(p/name),bytes=(p/name).stat().st_size),'frame SHA/size')
        require((p/'rgb.npy').stat().st_size<=200000,'RGB cap')
        rgb=np.load(p/'rgb.npy',allow_pickle=False,max_header_size=1024)
        require(rgb.shape==(224,224,3) and rgb.dtype==np.uint8,'raw RGB ABI')
        return read_json(p/'full_public_state.json'),read_json(p/'transition.json'),rgb
    return load


def contacts(events,points):
    require(isinstance(events,list),'actual callback list')
    total=0
    roles={}
    for e in events:
        require(e['returned'] is True and e['counter_before']==total
            and e['counter_after']==total+len(e['points']),'actual callback count continuity')
        require(len(e['body_roles'])==2 and set(e['body_roles'])<=set(('agent','block','static')),'actual callback body identity')
        total=e['counter_after']
        key=':'.join(sorted(e['body_roles']))
        if key not in roles: roles[key]=dict(callback_entries=0,points=0)
        roles[key]['callback_entries']+=1
        roles[key]['points']+=len(e['points'])
    require(total==points,'actual callback/public point sum')
    # Raw impulses and points retained verbatim; do not sum vectors over distinct
    # bodies/substeps or convert them into forces/work/unique contact episodes.
    return dict(callback_entries=len(events),points=total,by_actual_body_pair=roles,events=events)


def assemble(plans,rows,model,load_frame,load_command,load_callbacks,arithmetic):
    require([p['arm'] for p in plans]==[r['arm'] for r in rows]==list(WORDS),'complete ordered four-word denominator')
    require(all(r['status']=='completed' and r['error'] is None for r in rows),'complete actual native branches')
    require(model['status']=='public_native_action_response_recorded' and model['errors']==model['finalization_errors']==[],
            'complete actual model report')
    require(model['words']==[dict(word=w,status='completed',error=None) for w in WORDS],'all four model words')
    for key in ('xy_vs_yx_patch_mse','x_minus_x_vs_zero_patch_mse'):
        require(np.isfinite(model['contrasts'][key]) and model['contrasts'][key]>=0,'finite model-only contrast')
    baseline_rows,frame_rows,pair_rows,summaries=[],[],[],[]
    banks={}
    for plan,row in zip(plans,rows):
        word=plan['arm']
        p,t,rgb=load_frame(row,10)
        baseline=arithmetic.quantities(p,t,10)
        base_callbacks=contacts(load_callbacks(row,10),baseline['callback_contact_points'])
        baseline_rows.append(dict(word=word,frame=10,quantities=baseline,actual_callbacks=base_callbacks))
        raw=np.asarray(plan['reference_suffix_raw_u'])
        require(raw.shape==(10,2) and raw.dtype.kind=='f' and np.isfinite(raw).all(),'full raw10 suffix plan')
        bank={}
        cumulative=np.zeros(2,np.float64)
        for frame in range(11,21):
            command=load_command(row,frame)
            u=np.asarray(command['raw_env_argument'],np.float64)
            require(command['frame']==frame and u.shape==(2,) and np.array_equal(u,raw[frame-11])
                    and command['pixel_displacement']==(100*u).tolist(),'exact model-plan raw command order')
            p,t,image=load_frame(row,frame)
            q=arithmetic.quantities(p,t,frame)
            require(q['native_goal_pose']==baseline['native_goal_pose'],'same native goal')
            require(p['env']['latest_action']==command['absolute_target'],'actual latest action/target')
            callback=contacts(load_callbacks(row,frame),q['callback_contact_points'])
            cumulative+=u
            item=dict(word=word,frame=frame,suffix_step=frame-10,quantities=q,actual_callbacks=callback,
                command=command,cumulative_raw_u=cumulative.tolist(),
                delta_from_own_prefix10=arithmetic.difference(q,baseline),
                rgb_vs_own_prefix10=arithmetic.rgb_difference(image,rgb))
            frame_rows.append(item)
            bank[frame]=(q,image,callback)
        banks[word]=(baseline,rgb,bank)
    # Same native full precision baseline and raw pixels; no independent-root claim.
    for word in WORDS[1:]:
        require(banks[word][0]==banks[WORDS[0]][0]
            and np.array_equal(banks[word][1],banks[WORDS[0]][1]),'shared original baseline identity')
    for left,right in PAIRS:
        paired=[]
        for frame in range(11,21):
            lq,li,lc=banks[left][2][frame]
            rq,ri,rc=banks[right][2][frame]
            item=dict(left_word=left,right_word=right,frame=frame,suffix_step=frame-10,
                delta=arithmetic.difference(lq,rq),rgb=arithmetic.rgb_difference(li,ri),
                callback_entries_left=lc['callback_entries'],callback_entries_right=rc['callback_entries'],
                actual_body_pairs_left=lc['by_actual_body_pair'],actual_body_pairs_right=rc['by_actual_body_pair'])
            pair_rows.append(item); paired.append(item)
        summaries.append(dict(left_word=left,right_word=right,frames=10,
            agent_position_max_l2=max(x['delta']['agent_xy']['l2'] for x in paired),
            agent_velocity_max_l2=max(x['delta']['agent_velocity']['l2'] for x in paired),
            block_position_max_l2=max(x['delta']['block_xy']['l2'] for x in paired),
            block_velocity_max_l2=max(x['delta']['block_velocity']['l2'] for x in paired),
            block_angle_max_abs_principal_rad=max(abs(x['delta']['block_angle']['principal_left_minus_right_rad']) for x in paired),
            any_exact_block_pose_difference=any(not x['delta']['block_xy']['exact_equal']
                or not x['delta']['block_angle']['exact_native_equal'] for x in paired),
            endpoint_delta=paired[-1]['delta'],endpoint_rgb=paired[-1]['rgb'],
            object_noncommutativity_claim=False,task_success=None))
    require(len(baseline_rows)==4 and len(frame_rows)==40 and len(pair_rows)==20,'unfiltered complete frame denominator')
    return dict(baselines=baseline_rows,suffix_frame_rows=frame_rows,pair_frame_rows=pair_rows,
        pair_descriptive_summaries=summaries,model_only_parallel=dict(contrasts=model['contrasts'],meter=model['meter'],
            scope='endpoint feature quantities only; not per-frame physical prediction or object mechanism'))


def analyze(manifest_path,manifest_sha,report_path,report_sha,consumer_source,consumer_sha,receipt_path,receipt_sha):
    require(manifest_sha==MANIFEST_SHA and report_sha==REPORT_SHA
            and consumer_sha==CONSUMER_SHA and receipt_sha==RECEIPT_SHA,
            'MAIN actual new four-word manifest/report/consumer/receipt SHA')
    # Require actual receipt BEFORE importing wrapper; no future-hash defaults.
    manifest=read_json(manifest_path,manifest_sha)
    report=read_json(report_path,report_sha)
    receipt=read_json(receipt_path,receipt_sha)
    require(receipt['status']=='all4_saved_native_words_verified' and receipt['denominator']==4
            and receipt['scientific_admission'] is False and receipt['task_success'] is None,'MAIN completed strict receipt')
    consumer=load_bound(consumer_source,consumer_sha)
    require(callable(getattr(consumer,'consume',None)),'MAIN strict wrapper must expose consume(manifest_path,manifest_sha,report_path,report_sha)')
    actual=consumer.consume(Path(manifest_path),manifest_sha,Path(report_path),report_sha)
    require(actual==receipt,'exact MAIN strict saved return')
    require(sha(ROOT/'experiments/spectral_reachability'/(PHYSICAL+'.py'))==PINS[PHYSICAL],'frozen physical source')
    require(manifest['schema']==PHYSICAL and report['schema']==PHYSICAL+'.report'
            and report['status']=='all4_native_word_traces_recorded' and report['global_error'] is None,'actual all4 physical report')
    authority=manifest['authority']
    require(authority['model_sha']==MODEL_MANIFEST_SHA and authority['report_sha']==MODEL_REPORT_SHA
            and authority['receipt_sha']==MODEL_RECEIPT_SHA,'exact model manifest/report/strict join')
    numeric=helper(MODEL)
    mm=read_json(authority['model_path'],MODEL_MANIFEST_SHA)
    model=read_json(authority['report_path'],MODEL_REPORT_SHA)
    model_receipt=read_json(authority['receipt_path'],MODEL_RECEIPT_SHA)
    require(numeric.consume(mm,authority['report_path'],MODEL_REPORT_SHA)==model_receipt,'exact saved model consumer')
    require(manifest['public_payload_sha256']==mm['public_payload_sha256']==actual['public_payload_sha256'],
            'same actual public context identity')
    plans=manifest['source_contract']['generator_private']['plans']
    for plan,word in zip(plans,WORDS):
        require(plan['arm']==word and np.array_equal(np.asarray(plan['reference_suffix_raw_u'],np.float64),
                np.asarray(mm['plans'][word]['raw'],np.float64).reshape(10,2)),'full native/model raw plan join')
    out=Path(report_path).parent
    def frame(row,f): return record_loader(out,row)(f)
    def command(row,f): return read_json(out/row['arm']/f'command_{f:02d}.json')
    def callbacks(row,f):
        saved=read_json(out/row['arm']/f'callbacks_{f:02d}.json')
        require(saved['frame']==f,'callback frame identity')
        return saved['events']
    data=assemble(plans,report['rows'],model,frame,command,callbacks,helper(ARITHMETIC))
    inputs={str(Path(p).resolve()):h for p,h in ((manifest_path,manifest_sha),(report_path,report_sha),
        (receipt_path,receipt_sha),(consumer_source,consumer_sha),(authority['model_path'],MODEL_MANIFEST_SHA),
        (authority['report_path'],MODEL_REPORT_SHA),(authority['receipt_path'],MODEL_RECEIPT_SHA))}
    inputs.update({str(ROOT/'experiments/spectral_reachability'/(n+'.py')):h for n,h in PINS.items()})
    for p in (Path(__file__).resolve(),ROOT/'tests'/('test_'+TAG+'.py'),ROOT/'innerdocs'/(TAG+'.md')):
        inputs[str(p)]=sha(p)
    require(all(sha(p)==h for p,h in inputs.items()),'final saved authority drift')
    return dict(schema=TAG+'.report',status='all4_saved_native_word_responses_analyzed',policy=policy(),inputs=inputs,
        physics_strict=receipt,model_strict=model_receipt,physical_meter=report['meter'],operation_meter=report['operation_meter'],
        wall_seconds_physical=report['wall_seconds'],neural_executed=False,physics_executed=False,
        scientific_admission=False,task_success=None,**data)


def write_new(path,value):
    p=Path(path)
    require(p.parent.is_dir() and not any(x.is_symlink() for x in (p,*p.parents)),'new nonsymlink output')
    data=canonical(value)+b'\n'
    require(len(data)<=CAP,'output cap')
    with p.open('xb') as f:
        require(f.write(data)==len(data),'short output write'); f.flush(); os.fsync(f.fileno())
    fd=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['analyze','verify'])
    for k in ('manifest','report','consumer','receipt'):
        p.add_argument('--'+k,type=Path,required=True)
        p.add_argument('--'+k+'-sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--output-sha256')
    a=p.parse_args(argv)
    if a.command=='analyze': require(not a.output.exists(),'exclusive new output')
    else: require(a.output_sha256,'actual saved analysis SHA')
    result=analyze(a.manifest,a.manifest_sha256,a.report,a.report_sha256,
        a.consumer,a.consumer_sha256,a.receipt,a.receipt_sha256)
    if a.command=='analyze': write_new(a.output,result)
    else: require(read_json(a.output,a.output_sha256)==result,'deterministic saved recomputation mismatch')
    print(json.dumps(dict(status='saved_analysis_written' if a.command=='analyze' else 'saved_analysis_verified',
        output=str(a.output.resolve()),sha256=sha(a.output),scientific_admission=False),sort_keys=True))
    return 0


if __name__=='__main__': raise SystemExit(main())
