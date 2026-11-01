"""A-only saved renderer-goalset selector. No neural/physics/outcome evaluation."""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
from experiments.spectral_reachability import agent_p1_terminal_witness_selector_20261028_v1 as old
from experiments.spectral_reachability import agent_p1_renderer_matched_goal_20261028_v1 as goals
from experiments.spectral_reachability import agent_allmode_physical_20261028_v1 as source

NAME='agent_renderer_goalset_selector_20261028_v1'
ROOT=old.ROOT
require=old.require
sha=old.base.sha
digest=old.sha_object
CAP,FLOOR,WALL=32*1024**2,200*1024**2,300
PINNED=((old,'3d933e44c48b9c45d9b5924f0b140158c2912df1f1d955701a1af35fac3b13b1'),
        (goals,'6d8535182fbfae08803caeb8f2c719b74c99ec1cb5e5725ffc5c823ad6d66076'),
        (source,'c0842f63685718e3d1d562fbbe5e2e3fdf9103954804f4276a1186d8db91efe4'))


def eq(a,b):return old.base.canonical(a)==old.base.canonical(b)


def native_literal(value,shape):
    a=np.asarray(value)
    require(a.shape==shape and a.dtype.kind in 'fi' and np.isfinite(a).all(),'finite native literal shape/type')
    b=a.astype(np.float32)
    require(np.isfinite(b).all() and np.array_equal(a,b),'no lossy float32 literal conversion')
    return b


def score(z,g):
    require(isinstance(z,np.ndarray) and isinstance(g,np.ndarray) and
        z.dtype==g.dtype==np.float32 and z.shape==(192,) and g.shape==(2,192) and
        np.isfinite(z).all() and np.isfinite(g).all(),'native float32 z192/two goals192')
    costs=[float(np.mean((z.astype(np.float64)-v.astype(np.float64))**2)) for v in g]
    i=min(range(2),key=lambda k:(costs[k],k))
    return dict(branch_Q=costs,value=costs[i],active_branch=('off','on')[i])


def check_image(pixels,pre,ein,eout,pin,pout,z):
    require(pixels.dtype==np.uint8 and pixels.shape==(224,224,3),'actual RGB224 pixels')
    expected=goals.strict.v3.pixels.preprocess_pixels(pixels)
    require(all(a.dtype==np.float32 for a in (pre,ein,eout,pin,pout,z)) and
        pre.shape==expected.shape and ein.shape==(1,3,224,224) and
        eout.shape==(1,257,192) and pin.shape==pout.shape==(1,192) and z.shape==(192,) and
        all(np.isfinite(a).all() for a in (pre,ein,eout,pin,pout,z)) and
        goals.v2.exact_array(pre,expected) and goals.v2.exact_array(ein,pre.reshape(1,3,224,224)) and
        goals.v2.exact_array(pin,eout[:,0]) and goals.v2.exact_array(z,pout[0]),
        'independent pixels -> original preprocessing -> encoder257 CLS -> projector -> native192')


def acquire_targets():
    """Only original source metadata, blind pixels and saved model encodings.

    Does NOT call goals.acquire(): that function also loads factual25 outcomes.
    source_material/root_metadata are saved source authority, not physical runs.
    """
    for module,h in PINNED:require(sha(module.__file__)==h,'frozen reused source')
    overlay=old.base.overlay
    material=source.source_material();native=source.root_metadata(material)
    bindings=dict(material['bindings']);images=[];expected=[];target_vectors={}
    def read(path,h,cap=4*1024**2):
        p=Path(path).resolve();require(p.is_file() and not p.is_symlink() and sha(p)==h,'fixed authority SHA')
        x=overlay.read(p,cap);bindings[str(p)]=h;return x
    encoded=read(goals.ENCODE,goals.ENCODE_SHA)
    prepared=read(encoded['prepared'],encoded['prepare_sha256'])
    original=read(overlay.SOURCE,overlay.SOURCE_SHA)
    require(encoded['schema']==overlay.NAME+'.report' and prepared['schema']==overlay.NAME+'.prepare' and
        encoded['status']=='overlay_recompiled_plans_sealed_not_admitted' and encoded['errors']==[] and
        encoded['source_verified'] is encoded['model_unchanged'] is True and
        encoded['source_report_sha256']==prepared['source_report_sha256']==overlay.SOURCE_SHA and
        encoded['model_hash_before']==encoded['model_hash_after']==prepared['expected_model_hash'] and
        prepared['contract']==overlay.contract() and prepared['contract']['edit_goal_pose'] is False and
        len(prepared['slots'])==4 and len(encoded['images'])==16,'actual original encoded coordinate and fixed16')
    for path,h in overlay.moving.MODELS.items():
        require(sha(path)==h and prepared['bindings'][path]==h,'original frozen checkpoint/config bytes')
        bindings[path]=h
    for module in (overlay,overlay.moving,overlay.blind,overlay.history):
        path=str(Path(module.__file__).resolve());h=prepared['bindings'][path]
        require(sha(path)==h,'original imported image/task/normalizer helper source');bindings[path]=h
    pxpath=str(Path(goals.strict.v3.pixels.__file__).resolve())
    pxsha='f7e3abb842cc185c77fe217922cd83ce709f4a6bf8fce0d9337c87bf4f66f75b'
    require(sha(pxpath)==pxsha,'frozen original224 independent pixel preprocessing');bindings[pxpath]=pxsha
    journal=overlay.SOURCE.parent/'journal.jsonl';events=goals.source_events(journal)
    require(events[-1]['kind']=='run_finished' and events[-1]['event_sha256']==original['journal_chain_tail'],
        'actual complete source journal tail')
    bindings[str(journal)]=sha(journal)
    for root in (0,1):
        physical=native[root]
        require(physical['root_slot']==root,'fixed native root metadata')
        trip=goals.validate_goal_triplet(original,events,root,physical['goal'])
        require(eq(trip['off']['pre']['goal_pose'],physical['goal_pose']) and
            eq(trip['off']['pre']['renderer'],physical['renderer']),'source physical goal/outline/renderer binding')
        for condition in ('off','on'):
            i=2*root+(condition=='on');slot=prepared['slots'][i]
            require((slot['order'],slot['root_slot'],slot['condition'])==(i,root,condition),'fixed actual encoded slots')
            packet=overlay.load_packet(slot['blind_path'],slot['blind_sha256'])
            bindings[slot['blind_path']]=slot['blind_sha256']
            src=material['exports'][root,condition]
            require(src==dict(path=slot['blind_path'],sha256=slot['blind_sha256']) and
                all(goals.v2.exact_array(packet[k],material['roots'][root]['packets'][condition][k])
                    for k in ('rgb_history','raw_action_history','goal_rgb')),'actual source packet and raw history')
            f=trip[condition];png=overlay.SOURCE.parent/f['png']
            pixels=goals.v2.teacher.rgb(png,f['file_sha256'],f['pixel_sha256'])
            bindings[str(png)]=f['file_sha256']
            require(goals.v2.exact_array(pixels,packet['goal_rgb']) and
                f['pixel_sha256']==physical['goal_pixels'][condition]==slot['pixel_sha256'][3],
                'same full physical goal -> actual RGB -> bound blind packet')
            zs=[]
            for j,px in enumerate((*packet['rgb_history'],packet['goal_rgb'])):
                image=encoded['images'][4*i+j]
                require((image['image_index'],image['root_slot'],image['condition'],image['status'])==
                    (4*i+j,root,condition,'encoded'),'actual per-image scope')
                def array(rec):
                    require(encoded['artifact_sha256'][rec['file']]==rec['file_sha256'],'original native array inventory')
                    a=goals.legacy_array(goals.ENCODE.parent,rec)
                    bindings[str(goals.ENCODE.parent/rec['file'])]=rec['file_sha256'];return a
                mods=image['modules']
                require([m['resource'] for m in mods]==['encoder','projector'] and
                    all(m['completed'] is True and len(m['inputs'])==1 for m in mods) and
                    mods==[m for m in encoded['module_ledger'] if m['scope']==dict(image_index=4*i+j)],
                    'exact original encoder/projector image journal')
                enc,proj=mods
                savedpx=array(image['pixels']);require(goals.v2.exact_array(savedpx,px),'actual original image pixels')
                pre,ein,eout,pin,pout,z=map(array,(image['preprocessed'],enc['inputs'][0],enc['output'],
                    proj['inputs'][0],proj['output'],image['latent']))
                check_image(px,pre,ein,eout,pin,pout,z);zs.append(z)
                images.append(dict(image_index=4*i+j,root_slot=root,condition=condition,
                    role='goal' if j==3 else f'context{j}',pixel_sha256=slot['pixel_sha256'][j],
                    latent=deepcopy(image['latent'])))
            taskfile=goals.ENCODE.parent/f'task_{i:02}.json.gz'
            tasksha=encoded['artifact_sha256'][taskfile.name];export=read(taskfile,tasksha)
            overlay.task_interface(export);t=export['task'];tc=export['task_contract']
            expected_context=dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,
                action_dim=10,latent_history=[z.tolist() for z in zs[:3]],
                action_history=old.base.kernel.normalized_history(packet['raw_action_history']).tolist())
            require((tc['root_slot'],tc['condition'])==(root,condition) and tc['blind_sha256']==slot['blind_sha256'] and
                tc['blind_path']==slot['blind_path'] and tc['prepare_sha256']==encoded['prepare_sha256'] and
                tc['pixel_sha256']==slot['pixel_sha256'] and t['goal_source_sha256']==f['pixel_sha256'] and
                eq(t['root_context'],expected_context) and goals.v2.exact_array(native_literal(t['goal'],(192,)),zs[3]),
                'real export exact three-frame context/history and original goal bits')
            target_vectors[root,condition]=zs[3].tolist()
            expected.append(dict(index=i,root_slot=root,condition=condition,task_id=export['task_id'],
                task_file=str(taskfile),task_sha256=tasksha,blind_sha256=slot['blind_sha256'],
                packet=dict(root_context=deepcopy(t['root_context']),goal=deepcopy(t['goal'])),q=deepcopy(t['projection']),
                physical_goal=deepcopy(physical['goal']),physical_goal_sha256=digest(physical['goal']),
                native_source=deepcopy(physical)))
    for a in expected:
        a['goal_vectors']=[target_vectors[a['root_slot'],c] for c in ('off','on')]
        a['goalset_sha256']=digest(a['goal_vectors'])
    require(all(sha(p)==h for p,h in bindings.items()),'target authority unchanged')
    return dict(slots=expected,bindings=bindings,images=images,model_hash=prepared['expected_model_hash'],
        coordinate=dict(policy=prepared['policy'],model_bytes=dict(overlay.moving.MODELS)),
        authority='real source metadata + off/on/off same-state journal + original16 image/module NPYS; no outcome reports')


def bind_slot(slot,a):
    for k in ('index','root_slot','condition','task_id','task_sha256','blind_sha256'):
        require(slot[k]==a[k], 'current slot task/root/condition/blind authority')
    require(eq(slot['packet'],a['packet']) and eq(slot['q'],a['q']) and
        a['physical_goal_sha256']==digest(a['physical_goal']), 'current context/goal signed bits and physical goal')
    if 'task_file' in a:require(slot['task_file']==a['task_file'],'literal source task path')
    gs=native_literal(a['goal_vectors'],(2,192));c=('off','on').index(slot['condition'])
    require(goals.v2.exact_array(native_literal(slot['packet']['goal'],(192,)),gs[c]),'original condition goal bits')
    require(np.array_equal(np.asarray(slot['q']),np.eye(192)),'original Q is unchanged identity192')
    return gs


def validate_current_prepares(targets,bindings):
    for family,n,module,stem,h,_ in old.SOURCES:
        path=ROOT/'results'/f'{stem}_prepare_MAIN01.json'
        require(sha(path)==h,'original current library prepare SHA')
        p=old.base.overlay.read(path,module.PREP_CAP);bindings[str(path)]=h
        require(p['schema']==stem+'.prepare' and p['contract']==module.contract() and len(p['slots'])==4 and
            p['source_report_sha256']==old.base.overlay.SOURCE_SHA and
            p['expected_model_hash']==targets['model_hash'] and p['policy']==targets['coordinate']['policy'],
            'current model coordinates and same source authority')
        for slot,a in zip(p['slots'],targets['slots'],strict=True):bind_slot(slot,a)


def select_panel(slot,previous,result,authority,count):
    """Whitelisted model fields only. No source state/outcome used in scoring."""
    gs=bind_slot(slot,authority)
    check=old.select_panel(slot['packet'],slot['q'],result,count)
    # Previous panels carry metadata; only compare their real model panel fields.
    require(all(eq(previous[k],v) for k,v in check.items()),'exact original selector panel reproduction')
    require(check['terminal_available']==count and check['literal_forward_requests']==[],
        'all terminal witnesses required; never select a reduced denominator')
    records=[];anchor=score(native_literal(slot['packet']['root_context']['latent_history'][-1],(192,)),gs)
    for i,(c,original) in enumerate(zip(result['candidates'],check['candidates'],strict=True)):
        z=native_literal(c['predicted_terminal'],(192,));p=native_literal(c['predicted_prefix'],(192,))
        terminal=score(z,gs);prefix=score(p,gs)
        require(terminal['branch_Q'][('off','on').index(slot['condition'])]==original['nonlinear_terminal_Q'],
            'original full terminal Q unchanged by new scoring')
        records.append(dict(**deepcopy(original),branch_Q=terminal['branch_Q'],goalset_Q=terminal['value'],
            active_branch=terminal['active_branch'],prefix_goalset_Q=prefix['value'],
            model_prefix_goalset_uphill_from_anchor=prefix['value']>anchor['value']))
    winner=min(records,key=lambda x:(x['goalset_Q'],x['index']));i=winner['index']
    selected=dict(index=i,score=winner['goalset_Q'],active_branch=winner['active_branch'],
        commands=deepcopy(result['candidates'][i]['commands']),raw_float32_sha256=winner['raw_float32_sha256'])
    return dict(candidate_denominator=count,terminal_available=count,candidates=records,
        original_stage_plus_ridge=deepcopy(check['original_stage_plus_ridge']),
        original_terminal_Q=deepcopy(check['nonlinear_terminal_Q']),goalset_terminal=selected,
        initial_goalset_Q=anchor,context_sha256=check['context_sha256'],original_goal_sha256=check['goal_sha256'],
        original_q_sha256=check['q_sha256'],goalset_sha256=authority['goalset_sha256'],
        physical_goal_sha256=authority['physical_goal_sha256'],task_sha256=slot['task_sha256'],
        task_id=slot['task_id'],blind_sha256=slot['blind_sha256'],
        literal_forward_requests=[],selection_uses_outcome=False,P1_admission=False)


def build():
    started=time.monotonic();targets=acquire_targets()
    bindings=dict(targets['bindings']);validate_current_prepares(targets,bindings)
    original=old.build()  # REAL model validation; never a fabricated wrapper schema.
    require(original['candidate_denominator']==1056 and len(original['panels'])==8,'full1056 authority')
    panels=[]
    for family,n,module,stem,prep_sha,report_sha in old.SOURCES:
        pp=ROOT/'results'/f'{stem}_prepare_MAIN01.json';rp=ROOT/'results'/f'{stem}_cuda_MAIN01'/'report.json'
        require(sha(pp)==prep_sha and sha(rp)==report_sha,'same original model authority')
        p=old.base.overlay.read(pp,module.PREP_CAP);r=old.base.overlay.read(rp,module.CAP)
        bindings[str(pp)]=prep_sha;bindings[str(rp)]=report_sha
        require(p['expected_model_hash']==r['model_hash_before']==targets['model_hash'] and
            p['policy']==targets['coordinate']['policy'],'same actual native model coordinates')
        for i,(slot,row) in enumerate(zip(p['slots'],r['rows'],strict=True)):
            previous=[x for x in original['panels'] if (x['family'],x['index'])==(family,i)]
            require(len(previous)==1,'unique actual original selector panel')
            panel=select_panel(slot,previous[0],row['result'],targets['slots'][i],n)
            panels.append(dict(family=family,index=i,root_slot=slot['root_slot'],condition=slot['condition'],**panel))
    require(len(panels)==8 and sum(p['candidate_denominator'] for p in panels)==1056,'all fixed panels retained')
    files=[Path(__file__).resolve(),ROOT/'tests'/f'test_{NAME}.py',ROOT/'innerdocs'/f'{NAME}.md',
        *(Path(m.__file__).resolve() for m,_ in PINNED),Path(old.base.__file__),Path(old.mixed.__file__),
        Path(old.base.kernel.__file__),Path(old.base.compiler.__file__),Path(goals.strict.v3.pixels.__file__)]
    bindings.update({str(p.resolve()):sha(p) for p in files});bindings.update(original['source_sha256'])
    require(all(sha(p)==h for p,h in bindings.items()),'all source/input bytes remain frozen')
    require(time.monotonic()-started<WALL,'saved-only wall cap')
    selection=dict(schema=NAME+'.selection',candidate_denominator=1056,panels=panels,
        goal_order=['off','on'],stable_selection='min(goalset terminal MSE, candidate_index); goal tie off',
        goalset_def='min over same-state off/on mean192 float64 square; original Q untouched',
        selection_uses_outcome=False,P1_admission=False)
    return dict(schema=NAME+'.analysis',status='A_only_model_selections_sealed_not_admitted',
        selection=selection,selection_sha256=digest(selection),
        target_authority=targets,original_selector_authority=dict(schema=original['schema'],inputs=original['inputs'],
            sources=original['source_sha256'],new_wrapper_not_original_schema=True),bindings=bindings,
        resources=dict(original_selector=original['resources'],
            goal_saved_images_validated=16,new_target_RGB=0,new_neural_calls=0,new_physics_calls=0,
            saved_target_array_reads=16*7,
            new_Q_fits=0,new_training=0,new_proposals=0,goal_branch_terminal_MSE=2112,
            goal_branch_prefix_MSE=2112,goal_branch_anchor_MSE=16,
            additional_original_selector_panel_cost_evaluations=8+2*1056,
            source_metadata_validation='saved source construction/PNG/NPZ/seals; hash/I/O not zero CPU',total_FLOPs=None),
        known_roots=2,fresh_roots=0,unique_root_conditions=4,library_panel_denominator=8,
        evaluation=None,physical_labels_loaded_for_selection=False,original_Q_overwritten=False,
        new_algorithm=True,prospective_efficacy=False,scientific_admission=False,P1_admission=False)


def write(path,value):
    p=Path(path).absolute()
    require(p.resolve()==p and not p.exists() and p.parent==ROOT/'results' and p.suffix=='.json' and
        p.name.startswith(NAME+'_'),'exclusive new own results JSON')
    data=old.base.canonical(value)
    require(len(data)<=CAP and shutil.disk_usage(p.parent).free>=FLOOR+len(data),'bounded output/disk floor')
    with p.open('xb') as f:require(f.write(data)==len(data),'complete durable write');f.flush();os.fsync(f.fileno())
    fd=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return dict(path=str(p),sha256=sha(p),bytes=len(data))


def verify_saved(path,h,value):
    p=Path(path);require(p.is_file() and not p.is_symlink() and sha(p)==h,'external saved analysis SHA')
    require(eq(old.base.overlay.read(p,CAP),value),'full deterministic selection/authority reconstruction')


def prepare_value():
    started=time.monotonic();targets=acquire_targets();validate_current_prepares(targets,targets['bindings'])
    own=[Path(__file__).resolve(),ROOT/'tests'/f'test_{NAME}.py',ROOT/'innerdocs'/f'{NAME}.md']
    require(time.monotonic()-started<WALL,'prepare saved-only wall cap')
    return dict(schema=NAME+'.prepare',targets=targets,own_files={str(p):sha(p) for p in own},
        A_only=True,new_neural_calls=0,new_physics_calls=0,P1_admission=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','analyze','verify'))
    parser.add_argument('--output');parser.add_argument('--prepared');parser.add_argument('--prepare-sha')
    parser.add_argument('--result');parser.add_argument('--sha')
    a=parser.parse_args(argv)
    if a.command=='prepare':
        out=write(a.output,prepare_value())
    else:
        if a.prepared or a.prepare_sha:
            require(a.prepared and a.prepare_sha,'prepared and external SHA together')
            verify_saved(a.prepared,a.prepare_sha,prepare_value())
        value=build()
        if a.command=='analyze':out=write(a.output,value)
        else:
            require(a.result and a.sha,'saved result/external SHA required');verify_saved(a.result,a.sha,value)
            out=dict(status='all1056_goalset_selections_verified_not_admitted',sha256=a.sha,P1_admission=False)
    print(json.dumps(out,sort_keys=True));return 0


if __name__=='__main__':raise SystemExit(main())
