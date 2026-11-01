"""Saved-only fixed80 renderer x goal factorial; no new encoder/renderer/selector."""
from __future__ import annotations
import argparse
from copy import deepcopy
from fractions import Fraction
import gzip
from io import BytesIO
import itertools
import os
from pathlib import Path
import numpy as np
from experiments.spectral_reachability import agent_p1_prospective_action_effect_20261028_v4 as strict

v2=strict.v2
ROOT=v2.ROOT
NAME='agent_p1_renderer_matched_goal_20261028_v1'
require=v2.require
REPORT=ROOT/'results/agent_p1_prospective_action_effect_20261028_v2_MAIN01/report.json'
REPORT_SHA=strict.ACTUAL_SHA
RECEIPT=ROOT/'results/agent_p1_prospective_action_effect_20261028_v4_MAIN01.json'
RECEIPT_SHA='5a5c1948dd37ff0725447887218f6d57e59aaae54a3f1df28925fb8fe32b01a8'
ENCODE=ROOT/'results/p1_moving_overlay_recompile_cuda_20261027_v1_cuda_MAIN01/report.json'
ENCODE_SHA='2b2e5064f74f8aca712ae0d0415fcbfb64a2c8e6325bdffe24ae4fdc3b373381'


def eq(a,b):return v2.canonical(a)==v2.canonical(b)
def sign(x):return int(x>0)-int(x<0)
def pack(x):return [str(x.numerator),str(x.denominator)]
def scalar(x):return dict(exact=pack(x),float64=float(x),sign=sign(x))
def vec(x):
    a=np.asarray(x)
    require(a.shape==(192,) and np.isfinite(a).all(),'finite native192')
    return [Fraction.from_float(float(y)) for y in a]
def dot(a,b):return sum((x*y for x,y in zip(a,b,strict=True)),Fraction())
def sub(a,b):return [x-y for x,y in zip(a,b,strict=True)]
def margin(a,b,g):
    return (dot(sub(a,g),sub(a,g))-dot(sub(b,g),sub(b,g)))/192


def factorial_pair(off,on,g_off,g_on,i,j,condition):
    """All four comparisons fixed before outcomes; /192 exactly once, no epsilon."""
    require(condition in ('off','on') and 0<=i<j<5,'fixed pair and renderer')
    zs={'off':[vec(off[i]),vec(off[j])],'on':[vec(on[i]),vec(on[j])]}
    gs={'off':vec(g_off),'on':vec(g_on)}
    ms={f'{r}_{g}':margin(*zs[r],gs[g]) for r in ('off','on') for g in ('off','on')}
    other='on' if condition=='off' else 'off'
    original=ms[f'{condition}_{condition}'];unmatched=ms[f'{other}_{condition}'];matched=ms[f'{other}_{other}']
    goal_shift=matched-unmatched
    expected=-2*dot(sub(*zs[other]),sub(gs[other],gs[condition]))/192
    interaction=ms['on_on']-ms['on_off']-ms['off_on']+ms['off_off']
    expected_interaction=-2*dot(sub(sub(*zs['on']),sub(*zs['off'])),sub(gs['on'],gs['off']))/192
    require(goal_shift==expected and interaction==expected_interaction,'exact renderer-goal margin identities')
    return dict(i=i,j=j,original_condition=condition,margins={k:scalar(v) for k,v in ms.items()},
        original_vs_alternate_same_goal_strict_reversal=original*unmatched<0,
        matched_renderer_strict_reversal=original*matched<0,
        goal_change_on_alternate_strict_reversal=unmatched*matched<0,
        unmatched_renderer_delta=scalar(unmatched-original),matched_renderer_delta=scalar(matched-original),
        goal_shift_alternate=scalar(goal_shift),goal_shift_projection=scalar(expected),
        renderer_goal_interaction=scalar(interaction),
        goal_shift_identity_residual=pack(goal_shift-expected),interaction_identity_residual=pack(interaction-expected_interaction))


def validate_goal_triplet(source,events,root,goal):
    frames=[e for e in events if e['kind']=='fresh_overlay_frame' and e.get('source_slot')==root and
        e.get('key')==f's{root}_construction' and e['receipt']['reference_key']=='20']
    require(len(frames)==3 and [e['receipt']['overlay_index'] for e in frames]==[0,1,2] and
        [e['receipt']['with_target'] for e in frames]==[False,True,False] and
        [e['index'] for e in frames]==sorted(set(e['index'] for e in frames)),'actual goal off/on/off triplet order')
    receipts=[e['receipt'] for e in frames];fixed=receipts[0]['pre']
    require(all(eq(x['pre'],x['post']) and eq(x['pre'],fixed) for x in receipts) and
        eq(fixed['snapshot']['state'],goal) and
        fixed['snapshot']['state_float64_bits']==np.asarray(goal,dtype='<f8').tobytes().hex(),
        'same complete physical goal state/snapshot/renderer, not native outline pose')
    require(receipts[0]['file_sha256']==receipts[2]['file_sha256'] and
        receipts[0]['pixel_sha256']==receipts[2]['pixel_sha256'] and receipts[2]['off_replay_match'] is True,
        'exact off restoration at real goal state')
    row=[r for r in source['rows'] if r['key']==f's{root}_construction']
    require(len(row)==1 and row[0]['status']=='source_episode_completed' and row[0]['missing_triplets']==[],
        'complete source construction identity')
    for condition,index in [('off',0),('on',1)]:
        r=receipts[index];declared=source['blind_exports'][root]['artifacts'][condition]
        require(r['pixel_sha256']==declared['goal_pixel_sha256'] and
            dict(reference_key='20',overlay_index=index,png=r['png'],file_sha256=r['file_sha256'],pixel_sha256=r['pixel_sha256']) in row[0]['frames'],
            'goal source triplet joins paired blind export and source row')
    return dict(off=receipts[0],on=receipts[1])


def legacy_array(base,receipt):
    """Original overlay arrays predate npy_sha256; enforce their actual ABI."""
    path=v2.teacher.flat(base,receipt['file'])
    require(path.is_file() and not path.is_symlink() and path.stat().st_size<=4*1024**2 and
        v2.sha(path)==receipt['file_sha256'],'original goal tensor file SHA')
    with gzip.open(path,'rb') as f:raw=f.read(4*1024**2+1)
    require(len(raw)<=4*1024**2,'bounded original goal NPY')
    stream=BytesIO(raw);a=np.load(stream,allow_pickle=False)
    require(stream.tell()==len(raw) and str(a.dtype)==receipt['dtype'] and list(a.shape)==receipt['shape'] and
        np.isfinite(a).all() and v2.teacher.overlay.hash_bytes(a.tobytes())==receipt['array_sha256'],'original goal array dtype/shape/bytes')
    return a


def source_events(path):
    """Use ORIGINAL source serialization/hash, not the later v2 JSON ABI."""
    paired=v2.phys.adapter.paired
    require(Path(path).is_file() and not Path(path).is_symlink() and Path(path).stat().st_size<=paired.CAP,'bounded paired source journal')
    events=[];previous=None
    with Path(path).open('rb') as stream:
        for line in stream:
            e=v2.phys.adapter.decode(line)
            require(line==paired.raw(e) and e['index']==len(events) and e['previous_sha256']==previous and
                e['event_sha256']==paired.digest({k:v for k,v in e.items() if k!='event_sha256'}),'original paired source journal chain/serialization')
            events.append(e);previous=e['event_sha256']
    return events


def acquire():
    """Reuse MAIN strict80 receipt; newly validate only goal chains and joins."""
    bindings={}
    def read(path,h,cap=v2.phys.REPORT_CAP):
        x=v2.teacher.read(path,h,cap);bindings[str(Path(path).resolve())]=h;return x
    report=read(REPORT,REPORT_SHA);accepted=read(RECEIPT,RECEIPT_SHA)
    require(accepted['schema']==strict.NAME+'.consume' and accepted['report_sha256']==REPORT_SHA and
        accepted['status']=='strict_phase_saved_adapter_audited_not_P1_admission' and accepted['P1_admission'] is False and
        accepted['adapter_usage']==dict(AST_call_substitutions=1,native_calls=1000,terminal_rows=40),
        'MAIN strict factual25 receipt, no producer rebuild')
    for path,h in {**accepted['frozen_authority'],**accepted['consumer_files']}.items():
        require(v2.sha(path)==h,'accepted strict consumer frozen');bindings[path]=h
    package=read(report['binding']['prepared'],report['binding']['prepare_sha256'])
    parent=read(package['parent'],package['parent_sha256'])
    require(package['parent']==str(v2.PARENT) and package['parent_sha256']==v2.PARENT_SHA,'original physical task authority')
    original=read(parent['compiler_prepared'],parent['compiler_prepare_sha256'])
    encoded=read(ENCODE,ENCODE_SHA);overlay=v2.teacher.overlay
    for module in (overlay,v2.phys.pose,v2.phys.adapter.paired):
        path=str(Path(module.__file__).resolve());h=package['bindings'][path]
        require(v2.sha(path)==h,'actual imported original saved utility');bindings[path]=h
    prep=read(encoded['prepared'],encoded['prepare_sha256'])
    source=read(overlay.SOURCE,overlay.SOURCE_SHA)
    require(encoded['source_report_sha256']==overlay.SOURCE_SHA and source['full_protocol'] is True and
        source['source_verified'] is True and source['errors']==[] and source['contract']['no_goal_pose_edit'] is True and
        encoded['status']=='overlay_recompiled_plans_sealed_not_admitted' and encoded['model_unchanged'] is True and
        encoded['model_hash_before']==encoded['model_hash_after']==package['expected_model_hash']==prep['expected_model_hash'] and
        prep['contract']['edit_goal_pose'] is False,'same frozen encoder and renderer-only paired source')
    journal=overlay.SOURCE.parent/'journal.jsonl';events=source_events(journal)
    require(events[-1]['event_sha256']==source['journal_chain_tail'] and events[-1]['kind']=='run_finished','source full journal signed tail')
    bindings[str(journal)]=v2.sha(journal)
    goals={};metadata=[]
    for root in (0,1):
        natives=[x['native'] for x in original['ancestry']['tasks'] if x['native']['root_slot']==root]
        require(len(natives)==2 and eq(natives[0],natives[1]),'off/on same physical task, full7 goal and native outline pose')
        physical=natives[0];trip=validate_goal_triplet(source,events,root,physical['goal'])
        require(eq(trip['off']['pre']['goal_pose'],physical['goal_pose']) and eq(trip['off']['pre']['renderer'],physical['renderer']),
            'goal renderer/outline identity with original P1 task')
        for condition in ('off','on'):
            order=root*2+(condition=='on');slot=prep['slots'][order]
            require((slot['root_slot'],slot['condition'])==(root,condition),'fixed existing four goal slots')
            packet=overlay.load_packet(slot['blind_path'],slot['blind_sha256']);bindings[slot['blind_path']]=slot['blind_sha256']
            frame=trip[condition];png=overlay.SOURCE.parent/frame['png']
            pixels=v2.teacher.rgb(png,frame['file_sha256'],frame['pixel_sha256']);bindings[str(png)]=frame['file_sha256']
            require(v2.exact_array(pixels,packet['goal_rgb']) and physical['goal_pixels'][condition]==frame['pixel_sha256']==slot['pixel_sha256'][3],
                'actual same-state rendered goal pixels -> blind packet')
            image=encoded['images'][4*order+3]
            require(image['image_index']==4*order+3 and image['root_slot']==root and image['condition']==condition and
                image['status']=='encoded','actual independently encoded goal image')
            def array(receipt):
                require(encoded['artifact_sha256'][receipt['file']]==receipt['file_sha256'],'original goal tensor inventory')
                a=legacy_array(ENCODE.parent,receipt);bindings[str(ENCODE.parent/receipt['file'])]=receipt['file_sha256'];return a
            require(v2.exact_array(array(image['pixels']),pixels),'real saved goal pixels, not synthetic replacement')
            pre=array(image['preprocessed'])
            require(v2.exact_array(pre,strict.v3.pixels.preprocess_pixels(pixels)),'goal pixels -> original224 preprocessing')
            mods=image['modules'];require([e['resource'] for e in mods]==['encoder','projector'] and
                all(e['completed'] is True for e in mods) and mods==[e for e in encoded['module_ledger'] if e['scope']==dict(image_index=image['image_index'])],
                'original goal native encoder/projector ledger')
            enc,proj=mods;ein=array(enc['inputs'][0]);eout=array(enc['output']);pin=array(proj['inputs'][0]);pout=array(proj['output']);z=array(image['latent'])
            require(all(a.dtype==np.float32 for a in (pre,ein,eout,pin,pout,z)) and eout.shape==(1,257,192) and z.shape==(192,) and
                pin.shape==pout.shape==(1,192) and len(enc['inputs'])==len(proj['inputs'])==1 and
                v2.exact_array(pre.reshape(1,3,224,224),ein) and v2.exact_array(eout[:,0],pin) and v2.exact_array(pout[0],z),
                'original goal exact CLS -> projector -> latent chain')
            anchors=[a for a in parent['anchors'] if (a['root_slot'],a['condition'])==(root,condition)]
            require(len(anchors)==2,'both fixed roles for goal')
            for a in anchors:
                require(v2.sha(a['task_file'])==a['task_sha256'],'sealed existing task export');bindings[a['task_file']]=a['task_sha256']
                exported=overlay.read(a['task_file']);tc=exported['task_contract'];task=exported['task']
                require(tc['root_slot']==root and tc['condition']==condition and tc['goal_pose_edited'] is False and
                    tc['blind_sha256']==slot['blind_sha256'] and task['goal_source_sha256']==frame['pixel_sha256'] and
                    eq(task['goal'],a['goal']) and v2.exact_array(z,np.asarray(a['goal'],np.float32)),
                    'old P1 encoded goal is the bound existing renderer-matched goal')
            goals[root,condition]=dict(latent=z.tolist(),physical_goal=deepcopy(physical['goal']),pixel_sha256=frame['pixel_sha256'])
            metadata.append(dict(root_slot=root,condition=condition,physical_goal=physical['goal'],native_outline_pose=physical['goal_pose'],
                png=str(png),pixel_sha256=frame['pixel_sha256'],blind_path=slot['blind_path'],encoding_image_index=image['image_index'],
                latent_array=image['latent']))
    return report,parent,goals,dict(bindings=bindings,goals=metadata,new_goal_RGB_needed=[],new_neural_calls=0,new_physics_calls=0,
        original_strict_receipt_reused=True,upstream_production_or_full_consume_repeated=False)


def analyze(report,parent,goals):
    require(len(report['rows'])==40 and len(report['anchors'])==len(parent['anchors'])==8,'fixed factual40/8')
    pairs=[];anchors=[]
    for ai,a in enumerate(parent['anchors']):
        rows=report['rows'][ai*5:ai*5+5];root=a['root_slot'];condition=a['condition'];other='on' if condition=='off' else 'off'
        g=goals[root,condition];other_g=goals[root,other]
        require(eq(g['physical_goal'],other_g['physical_goal']) and eq(g['latent'],a['goal']),'same physical goal with two renderer encodings')
        seal=report['anchors'][ai]['seal'];require(v2.selection(seal['predictions'],g['latent'])==seal,'original model-only selection unchanged')
        endpoints={'off':[],'on':[]};words=[]
        for word,row in enumerate(rows):
            require((row['index'],row['anchor_index'],row['word'],row['root_slot'],row['condition'])==(ai*5+word,ai,word,root,condition) and
                eq(row['goal'],g['physical_goal']) and row['goal_pixel_sha256']==g['pixel_sha256'] and
                row['terminal_return']['phase']=='suffix' and row['terminal_return']['step']==10 and row['status']=='terminal25_encoded',
                'same fixed action/task/renderer and actual terminal25')
            trip=row['terminal_triplet'];v2.validate_triplet(trip)
            require(eq(trip[0]['receipt']['state'],row['terminal_return']['state']) and
                eq(row['terminal_return']['state'],row['end_state']),'same physical endpoint for both renderers')
            endpoints[condition].append(row['endpoint']);endpoints[other].append(row['alternate_endpoint'])
            native=row['terminal_return'];diagnostics=v2.phys.pose.pose_diagnostics(g['physical_goal'],native['state'])
            require(diagnostics==native['diagnostics'] and v2.native_labels(g['physical_goal'],native['state'])==
                {k:native[k] for k in ('native_success','native_distance')},'existing physical labels retained')
            q={f'{r}_{gc}':float(np.mean((np.asarray(endpoints[r][-1],np.float64)-np.asarray(goals[root,gc]['latent'],np.float64))**2))
                for r in ('off','on') for gc in ('off','on')}
            words.append(dict(word=word,Q_float64=q,native_distance=native['native_distance'],native_success=native['native_success'],
                all_xy=diagnostics['independent']['all_xy_position_error'],wrapped_angle=diagnostics['independent']['wrapped_angle_error']))
        for i,j in itertools.combinations(range(5),2):
            p=factorial_pair(endpoints['off'],endpoints['on'],goals[root,'off']['latent'],goals[root,'on']['latent'],i,j,condition)
            p.update(anchor_index=ai,root_slot=root,physical_signs={k:sign(words[i][k]-words[j][k]) for k in ('native_distance','all_xy','wrapped_angle')})
            pairs.append(p)
        selected=words[seal['selected_word']];zero=words[0]
        unchanged=dict(Q_delta={k:selected['Q_float64'][k]-zero['Q_float64'][k] for k in selected['Q_float64']},
            native_distance_delta=selected['native_distance']-zero['native_distance'],
            all_xy_delta=selected['all_xy']-zero['all_xy'],wrapped_angle_delta=selected['wrapped_angle']-zero['wrapped_angle'])
        anchors.append(dict(anchor_index=ai,root_slot=root,condition=condition,selected_word_unchanged=seal['selected_word'],
            selected_vs_zero_unchanged=unchanged,words=words))
    require(len(pairs)==80,'full fixed80 retained')
    summary={k:sum(p[k] for p in pairs) for k in ('original_vs_alternate_same_goal_strict_reversal','matched_renderer_strict_reversal','goal_change_on_alternate_strict_reversal')}
    versus={k:{axis:dict(strict_reversals=0,latent_ties=0,physical_ties=0,total=80)
        for axis in ('native_distance','all_xy','wrapped_angle')}
        for k in ('original_matched','alternate_original_goal','alternate_matched')}
    for p in pairs:
        c=p['original_condition'];other='on' if c=='off' else 'off'
        views=dict(original_matched=f'{c}_{c}',alternate_original_goal=f'{other}_{c}',alternate_matched=f'{other}_{other}')
        for view,key in views.items():
            latent=p['margins'][key]['sign']
            for axis,physical in p['physical_signs'].items():
                s=versus[view][axis];s['strict_reversals']+=latent*physical<0
                s['latent_ties']+=latent==0;s['physical_ties']+=physical==0
    summary['versus_physical']=versus
    return dict(fixed_pairs=80,available_pairs=80,unknown_pairs=0,anchors=anchors,pairs=pairs,summary=summary,P1_admission=False,
        interpretation='post-hoc fixed-action renderer x goal diagnostic only; same physical goals/endpoints, no new model prediction or goal-conditioned rollout; not a deployable selector or P1 pass')


def build():
    report,parent,goals,evidence=acquire();out=analyze(report,parent,goals)
    own=[Path(__file__).resolve(),ROOT/'tests'/f'test_{NAME}.py',ROOT/'innerdocs'/f'{NAME}.md']
    bindings=evidence['bindings']
    for path in own:bindings[str(path)]=v2.sha(path)
    for path,h in bindings.items():require(v2.sha(path)==h,'saved inputs unchanged during diagnostic')
    return dict(schema=NAME+'.analysis',status='saved_renderer_matched_goals_not_P1_admission',**out,evidence=evidence,
        new_neural_calls=0,new_physics_calls=0,training=False,Qfit=False,selection_changed=False,gate_changed=False,known_roots=2,fresh_roots=0)


def write(path,value):
    p=Path(path).absolute()
    require(p.resolve()==p and p.parent==ROOT/'results' and p.name.startswith(NAME+'_') and p.suffix=='.json','exclusive new own saved output')
    raw=v2.canonical(value);require(len(raw)<=v2.phys.PREP_CAP,'bounded saved diagnostic')
    with p.open('xb') as f:require(f.write(raw)==len(raw),'complete output');f.flush();os.fsync(f.fileno())
    v2.phys.adapter.paired.fsync_dir(p.parent)
    return dict(path=str(p),sha256=v2.sha(p),status=value['status'])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('analyze','consume'))
    parser.add_argument('--output',required=True);parser.add_argument('--analysis');parser.add_argument('--analysis-sha')
    args=parser.parse_args();value=build()
    if args.command=='consume':
        require(args.analysis and args.analysis_sha,'analysis and external SHA required')
        saved=v2.teacher.read(args.analysis,args.analysis_sha,v2.phys.PREP_CAP)
        require(eq(saved,value),'entire saved factorial diagnostic exact recomputation')
        value=dict(schema=NAME+'.consume',status='saved_renderer_matched_goals_verified_not_P1_admission',analysis_sha256=args.analysis_sha,
            fixed_pairs=80,P1_admission=False,new_neural_calls=0,new_physics_calls=0,bindings=value['evidence']['bindings'])
    print(v2.canonical(write(args.output,value)).decode())


if __name__=='__main__':main()
