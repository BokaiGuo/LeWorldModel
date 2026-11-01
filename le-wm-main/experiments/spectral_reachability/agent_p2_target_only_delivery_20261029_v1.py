"""Saved-only known2 target delivery; no controller dispatch or full strict."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from experiments.spectral_reachability import agent_p2_history3_tracker_cuda_20261029_v1 as tracker

TAG='agent_p2_target_only_delivery_20261029_v1'
ROOT=Path(__file__).resolve().parents[2]
CAP=16*1024**2
need=tracker.need
PINS={
 'parity_prepare':('agent_p2_hilewm_computation_parity_20261029_v1_prepare_MAIN01.json','796c80ca14574b29b9920b0ef53b02cc6c3b3b47842708951e5bc0b17aadf4e1'),
 'parity_report':('agent_p2_hilewm_computation_parity_20261029_v1_cuda_MAIN01/report.json','12fe43cc4a796fffc18b98fcc867283f8534c22110d794729652ac112aedead8'),
 'parity_receipt':('agent_p2_hilewm_computation_parity_20261029_v1_consume_MAIN01.json','3555696c02329062084d791c0c40958084d7352e6f7680938bade0068afe9480'),
 'prior_prepare':('agent_hilewm_rgb_high_low_20261028_v1_prepare_MAIN01/manifest.json','a3255d38e9cf718f9560c1b9e381abe38750edd9e91d3f04e5ad332cf414d7e9'),
 'prior_report':('agent_hilewm_rgb_high_low_20261028_v1_cuda_MAIN01/report.json','083d0d9aa190b561e6ccf865b09d7224b97d404b16ba38c792bd8af386f5622a'),
 'prior_receipt':('agent_hilewm_rgb_high_low_20261028_v1_consume_MAIN01.json','40f28414682c9540e9e195b361a36dfc514f816358adbeacb379fedc460e4ef0'),
 'program_report':('agent_p2_three_program_20261028_v1_cuda_MAIN01/report.json','56e8b4f2df522f7a0c4cb9bc37ad74bcbc5d4f89e2c5cd601d42cd0eb071d8ef'),
 'program_receipt':('agent_p2_three_program_20261028_v1_consume_MAIN01.json','c491555fda5bb543af69f70987541edb0abb99fa32c559f3aa6190901393017f'),
 'tracker_prepare':('agent_p2_history3_tracker_cuda_20261029_v1_prepare_MAIN01.json','161af4ff5b70a656ad147f2775d169fb2087aa72a8e07e9b6e40aabce0ae5456'),
 'tracker_report':('agent_p2_history3_tracker_cuda_20261029_v1_cuda_MAIN01/report.json','ab89ea3c8c7198ce001280a93c73c312c5454f552b4452232e04a6776adf6884'),
 'tracker_receipt':('agent_p2_history3_tracker_cuda_20261029_v1_consume_MAIN01.json','080fe1173895621ea4c0565671509a267ba456135fedfac313ddac7894f98138'),
}


def encode(x):return tracker.encode(x)
def sha(p):return tracker.sha(p)


def authority_paths():
    return {k:dict(path=str(ROOT/'results'/p),sha256=h) for k,(p,h) in PINS.items()}


def load_authorities():
    refs=authority_paths()
    docs={k:tracker.read(r['path'],r['sha256']) for k,r in refs.items()}
    for tag in ('parity','prior','tracker'):
        report,receipt=docs[tag+'_report'],docs[tag+'_receipt']
        need(report['manifest_sha256']==refs[tag+'_prepare']['sha256'] and
             Path(report['manifest']).resolve()==Path(refs[tag+'_prepare']['path']),tag+' report->prepare')
        need(receipt['report_sha256']==refs[tag+'_report']['sha256'],tag+' receipt->report')
    need(docs['program_receipt']['planner_report_sha256']==refs['program_report']['sha256'],'program receipt->report')
    need(docs['tracker_prepare']['authorities']=={
        k:refs['parity_'+v] for k,v in (('manifest','prepare'),('report','report'),('receipt','receipt'))},'tracker exact parity authorities')
    need(docs['tracker_prepare']['upstream_receipt']==docs['parity_receipt'],'entire parity receipt bound by tracker')
    need(docs['tracker_receipt']['manifest_sha256']==refs['tracker_prepare']['sha256'],'tracker receipt->prepare')
    need(docs['parity_receipt']['all_values_exact'] is True and docs['tracker_report']['completed'] is True
         and docs['tracker_report']['errors']==[],'complete original parity/tracker')
    need(docs['tracker_receipt']['status']=='all_known2_normalized_tracker_saved_verified' and
         len(docs['tracker_receipt']['local_checks'])==2 and docs['tracker_receipt']['predict_API']==24 and
         docs['tracker_receipt']['predictor_samples']==108,'complete original tracker receipt')
    return refs,docs


def same(a,b):return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()


def payload(z,a,target):
    tracker.core.check_inputs(z,a,target,tracker.core.RAMOps())
    out={k:v.tolist() for k,v in dict(z=z,a=a,target=target).items()}
    validate_payload(out)
    return out


def validate_payload(p):
    need(type(p) is dict and set(p)=={'z','a','target'},'controller allowlist only: no witness/warmstart')
    arrays={k:np.asarray(v,np.float32) for k,v in p.items()}
    tracker.core.check_inputs(arrays['z'],arrays['a'],arrays['target'],tracker.core.RAMOps())
    return arrays


def assemble(docs,get):
    """Pure RAM/saved join. get(role,array_name) verifies exact owner descriptor."""
    pm,pr=docs['parity_prepare'],docs['parity_report']
    hm,hr,hc=(docs[k] for k in ('prior_prepare','prior_report','prior_receipt'))
    programs=docs['program_receipt']['plans']; rows=[]
    need([(s['root'],s['condition']) for s in pm['slots']]==[(0,'off'),(1,'off')],'fixed known2')
    need([(r['root_slot'],r['condition']) for r in hr['rows']]==[(0,'off'),(0,'on'),(1,'off'),(1,'on')],'original high four denominator')
    need(len(programs)==4 and len(pr['rows'])==2,'complete original denominators')
    need([(p['root_slot'],p['arm']) for p in programs]==[(i,arm) for i in (0,1)
         for arm in ('joint_H3_no_composition','three_program_imagined_recompile')],'fixed program slots not selected by outcome')
    need(hr['completed'] is True and hr['errors']==[] and len(hc['rows'])==4,'complete prior saved authorities')
    for root in (0,1):
        s,original=pm['slots'][root],pr['rows'][root]
        high=hr['rows'][2*root]; hs=hm['slots'][2*root]; check=hc['rows'][2*root]; p=programs[1+2*root]
        need(original['root']==root and original['status']=='returned','parity root identity')
        low=original['routes']['hi_whole']
        z,a=get('parity',low['z']),get('parity',low['a'])
        goal=get('parity',original['pixels'][3]['paths']['hi']['output'])
        need(same(z,np.asarray(s['z'],np.float32)[None]) and same(a,np.asarray(s['a'],np.float32)[None]),'original history/action exact')
        need(hs['pixel_sha256']==s['pixel_sha256'] and hs['blind_sha256']==s['blind_sha256'],'same RGB/source episode')
        need(same(get('prior',high['z_init']),z[:,-1]) and same(get('prior',high['z_goal']),goal),'actual high current/goal same coordinate')
        first=get('prior',high['high_target']); roll=get('prior',high['selected_rollout'])
        need(roll.shape==(1,1,2,192) and same(first,roll[:,0,0,:]),'exact original FIRST not final waypoint')
        need(check['root_slot']==root and check['condition']=='off' and check['target_delivered_exact'] is True
             and check['high_target_sha256']==check['low_goal_sha256']==hashlib.sha256(first.tobytes()).hexdigest(),'prior receipt full target byte join')
        need(p['task_id']==s['task_id'] and p['task_sha256']==s['task_sha256'] and p['blind_sha256']==s['blind_sha256'],'program exact original task')
        ctx=p['root_context']
        need(same(np.asarray(ctx['latent_history'],np.float32)[None],z) and
             same(np.asarray(ctx['action_history'],np.float32)[None],a) and same(np.asarray(p['goal'],np.float32)[None],goal),'program original context/goal same coordinate')
        need(len(p['edges'])==3 and p['compiled_nodes']==[0,5,10] and p['raw_steps']==15,'original three postconditions')
        e=p['edges'][0]
        need(e['index']==0 and e['input_context']==ctx and e['task_id']==s['task_id'],'first edge original context')
        post=np.asarray(e['output_context']['latent_history'],np.float32)
        need(post.shape==(3,192),'original full history3 postcondition; exact pointer index2')
        target=post[-1][None]
        shared=dict(root=root,condition='off',task_id=s['task_id'],task_sha256=s['task_sha256'],
                    blind_sha256=s['blind_sha256'],history_sha256=hashlib.sha256(z.tobytes()).hexdigest(),
                    action_history_sha256=hashlib.sha256(a.tobytes()).hexdigest())
        for arm,t,origin in (
            ('prior_FIRST',first,dict(role='prior_report',pointer=f'/rows/{2*root}/high_target',array=high['high_target'],
                receipt_pointer=f'/rows/{2*root}',physical_time=None)),
            ('program_first_postcondition',target,dict(role='program_receipt',pointer=f'/plans/{1+2*root}/edges/0/output_context/latent_history/2',
                selection_digest=p['selection_digest'],physical_time=dict(source_raw_step=5,controller_terminal_equivalence=False)))):
            rows.append(dict(**shared,arm=arm,status='assembled_not_dispatched',controller_payload=payload(z,a,t),
                target_sha256=hashlib.sha256(t.tobytes()).hexdigest(),origin=origin))
        rows.append(dict(**shared,arm='arbitrary',status='missing',controller_payload=None,
                         reason='no authenticated direct-latent optimizer output on the common ABI'))
    return rows


def build():
    refs,docs=load_authorities(); sources={r['path']:r['sha256'] for r in refs.values()}
    # Frozen pure saved-array/source helpers only; no consumer/verify/model calls.
    helpers=tracker.closure()
    need(all(docs['tracker_prepare']['sources'].get(p)==h for p,h in helpers.items()),
         'all reused executable helpers exact original tracker closure')
    sources.update(helpers)
    for folder,name in (('experiments/spectral_reachability',TAG+'.py'),('tests','test_'+TAG+'.py'),('innerdocs',TAG+'.md')):
        p=ROOT/folder/name;sources[str(p)]=sha(p)
    def get(role,name):
        report=docs[role+'_report']; p=Path(refs[role+'_report']['path']).parent/name
        need(Path(name).name==name,'flat original NPY name')
        desc=report['arrays'][name];arr=tracker.parity.load_array(p.parent,name,desc)
        sources[str(p)]=desc['file_sha256'];return arr
    rows=assemble(docs,get)
    need(all(sha(p)==h for p,h in sources.items()),'all selected sources/arrays stable after join')
    return dict(schema=TAG+'.prepare',authorities=refs,full_authority_receipts={k:docs[k] for k in ('parity_receipt','prior_receipt','tracker_receipt')},
        sources=sources,rows=rows,denominator=dict(roots=2,arms=3,total=6,assembled=4,missing=2,dispatched=0),
        common_controller=dict(source=str(Path(tracker.core.__file__).resolve()),source_sha256=sha(tracker.core.__file__),
            callable='track_cuda',input_keys=['z','a','target'],target_shape=[1,192],dtype='float32',
            privileged_witness_warmstart=False,dispatch=False,spec_not_authorized_by_delivery=True),
        limitations=dict(known_only=True,fresh=False,arbitrary='missing',prior_physical_time=None,
            three_target_time_alignment=False,controller_H3_is_not_source_cp5=True,
            same_generation_budget=False,inherited_proposal_costs_not_reexecuted=True,
            generation_budget_comparison=None,raw_scaler=None,raw_inverse=False,scientific_admission=False),
        actual_new_calls=dict(encoder=0,predictor=0,high=0,AD=0,physics=0,full_strict=0),
        evidence_scope='pinned complete saved receipts imported; selected array arithmetic only; NOT full strict rebuilt')


def output_path(p):
    p=Path(p);need(p.parent.resolve()==ROOT/'results' and p.name.startswith(TAG+'_') and not p.is_symlink(),'exclusive own results child');return p


def prepare(output):
    artifact=build();b=encode(artifact);p=output_path(output)
    need(len(b)<=CAP and shutil.disk_usage(p.parent).free>=200*1024**2+len(b),'metadata cap/floor')
    tracker.durable(p,b);return dict(path=str(p.resolve()),sha256=sha(p))


def verify(path,h):
    old=tracker.read(path,h);need(old==build(),'entire delivery reconstructed, not status-only')
    return dict(verified=True,sha256=h,dispatched=False,full_strict_rebuilt=False)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('authorities','prepare','verify'))
    p.add_argument('--output',type=Path);p.add_argument('--manifest',type=Path);p.add_argument('--manifest-sha256');a=p.parse_args(argv)
    out=authority_paths() if a.command=='authorities' else prepare(a.output) if a.command=='prepare' else verify(a.manifest,a.manifest_sha256)
    print(json.dumps(out,sort_keys=True,allow_nan=False))


if __name__=='__main__':main()
