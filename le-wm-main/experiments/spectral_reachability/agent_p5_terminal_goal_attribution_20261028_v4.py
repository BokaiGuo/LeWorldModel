"""v4 order-correct strict terminal RGB diagnostic.

Public authority order is goal, physical, numeric. The frozen v2 delegate
requires goal, numeric, physical; this adapter performs that explicit swap.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from experiments.spectral_reachability import agent_p5_terminal_goal_attribution_20261028_v2 as impl

TAG='agent_p5_terminal_goal_attribution_20261028_v4'; ROOT=impl.ROOT
ORDER=('goal_prepare','goal_report','goal_receipt','physical_manifest','physical_report','physical_receipt','numeric_manifest','numeric_report','numeric_receipt')

def require(ok,msg):
    if not ok: raise ValueError(msg)
def protocol():
    p=dict(impl.protocol()); p.update(schema=TAG,authority_order=list(ORDER),v2_unchanged=True,v3_unchanged=True)
    return p
def paths(values):
    require(len(values)==9,'authority requires goal(3), physical(3), numeric(3)')
    return dict(zip(ORDER,map(Path,values)))
def fixed_authorities(gp,gr,gc,pm,pr,pc,sm,sr,sc):
    """v2 join with distinct path/report variable names; all saved only."""
    require((impl.sha(gp),impl.sha(gr),impl.sha(gc))==(impl.GOAL_PREP,impl.GOAL_REPORT,impl.GOAL_RECEIPT),'goal authority SHA')
    require((impl.sha(pm),impl.sha(pr),impl.sha(pc))==(impl.NUM_MANIFEST_SHA,impl.NUM_REPORT_SHA,impl.NUM_RECEIPT_SHA),'numeric authority SHA')
    require((impl.sha(sm),impl.sha(sr),impl.sha(sc))==(impl.PHYS_MANIFEST_SHA,impl.PHYS_REPORT_SHA,impl.PHYS_RECEIPT_SHA),'physical authority SHA')
    gm=impl.read_json(gp,impl.GOAL_PREP); goal_report=impl.read_json(gr,impl.GOAL_REPORT); goal_receipt=impl.read_json(gc,impl.GOAL_RECEIPT)
    nm=impl.read_json(pm,impl.NUM_MANIFEST_SHA); num_report=impl.read_json(pr,impl.NUM_REPORT_SHA); num_receipt=impl.read_json(pc,impl.NUM_RECEIPT_SHA)
    smv=impl.read_json(sm,impl.PHYS_MANIFEST_SHA); phys_report=impl.read_json(sr,impl.PHYS_REPORT_SHA); phys_receipt=impl.read_json(sc,impl.PHYS_RECEIPT_SHA)
    goalmod=impl.strict_module(impl.ROOT/'experiments/spectral_reachability'/(impl.GOAL_SOURCE+'.py'),impl.GOAL_SOURCE_SHA,'goal_v4')
    nummod=impl.strict_module(impl.ROOT/'experiments/spectral_reachability'/(impl.NUM+'.py'),impl.NUM_SOURCE_SHA,'numeric_v4')
    wrapper=impl.strict_module(impl.ROOT/'experiments/spectral_reachability/consume_contact_words_MAIN_20261028_v1.py',impl.PHYS_CONSUMER_SHA,'physical_v4')
    require(goalmod.consume(gm,gr,impl.GOAL_REPORT)==goal_receipt,'goal report/receipt exact return')
    require(nummod.consume(nm,pr,impl.NUM_REPORT_SHA)==num_receipt,'numeric report/receipt exact return')
    require(wrapper.consume(smv,impl.PHYS_MANIFEST_SHA,sr,impl.PHYS_REPORT_SHA)==phys_receipt,'physical report/receipt exact return')
    require(goal_receipt['status']=='all4_saved_goal_scores_verified' and num_receipt['status']=='all4_saved_public_action_words_verified' and phys_receipt['status']=='all4_saved_native_words_verified','all strict statuses')
    goalrec=goal_report['goal_patch']; require(goalrec['shape']==list(impl.SHAPE) and impl.sha(goalrec['path'])==goalrec['sha256'] and Path(goalrec['path']).stat().st_size==goalrec['bytes'],'goal patch SHA/bytes')
    prediction={w:num_report['artifacts'][f'{w}_1_visual_out'] for w in impl.WORDS}
    for rec in prediction.values(): require(rec['shape']==[3,*impl.SHAPE] and impl.sha(rec['path'])==rec['sha256'] and Path(rec['path']).stat().st_size==rec['bytes'],'prediction SHA/bytes')
    require(phys_report['status']=='all4_native_word_traces_recorded' and phys_report['global_error'] is None and [x['arm'] for x in phys_report['rows']]==list(impl.WORDS),'physical rows')
    terminals={}
    for row in phys_report['rows']:
        rec=next(x for x in row['records'] if x['frame']==20); base=Path(sr).parent/row['arm']/'frame_20'; files=rec['files']; members={k:base/k for k in ('rgb.npy','full_public_state.json','transition.json')}
        for k,p in members.items(): require(impl.sha(p)==files[k]['sha256'] and p.stat().st_size==files[k]['bytes'],'terminal SHA/bytes')
        terminals[row['arm']]=dict(rgb=dict(path=str(members['rgb.npy'].resolve()),sha256=files['rgb.npy']['sha256'],bytes=files['rgb.npy']['bytes']),native=dict(state=rec['state'],proprio=rec['proprio'],state_file=str(members['full_public_state.json'].resolve()),transition_file=str(members['transition.json'].resolve()),state_sha=files['full_public_state.json']['sha256'],transition_sha=files['transition.json']['sha256']))
    return dict(gm=gm,gr=goal_receipt,greport=goal_report,nm=nm,nr=num_receipt,nreport=num_report,sm=smv,sr=phys_receipt,sreport=phys_report,goalrec=goalrec,prediction=prediction,terminals=terminals,model_spec=nm['model_spec'],paths=dict(goal_prepare=str(Path(gp).resolve()),goal_report=str(Path(gr).resolve()),goal_receipt=str(Path(gc).resolve()),num_manifest=str(Path(pm).resolve()),num_report=str(Path(pr).resolve()),num_receipt=str(Path(pc).resolve()),phys_manifest=str(Path(sm).resolve()),phys_report=str(Path(sr).resolve()),phys_receipt=str(Path(sc).resolve())))
def prepare(*values):
    p=paths(values)
    # IMPORTANT: v2 internal ABI is goal,numeric,physical.
    old_join=impl.exact_authorities
    try:
        impl.exact_authorities=fixed_authorities
        value=impl.prepare(p['goal_prepare'],p['goal_report'],p['goal_receipt'],
            p['numeric_manifest'],p['numeric_report'],p['numeric_receipt'],
            p['physical_manifest'],p['physical_report'],p['physical_receipt'])
    finally:
        impl.exact_authorities=old_join
    value['schema']=TAG; value['protocol']=protocol(); value['authority_order']=list(ORDER)
    value['v4_source_sha256']=impl.sha(Path(__file__))
    value['bindings'][str(Path(__file__).resolve())]=impl.sha(Path(__file__))
    value['bindings'][str(ROOT/'tests'/('test_'+TAG+'.py'))]=impl.sha(ROOT/'tests'/('test_'+TAG+'.py'))
    value['bindings'][str(ROOT/'innerdocs'/(TAG+'.md'))]=impl.sha(ROOT/'innerdocs'/(TAG+'.md'))
    return value
def verify(value):
    require(value['schema']==TAG and value['protocol']==protocol() and value['authority_order']==list(ORDER),'v4 manifest/order')
    require(all(impl.sha(p)==h for p,h in value['bindings'].items()),'v4 binding drift')
    q=value['authority']['paths']
    fresh=prepare(q['goal_prepare'],q['phys_manifest'],q['phys_report'],q['phys_receipt'],q['num_manifest'],q['num_report'],q['num_receipt'],q['goal_report'],q['goal_receipt'])
    # authority paths in the delegate are canonicalized under its internal names;
    # compare all fields after replacing only v4 identity fields.
    require(fresh==value,'full v4 authority recomputation')
def run_cuda(value,out,confirm=False):
    verify(value); old_tag,old_protocol,old_verify=impl.TAG,impl.protocol,impl.verify
    try:
        impl.TAG=TAG; impl.protocol=protocol; impl.verify=verify
        return impl.run_cuda(value,out,confirm)
    finally: impl.TAG,impl.protocol,impl.verify=old_tag,old_protocol,old_verify
def consume(value,report,report_sha):
    verify(value); old_tag,old_protocol,old_verify=impl.TAG,impl.protocol,impl.verify
    try:
        impl.TAG=TAG; impl.protocol=protocol; impl.verify=verify
        return impl.consume(value,report,report_sha)
    finally: impl.TAG,impl.protocol,impl.verify=old_tag,old_protocol,old_verify
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('command',choices=('prepare','verify','cuda','consume')); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--manifest-sha256'); p.add_argument('--output',type=Path); p.add_argument('--report',type=Path); p.add_argument('--report-sha256'); p.add_argument('--confirm-host-cuda',action='store_true'); p.add_argument('--authority',nargs=9); a=p.parse_args(argv)
    if a.command=='prepare': require(a.authority is not None,'authority order is goal(3), physical(3), numeric(3)'); value=prepare(*a.authority); impl.write_new(a.manifest,value)
    else:
        require(a.manifest_sha256 and impl.sha(a.manifest)==a.manifest_sha256,'actual v4 manifest SHA'); value=impl.read_json(a.manifest,a.manifest_sha256)
        if a.command=='verify': verify(value); result={'status':'v4_verified'}
        elif a.command=='cuda': result=run_cuda(value,a.output,a.confirm_host_cuda)
        else: require(a.report is not None and a.report_sha256 and a.output is not None,'report/output required'); result=consume(value,a.report,a.report_sha256); impl.write_new(a.output,result)
    print(json.dumps({'status':result['status'],'scientific_admission':False},sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
