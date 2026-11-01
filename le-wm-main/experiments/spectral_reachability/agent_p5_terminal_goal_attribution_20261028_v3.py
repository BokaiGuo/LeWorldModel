"""v3 order-safe wrapper for the frozen strict terminal RGB diagnostic.

The only contract change from v2 is the explicit authority order:
goal(3), physical(3), numeric(3). v2 and all assets remain untouched.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from experiments.spectral_reachability import agent_p5_terminal_goal_attribution_20261028_v2 as impl

TAG='agent_p5_terminal_goal_attribution_20261028_v3'
ROOT=impl.ROOT
ORDER=('goal_prepare','goal_report','goal_receipt','physical_manifest','physical_report','physical_receipt','numeric_manifest','numeric_report','numeric_receipt')

def require(ok,msg):
    if not ok: raise ValueError(msg)

def protocol():
    p=dict(impl.protocol()); p['schema']=TAG; p['authority_order']=list(ORDER); p['v2_unchanged']=True
    return p

def _paths(values):
    require(len(values)==9,'authority requires goal(3), physical(3), numeric(3)')
    return dict(zip(ORDER,map(Path,values)))

def prepare(*values):
    """Public positional ABI: goal triple, then physical triple, then numeric triple."""
    p=_paths(values)
    # v2's internal function is called in its own historical order; this is the
    # sole translation point and is covered by real-path order tests.
    value=impl.prepare(p['goal_prepare'],p['goal_report'],p['goal_receipt'],
        p['physical_manifest'],p['physical_report'],p['physical_receipt'],
        p['numeric_manifest'],p['numeric_report'],p['numeric_receipt'])
    value['schema']=TAG; value['protocol']=protocol(); value['authority_order']=list(ORDER)
    value['v2_source_sha256']=impl.sha(ROOT/'experiments/spectral_reachability/agent_p5_terminal_goal_attribution_20261028_v2.py')
    value['bindings'][str(Path(__file__).resolve())]=impl.sha(Path(__file__))
    value['bindings'][str(ROOT/'tests'/('test_'+TAG+'.py'))]=impl.sha(ROOT/'tests'/('test_'+TAG+'.py'))
    value['bindings'][str(ROOT/'innerdocs'/(TAG+'.md'))]=impl.sha(ROOT/'innerdocs'/(TAG+'.md'))
    return value

def verify(value):
    require(value['schema']==TAG and value['protocol']==protocol() and value['authority_order']==list(ORDER),'v3 manifest/order')
    require(all(impl.sha(p)==h for p,h in value['bindings'].items()),'v3 binding drift')
    a=value['authority']; q=a['paths']
    expected=prepare(q['goal_prepare'],q['goal_report'],q['goal_receipt'],q['phys_manifest'],q['phys_report'],q['phys_receipt'],q['num_manifest'],q['num_report'],q['num_receipt'])
    # v2 stores compact path keys; order-normalized v3 output must be identical.
    require(expected==value,'full v3 authority recomputation')

def run_cuda(value,out,confirm=False):
    # Make the frozen implementation emit v3 schema/protocol while retaining
    # its v2 runner; no source/global file is changed and this process is fresh.
    verify(value); old_tag,old_protocol,old_verify=impl.TAG,impl.protocol,impl.verify
    try:
        impl.TAG=TAG; impl.protocol=protocol; impl.verify=verify
        return impl.run_cuda(value,out,confirm)
    finally:
        impl.TAG,impl.protocol,impl.verify=old_tag,old_protocol,old_verify

def consume(value,report,report_sha):
    verify(value); old_tag,old_protocol,old_verify=impl.TAG,impl.protocol,impl.verify
    try:
        impl.TAG=TAG; impl.protocol=protocol; impl.verify=verify
        return impl.consume(value,report,report_sha)
    finally:
        impl.TAG,impl.protocol,impl.verify=old_tag,old_protocol,old_verify

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('command',choices=('prepare','verify','cuda','consume'))
    p.add_argument('--manifest',type=Path,required=True); p.add_argument('--manifest-sha256'); p.add_argument('--output',type=Path)
    p.add_argument('--report',type=Path); p.add_argument('--report-sha256'); p.add_argument('--confirm-host-cuda',action='store_true'); p.add_argument('--authority',nargs=9)
    a=p.parse_args(argv)
    if a.command=='prepare':
        require(a.authority is not None,'authority order is goal(3), physical(3), numeric(3)'); value=prepare(*a.authority); impl.write_new(a.manifest,value)
    else:
        require(a.manifest_sha256 and impl.sha(a.manifest)==a.manifest_sha256,'actual v3 manifest SHA'); value=impl.read_json(a.manifest,a.manifest_sha256)
        if a.command=='verify': verify(value); result={'status':'v3_verified'}
        elif a.command=='cuda': result=run_cuda(value,a.output,a.confirm_host_cuda)
        else: require(a.report is not None and a.report_sha256 and a.output is not None,'report/output required'); result=consume(value,a.report,a.report_sha256); impl.write_new(a.output,result)
    print(json.dumps({'status':result['status'],'scientific_admission':False},sort_keys=True)); return 0

if __name__=='__main__': raise SystemExit(main())
