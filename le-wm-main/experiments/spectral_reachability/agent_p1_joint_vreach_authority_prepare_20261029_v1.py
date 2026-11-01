"""Strict original-winner authority preparation; not a CUDA host."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import shutil

from experiments.spectral_reachability import agent_p1_joint_vreach_multistart_20261029_v1 as method
from experiments.spectral_reachability import agent_p1_joint_vreach_signal_bridge_20261029_v1 as bridge
from experiments.spectral_reachability import agent_p1_joint_vreach_native_callback_20261029_v1 as callback
from experiments.spectral_reachability import agent_p1_joint_vreach_saved_audit_20261029_v1 as audit
from experiments.spectral_reachability import agent_p1_two_packet_winner_authority_20261029_v1 as authority
from experiments.spectral_reachability import agent_p3_parent_r_native_components_20261029_v1 as component

NAME='agent_p1_joint_vreach_authority_prepare_20261029_v1'
F,core,signal=method.F,method.core,method.F.signal
ROOT=F.ROOT
require,sha,raw=F.require,F.sha,F.raw
PINS=((method,'90893b6fa0925a9e452a3fbf14cb81d48042b1349e3bdb5cc6237c9fa3f36159'),
      (bridge,'22450104687378143f9e7eced355b8a5aef391836b5c8896461d44295738456a'),
      (callback,'72ad731d3be856c0908db83bc5af155f659e56b0c3966c754c2310251603cf5c'),
      (audit,'8c0f7453e3181dfe36cd573c327da4b5eddd6c75710b0fc5f1884b85ef4228db'),
      (authority,'a910df6135edd00b1dfce8df8a7bf7e93c1712c6cddde5381945c821d8e83054'),
      (component,'ce48a8c665ad5434fe65f5238b303570ffc18183956d24a0c830864fb5f12c7a'))
MAX_BYTES=128*1024**2
DISK_FLOOR=200*1024**2


def contract():
    return dict(schema=NAME+'.contract', method=method.contract(), fixed_slots=4, arms=8,
        candidates=160, source='full original mixedmode strict reconstructed by original authority',
        scope='prepare only; no run or model certificate', production_ready=False,
        P1_admission=False, freshness=False, physical_release=False, FLOPs=None)


def own_files():
    return [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]


def add(bindings,path,h):
    path=str(Path(path).resolve())
    require(path not in bindings or bindings[path]==h, 'source binding collision: '+path)
    bindings[path]=h


def build(source_output,source_report_sha):
    core.hash_value(source_report_sha)
    source=Path(source_output)
    require(source.is_absolute() and source.resolve()==source and source.is_dir(), 'absolute literal source output')
    for module,h in PINS: require(sha(module.__file__)==h, 'sealed method/component source')
    component.check_sources()
    a=authority.material(str(source),source_report_sha)
    require(a['schema']==authority.NAME+'.material' and a['source_report_sha256']==source_report_sha and
        a['whole_producer_strict_reconstructed'] is True and a['P1_admission'] is False and
        a['physical_release'] is False and len(a['rows'])==4, 'original complete authority')
    for i,row in enumerate(a['rows']):
        q=row['request'];method.original.request(q)
        require(q['slot']==i, 'four original fixed slots in order')
    bindings=deepcopy(a['bindings'])
    for module,h in PINS:
        add(bindings,module.__file__,h)
        stem=module.__name__.rsplit('.',1)[-1]
        for f in (ROOT/'tests'/('test_'+stem+'.py'), ROOT/'innerdocs'/(stem+'.md')):
            add(bindings,f,sha(f))
    for module in (F,core,signal,F.transport,method.original,method.compiler,bridge.parent,
                   bridge.native,component.kernel,component.kernel.original):
        add(bindings,module.__file__,sha(module.__file__))
    for name,h in component.PINS.items():
        add(bindings,Path(component.__file__).with_name(name),h)
    for f in own_files(): add(bindings,f,sha(f))
    installed=F.installed_class_sources()
    for v in installed.values(): add(bindings,v['path'],v['sha256'])
    F.check_bindings(bindings)
    return dict(schema=NAME+'.prepare',arguments=dict(source_output=str(source),source_report_sha=source_report_sha),
        contract=contract(),authority=a,bindings=bindings,installed_class_sources=installed,
        production_ready=False,P1_admission=False,freshness=False,physical_release=False)


def verify(prepared,expected):
    core.hash_value(expected);path=Path(prepared)
    require(path.is_absolute() and path.resolve()==path and sha(path)==expected, 'exact own prepare SHA')
    p=signal.read(path,MAX_BYTES)
    require(p['schema']==NAME+'.prepare' and raw(p)==raw(build(**p['arguments'])) and sha(path)==expected,
        'full authority and source rebuild; unchanged prepare')
    return p


def write(path,value):
    p=Path(path); data=raw(value)
    require(p.is_absolute() and p.parent==ROOT/'results' and p.resolve()==p and p.name.startswith(NAME+'_')
        and not p.exists() and not p.is_symlink() and len(data)<=MAX_BYTES, 'exclusive bounded prepare path')
    require(shutil.disk_usage(p.parent).free >= DISK_FLOOR+len(data), 'disk floor')
    with p.open('xb') as f:
        require(f.write(data)==len(data),'complete prepare write');f.flush();os.fsync(f.fileno())
    signal.fsync_dir(p.parent)
    return dict(path=str(p),sha256=sha(p),production_ready=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--source-output',required=True);p.add_argument('--source-report-sha',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('verify');p.add_argument('--prepared',required=True);p.add_argument('--prepare-sha',required=True)
    a=parser.parse_args()
    out=write(a.output,build(a.source_output,a.source_report_sha)) if a.command=='prepare' else dict(
        verified=True,production_ready=verify(a.prepared,a.prepare_sha)['production_ready'])
    print(raw(out).decode(),flush=True)


if __name__=='__main__': main()
