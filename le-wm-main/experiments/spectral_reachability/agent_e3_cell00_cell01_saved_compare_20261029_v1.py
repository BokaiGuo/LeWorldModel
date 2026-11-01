"""Saved native semantics and fixed known-root cell00/cell01 comparison.

Requires externally confirmed terminal publications; never runs CUDA or physics.
"""
import argparse
import hashlib
import math
import os
from pathlib import Path
import struct

from experiments.spectral_reachability import agent_e3_native_saved_semantics_20261029_v1 as native
from experiments.spectral_reachability import agent_e3_terminal_row_20261029_v1 as terminal

NAME='agent_e3_cell00_cell01_saved_compare_20261029_v1'
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'results/agent_e3_chunked_identity_host_20261029_v1_MAIN06'
BASE_REPORT='3ee72bd9b3ade29020083359f878b67accf4ec531dbccefa2acb3d94f1065bea'
BASE_COMPLETION='9adfc093f3b6293c92e56ec245daeb431d0a0c59548805dcd3d67560f745806f'
AUTH0=dict(path=str(ROOT/'results/agent_e3_observed_authority_20261029_v1_MAIN02.json'),sha256='ea2d7d747e02c0054eca90092ca947e136cdba10aca3a204dfa85d8b01ec64e9')
AUTH1=dict(path=str(ROOT/'results/agent_e3_cell01_feedback_prepare_20261029_v1_MAIN01.json'),sha256='0e525436e311dc8ffcb4cf7ec04fd0ffcbccafedc3ea52391fe9e888bb5105f8')
TARGET=ROOT/'results/agent_e3_chunked_identity_host_20261029_v1_CELL01_MAIN01'
ACK='MAIN_CONFIRMED_CELL01_TERMINAL'
base=native.pub.base
need=base.need


def unique(events, resource):
    found=[e for e in events if e['origin']=='physical' and e['resource']==resource]
    need(len(found)<=1,'unique '+resource)
    return found[0] if found else None


def validate_dx_return(d):
    bits=bytes.fromhex(d['raw_bits']);need(len(bits)==8,'float32 action bits')
    if d['step']>5:need(bits[4:]==b'\0'*4,'Dx task action y must be positive float32 zero')
    need(struct.pack('<2f',*d['command'])==bits,'returned command raw bits')


def inspect_saved(path,report_sha,completion_sha,episode,cell,authority):
    audit=native.build(path,report_sha,completion_sha,episode)
    data=base.read_bytes(Path(path)/'events.jsonl',base.CAP)
    report=base.decode(base.read_bytes(Path(path)/'report.json',native.pub.FINAL_CAP))
    need(base.fsha(data)==report['journal_sha256'], 'stable journal binding')
    events=[base.decode(x) for x in data.splitlines()]
    authority_bytes=base.read_bytes(Path(authority['path']),16*1024**2)
    need(base.fsha(authority_bytes)==authority['sha256'], 'external authority bytes')
    value=base.decode(authority_bytes);need(value['cell']==cell,'authority cell')
    marker=unique(events,'launcher_authority_verified')
    if marker is not None:
        need(marker['kind']=='record' and marker['detail']==dict(cell=cell,manifest_path=authority['path'],manifest_sha256=authority['sha256']), 'recorded authority identity')
    if audit['transport_complete']:need(marker is not None,'complete needs authority marker')
    obs={}
    for event in events:
        if event['origin']=='physical' and event['resource']=='native_fresh_observation':
            r=event['detail']['receipt']
            if r['phase']=='plan' and r['step'] in (0,5,10,15,20):obs[r['step']]=r
    guard=unique(events,'native_actual_root_guard')
    checkpoint=unique(events,'native_actual_cp5_guard')
    returns=[e for e in events if e['origin']=='physical' and e['resource']=='native_control_return']
    if cell==1:
        for e in returns:
            validate_dx_return(e['detail'])
    n=audit['native']
    return dict(cell=cell,episode=episode,authority=authority,audit=audit,observations=obs,
        guard=None if guard is None else guard['detail'],cp5=None if checkpoint is None else checkpoint['detail'],
        dispatch=n['dispatched'],returned=n['returned'],terminal_status=n['terminal_status'])


def distance(receipt,goal):
    state=receipt['state'];need(type(state) is list and len(state)==7 and type(goal) is list and len(goal)==7 and
        all(type(x) in (int,float) and math.isfinite(x) for x in state+goal), 'native7 finite state/goal')
    return math.sqrt(sum((state[i]-goal[i])**2 for i in range(4)))


def compare(left,right):
    need(left['cell']==0 and right['cell']==1 and left['episode']=='cell00' and right['episode']=='cell01','fixed ordered cells')
    g0=left['guard'];g1=right['guard'];aligned=False;reasons=[]
    if g0 is None or g1 is None:
        reasons.append('missing_recorded_root_or_goal_guard')
    else:
        for key in ('original_root_sha256','original_goal','new_goal','new_goal_pixel_sha256','history_pixels'):
            need(g0[key]==g1[key],'same root/goal/history binding: '+key)
        a,b=g0['root_observation'],g1['root_observation']
        need(a['snapshot']==b['snapshot'] and a['state']==b['state'] and a['pixel_sha256']==b['pixel_sha256'], 'same original physical root')
        aligned=True
    rows=[]
    for step in (0,5,10,15,20):
        aa=left['observations'].get(step);bb=right['observations'].get(step)
        def side(receipt,g):
            if receipt is None or g is None:return None
            need(receipt['goal']==g['new_goal'] and receipt['goal_rgb_sha256']==g['new_goal_pixel_sha256'], 'observation task goal')
            return dict(state=receipt['state'],pixel_sha256=receipt['pixel_sha256'],
                        joint_xy_distance=distance(receipt,g['new_goal']))
        l=side(aa,g0);r=side(bb,g1)
        rows.append(dict(control_step=step,cell00=l,cell01=r,
                         paired_joint_xy_delta=None if not aligned or l is None or r is None else r['joint_xy_distance']-l['joint_xy_distance']))
    return dict(root_goal_aligned=aligned,blocked_reasons=reasons,checkpoints=rows,
        missing_not_filled=True,same_root_independent_sample=False,
        scope='descriptive same-time observed native joint XY; no early-stop padding, no goal or root averaging')


def build(report_sha,completion_sha,confirmation):
    need(confirmation==ACK,'explicit MAIN terminal confirmation required')
    for digest in (report_sha,completion_sha):
        need(type(digest) is str and len(digest)==64 and all(c in '0123456789abcdef' for c in digest),'external publication SHA')
    left=inspect_saved(str(BASE),BASE_REPORT,BASE_COMPLETION,'cell00',0,AUTH0)
    right=inspect_saved(str(TARGET),report_sha,completion_sha,'cell01',1,AUTH1)
    comparison=compare(left,right)
    return dict(schema=NAME+'.report',cell00=left,cell01=right,comparison=comparison,
        source_authority_bytes_checked=True,authority_sources_reverified=False,physics_replayed=False,
        neural_recomputed=False,full_saved_strict=False,freshness=False,scientific_admission=False,
        caveats=['terminal confirmation is caller supplied, not a process attestation',
            'native saved audit does not establish unrecorded effects, complete failure semantics, or independent physical/neural parity',
            'known root and unequal action domains; descriptive within-root comparison, not a general causal or fairness claim'])


def save(path,value):
    p=Path(path);data=base.raw(value)
    need(p.is_absolute() and p==p.resolve() and p.parent==ROOT/'results' and p.name.startswith(NAME+'_') and
         not p.exists() and not p.is_symlink() and len(data)<=16*1024**2,'exclusive owned output')
    with p.open('xb') as f:need(f.write(data)==len(data),'complete report');f.flush();os.fsync(f.fileno())
    base.syncdir(p.parent)
    return base.fsha(data)


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    for command in ('audit','verify'):
        x=s.add_parser(command)
        for option in ('report-sha256','completion-sha256','terminal-confirmation','saved'):x.add_argument('--'+option,required=True)
        if command=='verify':x.add_argument('--saved-sha256',required=True)
    x=p.parse_args();value=build(x.report_sha256,x.completion_sha256,x.terminal_confirmation)
    if x.command=='audit':sha=save(x.saved,value)
    else:
        data=base.read_bytes(Path(x.saved),16*1024**2)
        need(base.fsha(data)==x.saved_sha256 and data==base.raw(value),'complete saved reconstruction');sha=x.saved_sha256
    print(base.raw(dict(path=x.saved,sha256=sha,verified=x.command=='verify',freshness=False,scientific_admission=False)).decode())


if __name__=='__main__':main()
