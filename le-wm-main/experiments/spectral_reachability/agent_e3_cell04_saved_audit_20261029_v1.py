"""Fixed cell04 saved-only audit; caller must confirm terminal publication."""
import argparse
from pathlib import Path

from experiments.spectral_reachability import agent_e3_chunked_identity_host_20261029_v1 as host
from experiments.spectral_reachability import agent_e3_native_saved_semantics_20261029_v1 as native

NAME='agent_e3_cell04_saved_audit_20261029_v1'
ROOT=host.launch.module_census.ROOT
OUTPUT=str(ROOT/'results/agent_e3_chunked_identity_host_20261029_v1_CELL04_MAIN01')
AUTH=str(ROOT/'results/agent_e3_cell04_feedback_prepare_20261029_v1_MAIN01.json')
AUTH_SHA='8d8ad3a7084a1db2e0a4309707482eaaf42d9ce72ab1eddb2acd72df52477ca2'
ACK='MAIN_CONFIRMED_CELL04_TERMINAL'
need=host.need
base=native.pub.base


def project(events,audit):
    one=lambda name:[e for e in events if e['origin']=='physical' and e['resource']==name]
    markers=one('launcher_authority_verified')
    if audit['transport_complete']:
        need(len(markers)==1 and markers[0]['detail']==dict(cell=4,manifest_path=AUTH,manifest_sha256=AUTH_SHA),'fixed authority and cell marker')
    roots=one('native_actual_root_guard');need(len(roots)<=1,'unique actual root guard')
    cps=one('native_actual_cp5_guard');need(len(cps)<=1,'unique cp5 guard')
    evaluations=one('native_initial_native_evaluation');need(len(evaluations)<=1,'unique initial native evaluation')
    row_events=[e for e in one('terminal_row') if e['kind']=='return']
    row=native.terminal.read(events) if row_events else None
    summary=dict(status=None if row is None else row.get('status'),initial_success=None,initial_distance=None,
        final_success=None,final_distance=None,goal=None,root_state=None,cp5_state=None)
    if roots:
        guard=roots[0]['detail'];need(guard['episode_id']=='cell04','cell04 root guard')
        summary.update(goal=guard['new_goal'],root_state=guard['root_observation']['state'])
    if cps:summary['cp5_state']=cps[0]['detail']['observation']['state']
    if evaluations:
        need(roots and cps and row is not None,'initial evaluation requires goal, cp5, terminal')
        evaluation=evaluations[0]['detail'];success,distance=native.prior.native(summary['goal'],summary['cp5_state'])
        need(evaluation['episode_id']=='cell04' and evaluation['boundary']==5 and
             evaluation['success'] is success and evaluation['distance']==distance and
             row.get('initial_success') is success and row.get('initial_native_distance')==distance,'independent cp5 native evaluation')
        summary.update(initial_success=success,initial_distance=distance)
    if row is not None and roots and row.get('final_state') is not None and row.get('native_success') is not None:
        success,distance=native.prior.native(summary['goal'],row['final_state'])
        need(row['native_success'] is success,'final native success exact')
        summary.update(final_success=success,final_distance=distance)
    return summary


def build(report_sha,completion_sha,confirmation):
    need(confirmation==ACK,'MAIN terminal confirmation required')
    for digest in (report_sha,completion_sha):
        need(type(digest) is str and len(digest)==64 and all(x in '0123456789abcdef' for x in digest),'external SHA')
    # Original inspector authenticates authority sources, transport, identity and reached feedback nodes.
    inspected=host.inspect(OUTPUT,report_sha,completion_sha,AUTH,AUTH_SHA)
    audit=native.build(OUTPUT,report_sha,completion_sha,'cell04')
    report=base.decode(base.read_bytes(Path(OUTPUT)/'report.json',native.pub.FINAL_CAP))
    journal=base.read_bytes(Path(OUTPUT)/'events.jsonl',base.CAP)
    need(base.fsha(journal)==report['journal_sha256'],'stable journal')
    events=[base.decode(line) for line in journal.splitlines()]
    summary=project(events,audit)
    need(base.fsha(base.read_bytes(Path(OUTPUT)/'events.jsonl',base.CAP))==base.fsha(journal),'stable source journal')
    return dict(schema=NAME+'.audit',cell=4,task='r0.g1.h1.dDxy',report_sha256=report_sha,
        completion_sha256=completion_sha,authority_sha256=AUTH_SHA,source_guard_inspect_returned=True,
        identity_binding_chunks_complete=inspected['binding_chunks_complete'],saved_native=audit,
        summary=summary,full_saved_strict=False,freshness=False,scientific_admission=False,
        physics_replayed=False,neural_recomputed=False,
        caveats=['source-trusted saved identity is not independent process-memory attestation',
                 'partial and early-stop rows remain partial; no future-step padding',
                 'failure semantics and physical/neural parity remain incomplete'])


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    for mode in ('audit','verify'):
        x=s.add_parser(mode)
        for arg in ('report-sha256','completion-sha256','terminal-confirmation','saved'):x.add_argument('--'+arg,required=True)
        if mode=='verify':x.add_argument('--saved-sha256',required=True)
    x=p.parse_args();data=base.raw(build(x.report_sha256,x.completion_sha256,x.terminal_confirmation));path=Path(x.saved)
    need(path.is_absolute() and path.resolve()==path and path.parent==ROOT/'results' and path.name.startswith(NAME+'_'), 'own result path')
    if x.command=='audit':
        import os
        with path.open('xb') as f:need(f.write(data)==len(data),'complete audit output');f.flush();os.fsync(f.fileno())
        base.syncdir(path.parent);sha=base.fsha(data)
    else:
        saved=base.read_bytes(path,16*1024**2);need(base.fsha(saved)==x.saved_sha256 and saved==data,'exact saved reconstruction');sha=x.saved_sha256
    print(base.raw(dict(path=str(path),sha256=sha,verified=x.command=='verify',freshness=False,scientific_admission=False)).decode())


if __name__=='__main__':main()
