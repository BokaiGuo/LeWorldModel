"""E3 HOST launcher and saved-evidence integration; strict remains unavailable."""
import argparse
from pathlib import Path
import time

from experiments.spectral_reachability import agent_e3_continuous_host_launcher_20261029_v1 as launch
from experiments.spectral_reachability import agent_e3_continuous_saved_consumer_20261029_v1 as saved
from experiments.spectral_reachability import agent_e3_partial_guard_saved_20261029_v1 as partial_guard

NAME='agent_e3_integrated_host_saved_20261029_v1'
need=launch.need


def partial_native_evidence(output, events, episode):
    """Check completed observation snapshots without inventing missing events."""
    frames={};observed=set();previous=-1
    for e in events:
        if e['origin']!='physical':continue
        if e['resource']=='native_frame':
            frame=e['detail']['frame'];name=frame['file']
            need(type(name) is str and Path(name).name==name and name not in frames,'unique partial frame')
            frames[name]=(e,frame)
        if e['resource']!='native_fresh_observation':continue
        detail=e['detail'];r=detail['receipt'];name=Path(r['path']).name
        need(detail['episode_id']==episode and name in frames and name not in observed
             and Path(r['path'])==Path(output)/name and previous<frames[name][0]['index']<e['index']
             and r['observed_event_index']==e['index'] and r['phase'] in ('warmup','plan')
             and type(r['step']) is int and r['step']>=0,'partial observation causal identity')
        frame=frames[name][1]
        need(r['file_sha256']==frame['sha256'] and r['pixel_sha256']==frame['pixel_sha256'],
             'partial observation exact frame seals')
        saved.validate_snapshot(r['snapshot'],r['state'])
        observed.add(name);previous=e['index']
    failures=[e for e in events if e['origin']=='physical' and e['resource']=='native_episode_failure']
    need(len(failures)<=1,'unique episode failure')
    terminal_returns=[e for e in events if e['origin']=='physical' and e['resource']=='terminal_row' and e['kind']=='return']
    row_verified=False
    if terminal_returns:
        from experiments.spectral_reachability import agent_e3_terminal_row_20261029_v1 as terminal
        row=terminal.read(events);need(row['episode_id']==episode,'partial terminal row identity')
        if failures:
            err=failures[0]['detail']['error']
            need(row.get('status')=='failed' and row.get('error')==err
                 and failures[0]['index']<terminal_returns[0]['index'], 'episode failure preserved in terminal row')
        row_verified=True
    return dict(observations=len(observed),unmatched_frame_count=len(frames)-len(observed),
                episode_failure_observed=bool(failures),terminal_row_verified=row_verified,
                physics_replayed=False,full_failure_semantics=False)


def successful_lifecycle_chronology(events):
    """Require loader and actual environment lifecycle within paired guards."""
    def one(resource):
        matches=[e for e in events if e['origin']=='physical' and e['resource']==resource]
        need(len(matches)==1 and matches[0]['kind']=='record','unique success '+resource)
        return matches[0]['index']
    loader=one('loader_receipt');before=one('guard_before');after=one('guard_after')
    cleanup_start=one('native_cleanup_intent');cleanup=one('native_cleanup_return')
    done=one('launcher_guarded_return')
    native=[e for e in events if e['origin']=='physical' and e['resource'].startswith('native_')
            and e['resource'] not in ('native_cleanup_intent','native_cleanup_return','native_meter/close')]
    need(native and loader<before<native[0]['index']<cleanup_start<cleanup<after<done,
         'loader guard native cleanup chronology')
    need(all(before<e['index']<cleanup_start for e in native),
         'all native records occur before cleanup')
    need(not any(e['resource'] in ('native_cleanup_failure','native_episode_failure','guard_failure','launcher_failure')
                 for e in events if e['origin']=='physical'),'no success failure records')
    return dict(loader_before_guard=True,native_between_guard_and_cleanup=True,
                guard_after_cleanup=True,full_saved_strict=False)


def encoder_ledger(events, complete):
    """Rebuild encoder/projector event attempts; incomplete work is not refunded."""
    need(type(complete) is bool,'boolean completion')
    expected=[('encoder','intent'),('encoder','return'),('projector','intent'),
              ('projector','return'),('encoding_complete','record')]
    rows=[];last=-1;incomplete=False
    for boundary in (10,15):
        ops=[e for e in events if e['origin']=='encoding' and e['detail'].get('boundary')==boundary]
        begins=[e for e in events if e['origin']=='physical' and e['resource']=='native_e3_encode_intent'
                and e['detail'].get('boundary')==boundary]
        ends=[e for e in events if e['origin']=='physical' and e['resource']=='native_e3_encode_return'
              and e['detail'].get('boundary')==boundary]
        need(len(begins)<=1 and len(ends)<=1,'unique encoder boundary')
        if not begins:
            need(not ops and not ends,'no encoding without physical intent');continue
        need(not incomplete and last<begins[0]['index'] and len(ops)<=len(expected),
             'encoder boundaries in causal order')
        need([(e['resource'],e['kind']) for e in ops]==expected[:len(ops)],'encoder lifecycle exact prefix')
        prior=begins[0]['index'];pending=None;counts={k:dict(attempted=0,returned=0) for k in ('encoder','projector')}
        for e in ops:
            need(prior<e['index'],'encoder event ordering');prior=e['index']
            if e['kind']=='intent':
                need(pending is None and e['token']==e['index'],'encoder actual intent token')
                pending=(e['resource'],e['token']);counts[e['resource']]['attempted']+=1
            elif e['kind']=='return':
                need(pending==(e['resource'],e['token']),'encoder matching return token')
                pending=None;counts[e['resource']]['returned']+=1
        if ends:
            need(len(ops)==len(expected) and pending is None and prior<ends[0]['index'],
                 'encoder return follows full recorded IO');prior=ends[0]['index']
        else:incomplete=True
        if complete:need(bool(ends),'successful episode has no incomplete encoding')
        rows.append(dict(boundary=boundary,counts=counts,pending=None if pending is None else pending[0],
                         completed=bool(ends)))
        last=prior
    known=[e for e in events if e['origin']=='encoding']
    need(all(type(e['detail'].get('boundary')) is int and e['detail']['boundary'] in (10,15) for e in known),
         'no unbound encoding events')
    return dict(boundaries=rows,model_encoder_instance_identity_verified=False,
                reason='native events do not seal module instance identity')


def _authority(path,sha):
    binding=launch.read_authority(path,sha)
    sources=dict(binding['prepared']['bindings_sha256'])
    launch.host.source.merge(sources,binding['additional_sources'])
    required=[str(Path(m.__file__).resolve()) for m in (launch,saved,partial_guard)]
    required.append(str(Path(__file__).resolve()))
    launch.episode.old.guard.validate_sources(sources,required)
    return binding


def run(authority,authority_sha,output,authorization):
    """Use existing real HOST loader, guard and same-environment episode.

    This is not an authorization or a production-ready certification. The
    authority must additionally preseal all four integration source files.
    """
    need(authorization==launch.AUTH,'explicit HOST CUDA authorization')
    started=time.monotonic()
    return launch._run_lifecycle(output,started,lambda:_authority(authority,authority_sha),launch._host_perform)


def inspect(output,report_sha,completion_sha,authority,authority_sha):
    """Join source-pinned host, saved success inspector and partial guard.

    A transport-complete result is not promoted to a full strict receipt.
    """
    base=saved.consume(output,report_sha,completion_sha,authority,authority_sha)
    bound=None
    if base['status']!='partial' or base['authority_verified_here']:
        bound=_authority(authority,authority_sha)
    if base['status']=='partial':
        evidence=partial_guard.inspect(output,report_sha,completion_sha,authority,authority_sha)
        if bound is not None:
            data=launch.pub.base.read_bytes(Path(output)/'events.jsonl',launch.pub.base.CAP)
            report=launch.pub.base.read_bytes(Path(output)/'report.json',launch.pub.FINAL_CAP)
            need(launch.pub.base.fsha(report)==report_sha and
                 launch.pub.base.fsha(data)==launch.pub.base.decode(report)['journal_sha256'],
                 'stable partial native journal')
            events=[launch.pub.base.decode(line) for line in data.splitlines()]
            native=partial_native_evidence(output,events,'cell%02d'%bound['cell'])
        else:native=None;events=[]
    else:
        evidence=None
        report=launch.pub.base.read_bytes(Path(output)/'report.json',launch.pub.FINAL_CAP)
        data=launch.pub.base.read_bytes(Path(output)/'events.jsonl',launch.pub.base.CAP)
        need(launch.pub.base.fsha(report)==report_sha and
             launch.pub.base.fsha(data)==launch.pub.base.decode(report)['journal_sha256'],
             'stable successful native journal')
        events=[launch.pub.base.decode(line) for line in data.splitlines()]
        native=partial_native_evidence(output,events,'cell%02d'%bound['cell'])
        need(native['unmatched_frame_count']==0 and native['terminal_row_verified'],
             'success requires every frame and terminal row')
        native['lifecycle']=successful_lifecycle_chronology(events)
    encoding=encoder_ledger(events,base['status']!='partial')
    return dict(schema=NAME+'.inspect',status=base['status'],cell=None if bound is None else bound['cell'],
                saved=base,partial_guard=evidence,partial_native=native,full_saved_strict=False,scientific_admission=False,
                encoding_ledger=encoding,missing=['model/encoder instance identity not recorded',
                         'complete native snapshot and failure-path semantic validation',
                         'same-episode real loader/model/source guard end-to-end fixture'])


def require_strict(*args):
    result=inspect(*args)
    need(result['full_saved_strict'] is True,'full saved strict unavailable: '+', '.join(result['missing']))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('run','inspect','require-strict'):
        p=sub.add_parser(name)
        for option in ('authority','authority-sha256','output'):p.add_argument('--'+option,required=True)
        if name=='run':p.add_argument('--authorization',required=True)
        else:
            p.add_argument('--report-sha256',required=True);p.add_argument('--completion-sha256',required=True)
    a=parser.parse_args()
    if a.command=='run':result=run(a.authority,a.authority_sha256,a.output,a.authorization)
    else:
        fn=inspect if a.command=='inspect' else require_strict
        result=fn(a.output,a.report_sha256,a.completion_sha256,a.authority,a.authority_sha256)
    print(launch.host.raw(result).decode())


if __name__=='__main__':main()
