"""Saved-only native snapshot and failure-prefix audit; never upgrades E3 gates."""
import argparse
from pathlib import Path

from experiments.spectral_reachability import agent_e3_transport_publication_20261029_v1 as pub
from experiments.spectral_reachability import agent_e3_continuous_saved_consumer_20261029_v1 as prior
from experiments.spectral_reachability import agent_e3_terminal_row_20261029_v1 as terminal

NAME='agent_e3_native_saved_semantics_20261029_v1'
need=pub.base.need


def snapshots(events):
    """Examine every nested saved native body snapshot, including duplicated receipts."""
    found=[]
    def visit(value,event,path):
        if type(value) is dict:
            if set(value)=={'state','state_float64_bits','bodies'}:
                prior.validate_snapshot(value,value['state'])
                found.append(dict(index=event['index'],resource=event['resource'],path=path))
                return
            for k,v in value.items():visit(v,event,path+'/'+k)
        elif type(value) is list:
            for i,v in enumerate(value):visit(v,event,path+'/'+str(i))
    for e in events:
        if e['origin']=='physical':visit(e['detail'],e,'/detail')
    return found


def audit(events,episode,complete):
    """Validate recorded native joins and failure prefixes; never infer omitted effects."""
    need(type(events) is list and type(episode) is str and bool(episode) and type(complete) is bool,'audit inputs')
    phy=[e for e in events if e['origin']=='physical']
    def get(name):return [e for e in phy if e['resource']=='native_'+name]
    def one(name):
        xs=get(name);need(len(xs)<=1,'duplicate '+name);return xs[0] if xs else None
    observed={};indices={};items=snapshots(events)
    for e in get('fresh_observation'):
        d=e['detail'];r=d['receipt'];key=(r['phase'],r['step'])
        need(d['episode_id']==episode and r['phase'] in ('warmup','plan') and type(r['step']) is int
             and 0<=r['step']<=20 and key not in observed and r['observed_event_index']==e['index'], 'observation identity')
        prior.validate_snapshot(r['snapshot'],r['state']);observed[key]=r;indices[key]=e['index']
    root=one('original_root_accepted');guard=one('actual_root_guard');cp5=one('actual_cp5_guard')
    links=[]
    if root:
        d=root['detail'];need(d['episode_id']==episode and ('warmup',10) in observed,'root requires returned warm10')
        w=observed['warmup',10];prior.validate_snapshot(d['snapshot'],d['state'])
        need(indices['warmup',10]<root['index'] and d['snapshot']==w['snapshot'] and d['state']==w['state'], 'warm10 original root exact')
        links.append('warm10_original_root')
    if guard:
        d=guard['detail'];need(root is not None and ('plan',0) in observed and d['episode_id']==episode,'plan root prerequisites')
        r=d['root_observation'];warm=observed['warmup',10];plan=observed['plan',0]
        need(root['index']<indices['plan',0]<guard['index'] and r==plan and
             warm['snapshot']==plan['snapshot'] and warm['state']==plan['state'] and
             d['original_root_sha256']==root['detail']['root_sha256'] and d['original_goal']==root['detail']['original_goal'],
             'goal bind unchanged full body snapshot and root identity')
        links.append('root_to_plan0_snapshot')
    if cp5:
        d=cp5['detail'];need(guard is not None and ('plan',5) in observed and d['episode_id']==episode and
             guard['index']<indices['plan',5]<cp5['index'] and d['observation']==observed['plan',5], 'cp5 exact receipt')
        links.append('cp5_guard')
    for e in get('e3_encode_intent'):
        d=e['detail'];step=d['boundary'];need(step in (10,15) and ('plan',step) in observed and
             d['observation']==observed['plan',step] and indices['plan',step]<e['index'], 'encoding uses exact same snapshot')
        links.append('encode_'+str(step))
    dispatch=get('authorized_dispatch');returns=get('control_return')
    need(len(returns)<=len(dispatch)<=len(returns)+1, 'at most one unreturned dispatch')
    stopped=False
    for i,d in enumerate(dispatch):
        x=d['detail'];need(x['episode_id']==episode and type(x['step']) is int and x['step']==i+1 and not stopped,
                           'dispatch contiguous and no dispatch after stop')
        if i>=len(returns):break
        r=returns[i];y=r['detail']
        need(d['index']<r['index'] and y['episode_id']==episode and y['step']==x['step'] and
             y['raw_bits']==x['raw_bits'] and type(y['terminated']) is bool and type(y['truncated']) is bool,
             'returned dispatch exact bytes/order')
        stopped=y['terminated'] or y['truncated']
    native_fail=one('episode_failure');ci=one('cleanup_intent');cr=one('cleanup_return');cf=one('cleanup_failure')
    need(not (cr and cf),'exclusive cleanup result')
    outcome=cr or cf
    if outcome:
        need(ci is not None and ci['index']<outcome['index'] and ci['detail']['episode_id']==episode and
             outcome['detail']['episode_id']==episode, 'cleanup order and episode')
    if native_fail:
        need(type(native_fail['detail'].get('error')) is dict,'native exception payload')
        if ci:need(native_fail['index']<ci['index'],'episode failure before cleanup')
    launchfail=[e for e in phy if e['resource']=='launcher_failure']
    need(len(launchfail)<=1,'unique launcher failure')
    if launchfail and outcome:need(outcome['index']<launchfail[0]['index'],'cleanup before launcher failure')
    terminal_returns=[e for e in phy if e['resource']=='terminal_row' and e['kind']=='return']
    need(len(terminal_returns)<=1,'unique terminal return')
    row=None
    if terminal_returns:
        row=terminal.read(events);need(row['episode_id']==episode,'terminal row episode')
        if native_fail:
            need(row.get('status')=='failed' and row.get('error')==native_fail['detail']['error'] and
                 native_fail['index']<terminal_returns[0]['index'], 'terminal preserves episode failure')
    if complete:
        need(ci is not None and cr is not None and not native_fail and not launchfail and not cf and
             row is not None and len(dispatch)==len(returns), 'complete native semantics prerequisites')
        if row.get('status')=='completed20':
            need(root is not None and guard is not None and cp5 is not None and len(returns)==20 and not stopped,
                 'completed20 native prerequisites')
    return dict(snapshot_occurrences=len(items),snapshot_locations=items,observations=len(observed),links=links,
        dispatched=len(dispatch),returned=len(returns),unreturned_dispatch=None if len(dispatch)==len(returns) else dispatch[-1]['detail'],
        native_stop_recorded=stopped,native_failure=None if native_fail is None else native_fail['detail'],
        cleanup='returned' if cr else 'failed' if cf else 'pending_or_not_started',
        launcher_failure=None if not launchfail else launchfail[0]['detail'],
        terminal_status=None if row is None else row.get('status'),
        full_failure_semantics=False,unrecorded_effects='unknown; no inferred physics or failed-step outcome')


def build(output,report_sha,completion_sha,episode):
    p=Path(output);transport=pub.consume(p,report_sha,completion_sha)
    report=pub.base.read_bytes(p/'report.json',pub.FINAL_CAP);need(pub.base.fsha(report)==report_sha,'stable report')
    journal=pub.base.read_bytes(p/'events.jsonl',pub.base.CAP);need(pub.base.fsha(journal)==pub.base.decode(report)['journal_sha256'],'stable journal')
    events=[pub.base.decode(line) for line in journal.splitlines()]
    data=audit(events,episode,transport['complete'])
    need(pub.base.fsha(pub.base.read_bytes(p/'events.jsonl',pub.base.CAP))==pub.base.fsha(journal),'journal changed during audit')
    need(pub.base.fsha(pub.base.read_bytes(p/'report.json',pub.FINAL_CAP))==report_sha and
         pub.base.fsha(pub.base.read_bytes(p/'completion.json',pub.FINAL_CAP))==completion_sha,'publication changed')
    return dict(schema=NAME+'.audit',output=str(p),report_sha256=report_sha,completion_sha256=completion_sha,
        episode=episode,transport_complete=transport['complete'],native=data,authority_sources_reverified=False,
        full_saved_strict=False,scientific_admission=False,physics_replayed=False,neural_executed=False,
        missing=['authority and source guard full independent revalidation','unrecorded goal-bind renderer/RNG before/after',
                 'exhaustive failure-path saved reconstruction','independent physical and neural parity'])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=('audit','verify'))
    for x in ('output','report-sha256','completion-sha256','episode','saved'):ap.add_argument('--'+x,required=True)
    ap.add_argument('--saved-sha256');a=ap.parse_args()
    value=build(a.output,a.report_sha256,a.completion_sha256,a.episode);data=pub.base.raw(value)
    path=Path(a.saved);need(path.is_absolute() and path.name.startswith(NAME+'_'),'dedicated absolute artifact')
    if a.command=='audit':
        need(a.saved_sha256 is None,'no saved sha on audit')
        with path.open('xb') as f:f.write(data);f.flush();__import__('os').fsync(f.fileno())
    else:
        need(a.saved_sha256 and pub.base.fsha(pub.base.read_bytes(path,16*1024**2))==a.saved_sha256 and
             pub.base.read_bytes(path,16*1024**2)==data,'whole saved reconstruction')
    print(pub.base.raw(dict(path=str(path),sha256=pub.base.fsha(data),verified=a.command=='verify',full_saved_strict=False)).decode())


if __name__=='__main__':main()
