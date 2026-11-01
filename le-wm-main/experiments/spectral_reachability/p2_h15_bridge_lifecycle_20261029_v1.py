"""Rebuild real P2-to-native identity and exercise two-root RAM lifecycle only."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from experiments.spectral_reachability import p2_h15_source_lifecycle_20261029_v1 as source
from experiments.spectral_reachability import p2_h15_sra_host_candidate_20261029_v1 as single
from experiments.spectral_reachability import agent_p1_nonlinear_suffix_library_20261028_v2 as producer
from experiments.spectral_reachability import agent_p3_runtime_contract_20261028_v1 as core
from experiments.spectral_reachability import agent_p3_c3_k17_physical_transport_20261029_v1 as transport
from experiments.spectral_reachability import agent_p3_fullcommit_live_20261028_v1 as accounting

NAME = 'p2_h15_bridge_lifecycle_20261029_v1'
P1 = source.prep.ROOT/'results/agent_p1_joint_vreach_host_cuda_20261029_v1_prepare_MAIN01.json'
P1_SHA = '97a312976831ca4104cae2f69283679e9b4a432a7339011bb122eeb8f46f82dd'
LIMIT = 256*1024**2


def need(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    return source.sha(path)


def build_bridge():
    manifest = source.build()
    h15 = source.prep.read(source.H15, source.H15_SHA)
    host = source.prep.read(source.HOST, source.HOST_SHA)
    p1 = source.prep.read(P1, P1_SHA)
    need(p1['expected_model_hash'] == manifest['expected_state_dict_sha256'], 'state-dict namespace')
    requests = [r for r in p1['requests'] if r['condition'] == 'off']
    need(len(requests) == 2 and [r['root_slot'] for r in requests] == [0, 1], 'two off roots')
    roots=[]
    for case, mapped, req, slot in zip(h15['cases'], manifest['roots'], requests, host['slots'], strict=True):
        task = core.validate_task(req['task'])
        rebuilt = producer.task_from(slot, p1['expected_model_hash'])
        need(task == rebuilt, 'native ABI producer reconstruction')
        b = task['binding']
        need(case['task_id'] == mapped['task_id'] == b['task_id'] and
             case['task_sha256'] == mapped['task_sha256'] == b['task_sha256'], 'original task join')
        need(task['root_context'] == case['root_context'] and task['goal'] == case['goal'] and
             task['projection'] == case['q'] and task['action_mean'] == case['action_normalizer']['mean'] and
             task['action_std'] == case['action_normalizer']['std'], 'causal and normalization join')
        need(b['model_sha256'] == manifest['expected_state_dict_sha256'] and
             case['model_sha256'] == manifest['checkpoint_file_sha256'], 'distinct model namespaces')
        need(case['native_root'] == mapped['physical_native_root'], 'physical root join')
        roots.append(dict(root_slot=case['root_slot'], physical_root_sha256=case['native_root']['root_sha256'],
                          checkpoint_file_sha256=manifest['checkpoint_file_sha256'], task=task))
    return dict(schema=NAME+'.bridge', p1_prepare_sha256=P1_SHA, h15_prepare_sha256=source.H15_SHA,
                roots=roots, host_authorized=False, physical_release=False,
                root_semantics='physical source root and derived P1 native ABI identifier are distinct',
                model_semantics='checkpoint file SHA versus historical expected state-dict SHA; runtime unverified')


def run_host(*args, **kwargs):
    raise ValueError('HOST dispatch not authorized')


def save_bridge(path):
    value = build_bridge()
    single._put(Path(path), value)
    return sha(path)


def verify_bridge(path, expected):
    raw = _read(path, 8*1024**2)
    need(hashlib.sha256(raw).hexdigest() == expected, 'external bridge SHA')
    value = json.loads(raw)
    need(raw == core.canonical(value)+b'\n' and value == build_bridge(), 'bridge rebuild')
    return value


def _read(path, limit=LIMIT):
    path=Path(path)
    need(path.is_file() and not path.is_symlink() and path.stat().st_size <= limit, 'bounded file')
    return path.read_bytes()


def _failed_slot(path, task, digest):
    data=_read(path/'report.json'); need(hashlib.sha256(data).hexdigest()==digest, 'failed report SHA')
    report=json.loads(data)
    need(data==core.canonical(report)+b'\n' and report['schema']==single.NAME+'.report' and
         report['status']=='failed_partial' and report['backend']=='RAM_FIXTURE' and
         report['task_sha256']==core.digest(task) and report['error'] is not None and
         report['ready_to_execute'] is False and report['physical_release'] is False, 'failed slot identity')
    data=_read(path/'events.jsonl'); need(hashlib.sha256(data).hexdigest()==report['journal_sha256'], 'failed journal SHA')
    events=[e['payload'] for e in transport.read_journal(path/'events.jsonl')]
    accounting.audit_resources(report['resources'],[e for e in events if e['kind'].startswith('resource_')])
    need(report['resources']['caps']==single.caps(), 'failure ledger caps')
    return report


def run_fixture(directory, factory):
    bridge=build_bridge(); path=Path(directory); need(not path.is_symlink(), 'no symlink')
    path.mkdir()
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)
    j=transport.Journal(path/'lifecycle.jsonl',1024*1024)
    transitions=[]; slots=[]
    try:
        j.emit('bridge',dict(sha256=core.digest(bridge),fixture_only=True))
        for i, root in enumerate(bridge['roots']):
            transitions.append(('start',i)); source.lifecycle(transitions)
            j.emit('start',dict(root=i,task_sha256=core.digest(root['task'])))
            try:
                result=single.run_fixture(path/('root'+str(i)),root['task'],factory)
                digest=result['report_sha256']
                if result['status']=='fixture_complete':
                    single.consume(path/('root'+str(i)),digest,root['task'],permit_fixture=True)
                    action='seal'
                else:
                    _failed_slot(path/('root'+str(i)),root['task'],digest)
                    action='fail'
                slot=dict(root=i,status=result['status'],report_sha256=digest)
            except Exception as exc:
                action='fail'
                slot=dict(root=i,status='orchestration_exception',report_sha256=None,
                          error=dict(type=type(exc).__name__,message=str(exc)[:2048]))
            transitions.append((action,i)); source.lifecycle(transitions)
            slots.append(slot)
            j.emit(action,slots[-1])
            if action=='fail': break
    finally:
        j.close()
    report=dict(schema=NAME+'.report',bridge_sha256=core.digest(bridge),slots=slots,
                root_status=source.lifecycle(transitions),fixed_denominator=2,fixture_only=True,
                host_authorized=False,physical_release=False,
                lifecycle_sha256=sha(path/'lifecycle.jsonl'))
    single._put(path/'report.json',report)
    return dict(report_sha256=sha(path/'report.json'),root_status=report['root_status'])


def consume(directory, expected):
    bridge=build_bridge(); path=Path(directory); need(path.is_dir() and not path.is_symlink(), 'directory')
    raw=_read(path/'report.json'); need(hashlib.sha256(raw).hexdigest()==expected, 'external report SHA')
    report=json.loads(raw); need(raw==core.canonical(report)+b'\n', 'canonical report')
    need(report['schema']==NAME+'.report' and report['bridge_sha256']==core.digest(bridge) and
         report['fixed_denominator']==2 and report['fixture_only'] is True and
         report['host_authorized'] is False and report['physical_release'] is False, 'report flags')
    raw=_read(path/'lifecycle.jsonl',1024*1024)
    need(hashlib.sha256(raw).hexdigest()==report['lifecycle_sha256'], 'lifecycle SHA')
    events=transport.read_journal(path/'lifecycle.jsonl')
    need(events and events[0]['kind']=='bridge' and events[0]['payload']==dict(sha256=core.digest(bridge),fixture_only=True), 'bridge event')
    transitions=[]; expected_events=[]
    for i, slot in enumerate(report['slots']):
        need(i<2 and slot['root']==i, 'slot order')
        task=bridge['roots'][i]['task']; p=path/('root'+str(i))
        expected_events.append(('start',dict(root=i,task_sha256=core.digest(task))))
        transitions.append(('start',i))
        if slot['status']=='fixture_complete':
            single.consume(p,slot['report_sha256'],task,permit_fixture=True); action='seal'
        elif slot['status']=='failed_partial':
            need(slot['status']=='failed_partial' and i==len(report['slots'])-1, 'failure stop')
            _failed_slot(p,task,slot['report_sha256']); action='fail'
        else:
            need(slot['status']=='orchestration_exception' and i==len(report['slots'])-1 and
                 slot['report_sha256'] is None and type(slot.get('error')) is dict and
                 set(slot['error'])=={'type','message'} and all(type(v) is str for v in slot['error'].values()),
                 'unaudited orchestration failure')
            action='fail'
        transitions.append((action,i)); expected_events.append((action,slot))
    need(report['slots'] and (len(report['slots'])==2 or report['slots'][-1]['status'] in ('failed_partial','orchestration_exception')), 'no truncated completion')
    need([(e['kind'],e['payload']) for e in events[1:]]==expected_events and
         source.lifecycle(transitions)==report['root_status'], 'lifecycle replay')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('prepare','verify'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sha256')
    a=p.parse_args()
    need(a.output.name.startswith(NAME+'_'), 'exclusive output namespace')
    if a.command=='prepare':
        need(a.sha256 is None, 'unexpected SHA'); digest=save_bridge(a.output)
    else:
        need(a.sha256 is not None, 'external SHA required'); verify_bridge(a.output,a.sha256); digest=a.sha256
    print(json.dumps(dict(path=str(a.output),sha256=digest,host_authorized=False)))


if __name__=='__main__': main()
