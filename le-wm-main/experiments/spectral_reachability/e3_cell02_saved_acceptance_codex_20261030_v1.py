"""Cell02 saved-only terminal acceptance; no execution or scientific admission."""
import argparse
import hashlib
import os
from pathlib import Path

from experiments.spectral_reachability import e3_cell02_exclusive_candidate_codex_20261030_v1 as candidate
from experiments.spectral_reachability import agent_e3_chunked_identity_host_20261029_v1 as host
from experiments.spectral_reachability import agent_e3_native_saved_semantics_20261029_v1 as native
from experiments.spectral_reachability import agent_e3_feedback_saved_join_20261029_v1 as feedback

NAME = 'e3_cell02_saved_acceptance_codex_20261030_v1'
DIRECTORY = candidate.ROOT/'deployment_manifests_e3_cell02'
ACK = 'CELL02_TERMINAL_CONFIRMED_NO_RETRY'
base = host.launch.pub.base
need = base.need
MAX_RECEIPT = 16*1024**2


def digest_ok(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def read_pinned(path, digest, cap=MAX_RECEIPT):
    need(digest_ok(digest), 'externally pinned SHA required')
    data = base.read_bytes(Path(path), cap)
    need(base.fsha(data) == digest, 'pinned bytes changed')
    return data


def own_path(path, prefix):
    p = Path(path)
    need(p.is_absolute() and p.parent == DIRECTORY and p.parent == p.parent.resolve()
         and p.name.startswith(prefix) and p.suffix == '.json', 'dedicated cell02 artifact path')
    return p


def code_pins(pins):
    files = {'consumer': Path(__file__).resolve(), 'native': Path(native.__file__).resolve(),
             'feedback': Path(feedback.__file__).resolve()}
    need(set(pins) == set(files), 'exact checker pins')
    for name, path in files.items():
        need(candidate.file_sha(path) == pins[name] and digest_ok(pins[name]), 'reviewed checker source '+name)
    return pins.copy()


def provenance(manifest_path, manifest_sha):
    path = own_path(manifest_path, 'e3_cell02_manifest_')
    raw = read_pinned(path, manifest_sha)
    manifest = base.decode(raw)
    need(raw in (candidate.canonical(manifest), candidate.canonical(manifest)+b'\n'), 'canonical manifest')
    need(manifest.get('schema') == candidate.SCHEMA and manifest.get('cell') == 2
         and manifest.get('authority') == str(candidate.AUTHORITY)
         and manifest.get('authority_sha256') == candidate.AUTHORITY_SHA
         and manifest.get('output') == str(candidate.OUTPUT)
         and manifest.get('claim') == str(candidate.CLAIM)
         and manifest.get('production_authorization') == 'external_decision_required'
         and manifest.get('scientific_admission') is False
         and manifest.get('child_code_sha256') == candidate.sha(candidate.CHILD.encode())
         and manifest.get('audit_code_sha256') == candidate.sha(candidate.AUDIT.encode())
         and manifest.get('precheck_code_sha256') == candidate.sha(candidate.PRECHECK.encode())
         and manifest.get('parent_sources', {}).get(str(Path(candidate.__file__).resolve()))
             == candidate.file_sha(Path(candidate.__file__).resolve()), 'candidate manifest identity')
    claim_raw = base.read_bytes(candidate.CLAIM, 65536)
    claim = base.decode(claim_raw)
    need(claim_raw == candidate.canonical(claim) and set(claim) ==
         {'schema', 'manifest_sha256', 'authority_sha256', 'output', 'pid', 'meaning'}
         and claim['schema'] == candidate.SCHEMA+'.claim'
         and claim['manifest_sha256'] == manifest_sha
         and claim['authority_sha256'] == candidate.AUTHORITY_SHA
         and claim['output'] == str(candidate.OUTPUT)
         and type(claim['pid']) is int and claim['pid'] > 0
         and claim['meaning'] == 'attempt claimed, outcome unknown', 'exclusive claim binding')
    return base.fsha(claim_raw)


def build(report_sha, completion_sha, manifest_path, manifest_sha, pins, confirmation):
    need(confirmation == ACK, 'explicit cell02 terminal confirmation required')
    need(digest_ok(report_sha) and digest_ok(completion_sha), 'external publication SHAs required')
    code_pins(pins)
    claim_sha = provenance(manifest_path, manifest_sha)
    output = str(candidate.OUTPUT)
    inspected = host.inspect(output, report_sha, completion_sha, str(candidate.AUTHORITY), candidate.AUTHORITY_SHA)
    saved = inspected['base']
    need(saved['cell'] == 2 and saved['status'] != 'partial'
         and inspected['binding_chunks_complete'] is True and inspected['identity'] is not None,
         'complete cell02 source and identity inspection')
    native_audit = native.build(output, report_sha, completion_sha, 'cell02')
    need(native_audit['transport_complete'] is True
         and native_audit['native']['terminal_status'] == saved['status'], 'consistent terminal status')
    status = saved['status']
    need(status in ('completed20', 'native_stopped', 'native_stopped_warmup',
                    'native_stopped_prefix', 'native_initial_success_control5'), 'known terminal status')
    complete_join = None
    if status == 'completed20':
        binding = host.authority(str(candidate.AUTHORITY), candidate.AUTHORITY_SHA)
        need(binding['cell'] == 2, 'fixed authority cell')
        old = host.launch.episode.old
        actual = old.authority.load(*binding['args'])
        setup = old.setup_authority.resolve(binding['resolver']['path'], binding['resolver']['sha256'],
                                            2, actual['body'])
        complete_join = feedback.consume(output, report_sha, completion_sha, actual['body'],
                                         setup['prefix5'], actual['first5'], 'cell02')
        need(complete_join['completed20_feedback_io_joined'] is True, 'complete feedback join')
        need([x['boundary'] for x in inspected['identity']['encodes']] == [10, 15],
             'both actual encoder identities')
    code_pins(pins)
    need(provenance(manifest_path, manifest_sha) == claim_sha, 'claim changed during acceptance')
    rb = read_pinned(candidate.OUTPUT/'report.json', report_sha, host.launch.pub.FINAL_CAP)
    read_pinned(candidate.OUTPUT/'completion.json', completion_sha, host.launch.pub.FINAL_CAP)
    read_pinned(candidate.OUTPUT/'events.jsonl', base.decode(rb)['journal_sha256'], base.CAP)
    return {'schema': NAME+'.receipt', 'cell': 2, 'task': 'r0.g0.h3.dDxy',
            'authority_sha256': candidate.AUTHORITY_SHA, 'manifest_path': str(manifest_path),
            'manifest_sha256': manifest_sha, 'claim_sha256': claim_sha,
            'report_sha256': report_sha, 'completion_sha256': completion_sha,
            'checker_sha256': pins, 'terminal_status': status,
            'saved_inspection': inspected, 'native_audit': native_audit, 'full_feedback_join': complete_join,
            'complete_feedback_chain_checked': complete_join is not None,
            'full_saved_strict': False, 'scientific_admission': False,
            'physics_replayed': False, 'neural_recomputed': False,
            'limitations': ['caller confirms terminal state; not process attestation',
                            'source-trusted snapshots are not same-byte import or process-memory attestation',
                            'no independent physical or neural parity or complete failure semantics',
                            'early stops retain only reached feedback, never padded']}


def save(path, value):
    p = own_path(path, 'e3_cell02_acceptance_')
    data = base.raw(value)
    need(len(data) <= MAX_RECEIPT, 'bounded receipt')
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        need(stream.write(data) == len(data), 'complete receipt write')
        stream.flush()
        os.fsync(stream.fileno())
    base.syncdir(p.parent)
    return base.fsha(data)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    for command in ('audit', 'verify'):
        a = sub.add_parser(command)
        for option in ('report-sha256', 'completion-sha256', 'manifest', 'manifest-sha256',
                       'consumer-sha256', 'native-sha256', 'feedback-sha256', 'confirmation', 'saved'):
            a.add_argument('--'+option, required=True)
        if command == 'verify':
            a.add_argument('--saved-sha256', required=True)
    a = p.parse_args()
    pins = {'consumer': a.consumer_sha256, 'native': a.native_sha256, 'feedback': a.feedback_sha256}
    result = build(a.report_sha256, a.completion_sha256, a.manifest, a.manifest_sha256, pins, a.confirmation)
    if a.command == 'audit':
        sha = save(a.saved, result)
    else:
        data = read_pinned(own_path(a.saved, 'e3_cell02_acceptance_'), a.saved_sha256)
        need(data == base.raw(result), 'exact receipt reconstruction')
        sha = a.saved_sha256
    print(base.raw({'path': a.saved, 'sha256': sha, 'verified': a.command == 'verify',
                    'full_saved_strict': False, 'scientific_admission': False}).decode())


if __name__ == '__main__':
    main()
