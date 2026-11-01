"""Preseal MAIN05 observed modules for future runs; not a complete import closure."""
import argparse
import base64
from pathlib import Path

from experiments.spectral_reachability import agent_e3_project_closure_20261029_v1 as project
from experiments.spectral_reachability import agent_e3_startup_closure_20261029_v1 as startup

host = project.host
launch = project.launch
census = launch.module_census
base = launch.pub.base
need = project.need
NAME = 'agent_e3_observed_closure_20261029_v1'
SOURCE_SHA = project.SOURCE_SHA
REPORT_SHA = '8ff1580f2c79482ecef10ec370367e0cc8f4b9b43e96ade0cbcea54913a5bb01'
COMPLETION_SHA = 'f12b2a20d46b7268ee8015e3848313d370bb6b095e9521ca3aef75033d990d28'
HISTORY = str(census.ROOT / 'results/agent_e3_chunked_identity_host_20261029_v1_MAIN05')


def observed(events):
    # Decode only to derive the candidate map; original saved() independently checks
    # all chunk ordering, digests, identities, and current file bytes.
    pins = {}
    for stage in ('before', 'after'):
        chunks = [e for e in events if e['origin'] == 'physical' and e['resource'] == 'module_census_chunk' and e['detail'].get('stage') == stage]
        need(chunks, 'missing historical census chunks')
        blob = b''.join(base64.b64decode(e['detail']['data'], validate=True) for e in chunks)
        need(len(blob) <= census.MAX, 'bounded candidate census')
        entries = base.decode(blob)
        need(type(entries) is list and entries, 'candidate list')
        for entry in entries:
            p, digest = entry['path'], entry['sha256']
            need(type(p) is str and type(digest) is str, 'candidate identity')
            need(p not in pins or pins[p] == digest, 'historical path hash conflict: '+p)
            pins[p] = digest
    for path, sha in pins.items():
        p = Path(path)
        need(p.is_absolute() and p == p.resolve() and census.ROOT in p.parents and p.is_file() and not p.is_symlink(), 'historical source path: '+path)
        need(startup.file_hash(path) == sha, 'historical loaded source drift: '+path)
    census.saved(events, pins)
    ends = {stage: [e for e in events if e['origin'] == 'physical' and e['resource'] == 'module_census_'+stage][0] for stage in ('before', 'after')}
    need(ends['before']['index'] < ends['after']['index'], 'census stage order')
    return pins, {stage: ends[stage]['detail']['sha256'] for stage in ends}


def historical():
    launch.pub.consume(HISTORY, REPORT_SHA, COMPLETION_SHA)
    journal = base.read_bytes(Path(HISTORY)/'events.jsonl', base.CAP)
    report = base.decode(base.read_bytes(Path(HISTORY)/'report.json', launch.pub.FINAL_CAP))
    need(launch.host.sha(str(Path(HISTORY)/'report.json')) == REPORT_SHA and
         base.fsha(journal) == report['journal_sha256'], 'stable historical journal')
    events = [base.decode(line) for line in journal.splitlines()]
    return observed(events)


def reconstruct(source, source_sha):
    need(source_sha == SOURCE_SHA, 'fixed installed authority')
    binding = host.authority(source, source_sha)
    prior = launch.host.read(source, source_sha)
    pins, seals = historical()
    # The preparer is a direct dependency of reconstruction, not a claim of
    # static transitive dependency discovery.
    p = str(Path(__file__).resolve())
    digest = startup.file_hash(p)
    need(p not in pins or pins[p] == digest, 'preparer collision')
    pins[p] = digest
    value = project.reconstruct(binding, prior, pins)
    merged = dict(binding['prepared']['bindings_sha256'])
    launch.host.source.merge(merged, value['additional_sources'])
    launch.episode.old.guard.validate_sources(merged, list(pins))
    need(launch.host.sha(source) == source_sha, 'stable source authority')
    # A second complete pass catches selected-source drift during reconstruction.
    for path, sha in pins.items():
        need(startup.file_hash(path) == sha, 'selected source drift: '+path)
    return value, pins, seals


def prepare(source, source_sha, output):
    value, pins, seals = reconstruct(source, source_sha)
    digest = startup.write_new(output, value)
    verify(source, source_sha, output, digest)
    return dict(path=output, sha256=digest, selected_paths=len(pins), census_sha256=seals,
                complete_import_closure=False, scientific_admission=False)


def verify(source, source_sha, output, output_sha):
    value, pins, seals = reconstruct(source, source_sha)
    need(launch.host.read(output, output_sha) == value, 'exact authority reconstruction')
    host.authority(output, output_sha)
    for p, sha in pins.items(): need(startup.file_hash(p) == sha, 'final selected source drift: '+p)
    need(launch.host.sha(source) == source_sha and launch.host.sha(output) == output_sha, 'stable authority inputs')
    return dict(verified=True, selected_paths=len(pins), census_sha256=seals,
                complete_import_closure=False, scientific_admission=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'verify'):
        sub = subs.add_parser(name)
        for arg in ('source', 'source-sha256', 'output'): sub.add_argument('--'+arg, required=True)
        if name == 'verify': sub.add_argument('--output-sha256', required=True)
    a = parser.parse_args()
    result = prepare(a.source, a.source_sha256, a.output) if a.command == 'prepare' else verify(a.source, a.source_sha256, a.output, a.output_sha256)
    print(launch.host.raw(result).decode())


if __name__ == '__main__': main()
