"""Durable, independently replayable P2 HOST startup refusal; no dispatch.

This is not a CUDA runner: the only accepted terminal is pre-dispatch refusal.
It cannot certify a model, measured budget, a physical result, or host execution.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

from experiments.spectral_reachability import p2_h15_source_lifecycle_20261029_v1 as source
from experiments.spectral_reachability import p2_h15_sra_host_candidate_20261029_v1 as candidate
from experiments.spectral_reachability import agent_p3_c3_k17_physical_transport_20261029_v1 as transport

NAME = 'p2_h15_two_root_saved_host_20261029_v1'
MAX_BYTES = 1024 * 1024
REASONS = ['host_dispatch_disabled', 'runtime_model_unverified', 'runtime_precision_unverified',
           'runtime_budget_unverified', 'physical_unverified']


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _write(path, data):
    with path.open('xb') as f:
        _need(f.write(data) == len(data), 'short report write')
        f.flush()
        os.fsync(f.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def _read(path, limit):
    _need(path.is_file() and not path.is_symlink() and path.stat().st_size <= limit, 'bounded regular file')
    return path.read_bytes()


def _summary(manifest_sha):
    return dict(schema=NAME+'.report', manifest_sha256=manifest_sha, status='startup_refused',
                root_status=source.lifecycle([]), fixed_denominator=2,
                attempted_predict_samples=0, completed_predict_samples=0,
                planned_caps_per_root=candidate.caps(), measured_runtime_budget=False,
                host_certified=False, runtime_model_verified=False, runtime_precision_verified=False,
                physical_release=False, scientific_admission=False, reasons=REASONS)


def run_host(directory, manifest_path, manifest_sha256):
    manifest = source.consume(manifest_path, manifest_sha256)
    _need(manifest['execution_authorized'] is False and manifest['fixed_denominator'] == 2, 'source fail closed')
    path = Path(directory)
    _need(not path.is_symlink() and path.parent.is_dir(), 'safe parent')
    path.mkdir()  # exclusive, including on retries
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)
    journal = transport.Journal(path/'events.jsonl', MAX_BYTES)
    try:
        journal.emit('startup', dict(manifest_sha256=manifest_sha256, roots=[r['task_id'] for r in manifest['roots']]))
        journal.emit('refusal', dict(reasons=REASONS, root_status=source.lifecycle([])))
    finally:
        journal.close()
    report = _summary(manifest_sha256)
    report['journal_sha256'] = hashlib.sha256(_read(path/'events.jsonl', MAX_BYTES)).hexdigest()
    data = transport.canonical(report) + b'\n'
    _need(len(data) <= MAX_BYTES, 'report bound')
    _write(path/'report.json', data)
    return dict(report_sha256=hashlib.sha256(data).hexdigest(), status=report['status'])


def consume(directory, report_sha256, manifest_path, manifest_sha256):
    manifest = source.consume(manifest_path, manifest_sha256)
    path = Path(directory)
    _need(path.is_dir() and not path.is_symlink(), 'directory')
    raw = _read(path/'report.json', MAX_BYTES)
    _need(hashlib.sha256(raw).hexdigest() == report_sha256, 'external report SHA')
    report = json.loads(raw)
    _need(raw == transport.canonical(report)+b'\n', 'canonical report')
    journal = _read(path/'events.jsonl', MAX_BYTES)
    _need(hashlib.sha256(journal).hexdigest() == report['journal_sha256'], 'journal SHA')
    events = transport.read_journal(path/'events.jsonl')
    expected = [('startup', dict(manifest_sha256=manifest_sha256, roots=[r['task_id'] for r in manifest['roots']])),
                ('refusal', dict(reasons=REASONS, root_status=source.lifecycle([])))]
    _need([(e['kind'], e['payload']) for e in events] == expected, 'startup journal sequence')
    expected_report = _summary(manifest_sha256)
    expected_report['journal_sha256'] = hashlib.sha256(journal).hexdigest()
    _need(report == expected_report, 'report rebuild')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('run', 'consume'))
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--manifest-sha256', required=True)
    p.add_argument('--report-sha256')
    a = p.parse_args()
    if a.command == 'run':
        _need(a.report_sha256 is None, 'unexpected report SHA')
        result = run_host(a.directory, a.manifest, a.manifest_sha256)
    else:
        _need(a.report_sha256 is not None, 'external report SHA required')
        result = consume(a.directory, a.report_sha256, a.manifest, a.manifest_sha256)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__': main()
