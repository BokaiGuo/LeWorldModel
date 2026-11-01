"""Saved-only, whole-object reconstruction and semantic receipt extraction."""
import argparse
import json
import os
from pathlib import Path

from experiments.spectral_reachability import agent_p1_intersection_goal_adapter_20261029_v2 as upstream
from experiments.spectral_reachability import agent_p1_intersection_goal_receipt_adapter_20261029_v1 as normalizer
from experiments.spectral_reachability import agent_p1_prospective_goal_semantics_20261029_v1 as semantic

NAME = 'agent_p1_intersection_semantic_extract_20261029_v1'


def load(path, digest):
    return json.loads(upstream.read(Path(path), digest))


def build(normalized_path, normalized_sha, spec_path, spec_sha, prepared_path, prepared_sha):
    n = load(normalized_path, normalized_sha)
    spec = load(spec_path, spec_sha)
    prepared = load(prepared_path, prepared_sha)
    b = n['bindings']
    upstream.need(set(b) == {'prepare', 'physical_receipt', 'physical_report', 'source'}, 'binding fields')
    for v in b.values():
        upstream.need(set(v) == {'path', 'sha256'}, 'binding descriptor')
        upstream.read(Path(v['path']), v['sha256'])
    upstream.need(Path(b['source']['path']) == upstream.SOURCE, 'physical source identity')
    upstream.read(upstream.RECEIPT, upstream.RECEIPT_SHA)
    p = load(b['prepare']['path'], b['prepare']['sha256'])
    r = load(b['physical_receipt']['path'], b['physical_receipt']['sha256'])
    projected = upstream.project(p, Path(b['prepare']['path']), b['prepare']['sha256'], upstream.SOURCE, b['source']['sha256'])
    upstream.need(spec == projected, 'spec projection mismatch')
    upstream.need(prepared == semantic.prepare(spec), 'semantic prepare mismatch')
    rebuilt = normalizer.normalize(spec, p, r, Path(b['physical_receipt']['path']), b['physical_receipt']['sha256'], Path(b['prepare']['path']), b['prepare']['sha256'], b['source']['sha256'], Path(b['physical_report']['path']), b['physical_report']['sha256'])
    upstream.need(n == rebuilt, 'whole normalized mismatch')
    receipt = n['receipt']
    result = semantic.analyze(prepared, receipt)
    analysis = dict(schema=NAME+'.analysis', normalized=dict(path=str(normalized_path),sha256=normalized_sha), spec=dict(path=str(spec_path),sha256=spec_sha), semantic_prepare=dict(path=str(prepared_path),sha256=prepared_sha), extracted_receipt_sha256=semantic.sha(semantic.encoded(receipt)), semantic_analysis=result, checkpoint_provenance=n['details'], denominator=n['denominator'], known_roots=2, fresh_roots=0, rgb_velocity_collision_tested=False, external_trusted_prospective_verified=False, upstream_full_strict_reexecuted=False, P1_admission=False, P2_admission=False, interpretation='Inherited strict receipt; cp0 is derived, not a native returned step. No external trusted timing, fresh roots, RGB collision test, or scientific admission is established.')
    return receipt, analysis


def output(command, path, data):
    p = Path(path)
    upstream.need(p.is_absolute() and p.name.startswith(NAME+'_') and not p.is_symlink(), 'own absolute output')
    if command.startswith('verify'):
        upstream.need(p.is_file() and p.read_bytes() == data, 'whole output mismatch')
    else:
        with p.open('xb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
    return semantic.sha(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['extract', 'verify-extract', 'analyze', 'verify-analysis'])
    for key in ('normalized','normalized-sha','spec','spec-sha','prepared','prepared-sha','output'):
        ap.add_argument('--'+key, required=True)
    a = ap.parse_args()
    receipt, analysis = build(a.normalized, a.normalized_sha, a.spec, a.spec_sha, a.prepared, a.prepared_sha)
    data = semantic.encoded(receipt if a.command in ('extract', 'verify-extract') else analysis)
    print(a.command, output(a.command, a.output, data))


if __name__ == '__main__':
    main()
