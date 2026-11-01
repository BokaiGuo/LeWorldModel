"""Official GitHub blob API follow-up when raw.githubusercontent is unavailable."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path

from experiments.spectral_reachability import p1_checkpoint_historical_contract_lookup_20261027_v1 as lookup

TAG = 'p1_checkpoint_historical_contract_github_api_20261027_v1'
MAX_BLOB = 128*1024
FILES = ('README.md','train.py','utils.py','config/train/lewm.yaml','config/train/data/pusht.yaml',
         'config/train/model/lewm.yaml','config/eval/pusht.yaml','eval.py')


def decode_blob(data, expected_sha):
    if not isinstance(data,dict) or data.get('encoding') != 'base64' or data.get('sha') != expected_sha:
        raise ValueError('expected GitHub base64 blob identity')
    if not isinstance(data.get('size'),int) or not 0 <= data['size'] <= MAX_BLOB:
        raise ValueError('bounded text blob required')
    content = base64.b64decode(''.join(data['content'].split()),validate=True)
    digest = hashlib.sha1(f'blob {len(content)}\0'.encode()+content).hexdigest()
    if len(content) != data['size'] or digest != expected_sha:
        raise ValueError('Git blob content digest mismatch')
    content.decode('utf8')
    return content


def run(parent, output):
    from experiments.spectral_reachability import d2p_prerun_source_seal as seal
    if output.exists():
        raise ValueError('exclusive NEW output required')
    parent_report = parent/'report.json'
    report = json.loads(parent_report.read_text())
    if report.get('schema') != lookup.TAG or not report.get('source_and_local_inputs_unchanged'):
        raise ValueError('completed parent lookup required')
    records = {r['label']:r for r in report['requests']}
    bound = {str(parent_report):lookup.sha(parent_report),str(parent/'protocol.json'):report['protocol_sha256']}
    for record in records.values():
        if record.get('body_path'):
            path = Path(record['body_path'])
            if lookup.sha(path) != record['body_sha256']:
                raise ValueError('parent response changed')
            bound[str(path)] = record['body_sha256']
    sources = [Path(__file__),Path(lookup.__file__),Path(seal.__file__),
               lookup.ROOT/f'tests/test_{TAG}.py',lookup.ROOT/f'innerdocs/{TAG}.md']
    bound.update({str(p):lookup.sha(p) for p in sources})
    output.mkdir(parents=True)
    (output/'responses').mkdir()
    (output/'decoded').mkdir()
    lookup.write_json(output/'protocol.json',dict(schema=TAG,P1=False,parent_report_sha256=lookup.sha(parent_report),
        input_bindings=bound,max_requests=32,max_response_bytes=lookup.MAX_RESPONSE,
        max_decoded_blob_bytes=MAX_BLOB,total_combined_body_limit_bytes=lookup.MAX_TOTAL,
        strategy='Exact previously recorded Git tree blob SHA; no raw-host retry, source execution, or larger assets.',
        selection=dict(project_files=FILES,dependency_files='pyproject plus reader/transform implementation only'),
        checkpoint_to_training_binding='unknown until explicit witness; dependency main is not training provenance'))
    source_seal = seal.create_seal(files=sources+[output/'protocol.json'],
        args_json=json.dumps(dict(protocol_sha256=lookup.sha(output/'protocol.json'))),output_dir=output/'source_seal')
    collector = lookup.Collector(output)
    # Carry the first-stage consumed body budget forward rather than silently resetting it.
    initial_bytes = report['counts']['body_bytes']
    collector.used = initial_bytes
    trees = []
    for i,revision in enumerate(report['candidate_github_revisions']):
        tree = lookup.parsed(records[f'github_tree_{i}'])
        if isinstance(tree,dict) and not tree.get('truncated',True):
            trees.append((lookup.REPO,revision,tree,FILES))
    dependency_records = []
    for i,repo in enumerate(lookup.DEPENDENCIES):
        head = lookup.parsed(records[f'dependency_head_{i}'])
        if not isinstance(head,dict) or not head.get('sha'):
            continue
        record = collector.fetch(f'dependency_tree_{i}',f'https://api.github.com/repos/{repo}/git/trees/{head["sha"]}?recursive=1')
        dependency_records.append(record)
        tree = lookup.parsed(record)
        if isinstance(tree,dict) and not tree.get('truncated',True):
            choices = ('pyproject.toml','stable_worldmodel/data/formats/hdf5.py','stable_worldmodel/data/formats/lance.py') if i==0 else (
                'pyproject.toml','stable_pretraining/data/transforms.py')
            trees.append((repo,head['sha'],tree,choices))
    targets = []
    for repo,revision,tree,choices in trees:
        for node in tree['tree']:
            if node['type']=='blob' and node['path'] in choices and 0<=node.get('size',MAX_BLOB+1)<=MAX_BLOB:
                targets.append(dict(repository=repo,revision=revision,path=node['path'],git_blob_sha1=node['sha']))
    # Deduplicate byte-identical blobs while retaining all revision/path edges.
    unique = {(row['repository'],row['git_blob_sha1']) for row in targets}
    if len(unique)+collector.requests > 32:
        raise ValueError('follow-up request budget exceeds frozen cap')
    responses = collector.batch([(f'blob_{i:02d}',f'https://api.github.com/repos/{repo}/git/blobs/{sha}')
                                 for i,(repo,sha) in enumerate(sorted(unique))])
    content_by_blob = {}
    for record in responses:
        data = lookup.parsed(record)
        if data is None:
            continue
        expected_sha = record['url'].rsplit('/',1)[-1]
        content = decode_blob(data,expected_sha)
        path = output/'decoded'/(expected_sha+'.txt')
        path.write_bytes(content)
        content_by_blob[expected_sha] = dict(decoded_path=str(path),sha256=hashlib.sha256(content).hexdigest(),
                                           response_sha256=record['body_sha256'],source_url=record['url'])
    for row in targets:
        row.update(content_by_blob.get(row['git_blob_sha1'],dict(status='unavailable')))
    if any(lookup.sha(p)!=digest for p,digest in bound.items()):
        raise ValueError('bound parent/source bytes changed')
    seal.verify_seal(output_dir=output/'source_seal',expected_manifest_sha256=source_seal['manifest_sha256'])
    result = dict(schema=TAG,status='official_blob_sources_recorded',P1=False,source_seal=source_seal,
        protocol_sha256=lookup.sha(output/'protocol.json'),parent_report_sha256=lookup.sha(parent_report),
        all_bound_inputs_unchanged=True,source_edges=targets,requests=sorted(collector.records,key=lambda r:r['id']),
        counts=dict(requests=collector.requests,body_bytes=collector.used-initial_bytes,
                    combined_body_bytes=collector.used,decoded_unique_blobs=len(content_by_blob),
                    neural=0,physics=0,weight_or_dataset_download_bytes=0,author_messages=0),
        historical_training_binding='unknown; source edges establish public code bytes, not fixed checkpoint run provenance')
    lookup.write_json(output/'report.json',result)
    print(json.dumps(dict(report=str(output/'report.json'),sha256=lookup.sha(output/'report.json'),counts=result['counts'])))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--out-dir',type=Path,required=True)
    args=parser.parse_args()
    run(args.parent.resolve(),args.out_dir.resolve())
