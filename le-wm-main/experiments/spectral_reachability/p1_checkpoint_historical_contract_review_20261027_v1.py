"""Offline consolidation of the bounded official-source lookup; unknown stays unknown."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from experiments.spectral_reachability import p1_checkpoint_historical_contract_lookup_20261027_v1 as lookup
from experiments.spectral_reachability import p1_checkpoint_historical_contract_github_api_20261027_v1 as api

TAG='p1_checkpoint_historical_contract_review_20261027_v1'
FIRST=lookup.ROOT/'results'/f'{lookup.TAG}_public_v1'
SECOND=lookup.ROOT/'results'/f'{api.TAG}_public_v1'


def git_blob(data):
    return hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()


def verify_body(record):
    if record.get('body_path') and lookup.sha(record['body_path']) != record['body_sha256']:
        raise ValueError('raw response changed')


def run(output):
    from experiments.spectral_reachability import d2p_prerun_source_seal as seal
    if output.exists():
        raise ValueError('exclusive NEW output required')
    bindings={}
    parents=[]
    for directory in (FIRST,SECOND):
        report=json.loads((directory/'report.json').read_text())
        if not (report.get('source_and_local_inputs_unchanged') or report.get('all_bound_inputs_unchanged')):
            raise ValueError('completed parent verification required')
        for path in (directory/'report.json',directory/'protocol.json'):
            bindings[str(path)]=lookup.sha(path)
        if bindings[str(directory/'protocol.json')]!=report['protocol_sha256']:
            raise ValueError('parent protocol digest differs')
        seal.verify_seal(output_dir=report['source_seal']['output_dir'],
                         expected_manifest_sha256=report['source_seal']['manifest_sha256'])
        for record in report['requests']:
            verify_body(record)
            if record.get('body_path'):
                bindings[record['body_path']]=record['body_sha256']
        parents.append(report)
    first,second=parents
    if second['parent_report_sha256']!=bindings[str(FIRST/'report.json')]:
        raise ValueError('parent-child binding mismatch')
    records={r['label']:r for r in first['requests']}
    public_history={key:lookup.parsed(records[key]) for key in
                    ('github_train_history','github_utils_history','github_data_config_history')}
    trees=[lookup.parsed(records[f'github_tree_{i}']) for i in range(len(first['candidate_github_revisions']))]
    local_matches=[]
    for revision,tree in zip(first['candidate_github_revisions'],trees):
        if not isinstance(tree,dict) or tree.get('truncated',True):
            continue
        for node in tree['tree']:
            if node['type']!='blob' or node['path'] not in api.FILES:
                continue
            path=lookup.ROOT/node['path']
            if path.is_file() and path.stat().st_size<=api.MAX_BLOB and git_blob(path.read_bytes())==node['sha']:
                bindings[str(path)]=lookup.sha(path)
                local_matches.append(dict(repository=lookup.REPO,revision=revision,path=node['path'],
                    official_git_blob_sha1=node['sha'],local_path=str(path),local_sha256=lookup.sha(path),
                    decoded_text=path.read_text(),authority='Entire local bytes match official pinned-tree Git blob; not checkpoint-training binding'))
    for edge in second['source_edges']:
        if edge.get('decoded_path'):
            content=Path(edge['decoded_path']).read_bytes()
            if git_blob(content)!=edge['git_blob_sha1'] or lookup.sha(edge['decoded_path'])!=edge['sha256']:
                raise ValueError('decoded official blob changed')
            bindings[edge['decoded_path']]=edge['sha256']
    facts=[]
    for group,history in public_history.items():
        if not isinstance(history,list):
            continue
        for row in history:
            facts.append(dict(history=group,commit=row['sha'],date=row['commit']['committer']['date'],
                              message=row['commit']['message'],authority='Official public history; no checkpoint-run identity'))
    missing_sources=[dict(repository=e['repository'],revision=e['revision'],path=e['path'],
                          git_blob_sha1=e['git_blob_sha1'],official_api_url=f'https://api.github.com/repos/{e["repository"]}/git/blobs/{e["git_blob_sha1"]}')
                     for e in second['source_edges'] if e['repository']==lookup.REPO and not e.get('decoded_path') and
                     e['path'] in ('utils.py','train.py')]
    sources=[Path(__file__),Path(lookup.__file__),Path(api.__file__),Path(seal.__file__),
             lookup.ROOT/f'tests/test_{TAG}.py',lookup.ROOT/f'innerdocs/{TAG}.md']
    bindings.update({str(p):lookup.sha(p) for p in sources})
    output.mkdir(parents=True)
    lookup.write_json(output/'protocol.json',dict(schema=TAG,P1=False,input_bindings=bindings,
        rules=['No new network requests','Verify both parent seals and all raw/decoded response digests',
               'Match complete local bytes against official tree blobs; never infer from similar text',
               'Do not promote public source/config to checkpoint training provenance']))
    source_seal=seal.create_seal(files=sources+[output/'protocol.json'],
        args_json=json.dumps(dict(protocol_sha256=lookup.sha(output/'protocol.json'))),output_dir=output/'source_seal')
    result=dict(schema=TAG,status='historical_contract_review_recorded',P1=False,source_seal=source_seal,
        protocol_sha256=lookup.sha(output/'protocol.json'),parent_report_sha256=[lookup.sha(d/'report.json') for d in (FIRST,SECOND)],
        official_repository=lookup.REPO,official_head=first['candidate_github_revisions'][0],
        public_history=facts,exact_local_bytes_matching_official_tree=local_matches,
        github_releases=lookup.parsed(records['github_releases']),github_tags=lookup.parsed(records['github_tags']),
        network_boundaries=dict(HF='All six scoped HF requests unavailable; no remote LFS digest or HF-tree absence conclusion',
            raw_GitHub='Connection reset/unreachable',GitHub_API='Public history/trees and six blobs obtained; then 403 rate limit, no API retries',
            public_GitHub_web='One HEAD-only reachability check failed; not used as artifact evidence'),
        verified_local_checkpoint_sha256=lookup.WEIGHTS,local_HF_cache_revision=lookup.REVISION,
        historical_contract=dict(
            checkpoint_remote_identity=first['remote_weight_identity']['status'],
            public_normalizer_algorithm='unknown from retrieved official bytes; public utils blob IDs recorded, local implementation not promoted',
            fitted_action_mean_std_ddof_and_action_units='unknown',
            checkpoint_training_commit_run='unknown',dataset_revision_and_data_selection='unknown',
            camera_render_resolution_crop_orientation='unknown',codec_and_encoder_parameters='unknown',
            known_public_data_format_transition='2026-05-14: updating data format to lance for training; public history only'),
        next_readonly_targets=missing_sources,
        decisive_witness_needed=['Pinned HF model tree LFS sha256 equal to fixed local digest',
            'Run/export manifest linking that weights digest to exact training commit and dependency versions',
            'Pinned dataset revision/content identity and exact fitted action mean/std, estimator and action units',
            'Data-generation camera/render/resize and storage codec/version/quality metadata for that dataset'],
        forbidden_inferences=['Upload/cache time is not training time','Lance migration does not prove this checkpoint used Lance/JPEG',
            'Official config blob match does not bind a model run','Dependency main reader/transform is not historical training provenance',
            'Network unavailable is not artifact absent','No stats replacement, camera fix or physical benefit is authorized by these unknowns'],
        counts=dict(successful_metadata_or_blob_GETs=sum(r.get('http_status')==200 for p in parents for r in p['requests']),
            recorded_GET_attempts=sum(p['counts']['requests'] for p in parents),
            total_downloaded_body_bytes=second['counts']['combined_body_bytes'],
            stage2_decoded_unique_blobs=second['counts']['decoded_unique_blobs'],
            transform_fixtures=0,neural=0,physics=0,weights_or_dataset_download_bytes=0,author_messages=0))
    if any(lookup.sha(p)!=digest for p,digest in bindings.items()):
        raise ValueError('review input changed')
    seal.verify_seal(output_dir=output/'source_seal',expected_manifest_sha256=source_seal['manifest_sha256'])
    result['all_bound_inputs_unchanged']=True
    lookup.write_json(output/'report.json',result)
    print(json.dumps(dict(report=str(output/'report.json'),sha256=lookup.sha(output/'report.json'),counts=result['counts'])))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path,required=True)
    run(parser.parse_args().out_dir.resolve())
