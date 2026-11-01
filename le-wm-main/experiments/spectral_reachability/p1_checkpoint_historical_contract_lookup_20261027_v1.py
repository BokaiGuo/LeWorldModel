"""Bounded public metadata/source lookup. No models, data downloads or contacts."""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
from pathlib import Path
import threading
import urllib.error
import urllib.parse
import urllib.request

TAG = 'p1_checkpoint_historical_contract_lookup_20261027_v1'
ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = Path('/home/gbk/.stable_worldmodel/checkpoints/pusht/lewm')
REVISION = '22b330c28c27ead4bfd1888615af1340e3fe9052'
WEIGHTS = '48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607'
REPO = 'lucas-maes/le-wm'
DEPENDENCIES = ('galilai-group/stable-worldmodel', 'galilai-group/stable-pretraining')
MAX_RESPONSE = 512*1024
MAX_TOTAL = 8*1024*1024
MAX_REQUESTS = 60
TEXT_FILES = ('README.md', 'train.py', 'utils.py', 'pyproject.toml', 'requirements.txt',
              'config/train/lewm.yaml', 'config/train/data/pusht.yaml',
              'config/eval/pusht.yaml', 'config/train/model/lewm.yaml', 'uv.lock')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, obj):
    with Path(path).open('x') as stream:
        json.dump(obj, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def allowed(url):
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != 'https' or parsed.username or parsed.password or parsed.port not in (None,443):
        return False
    path = parsed.path
    repos = (REPO, *DEPENDENCIES)
    if parsed.hostname == 'api.github.com':
        return any(path == '/repos/'+r or path.startswith('/repos/'+r+'/') for r in repos) and not any(
            x in path for x in ('/zipball', '/tarball', '/releases/assets/', '/actions/artifacts/'))
    if parsed.hostname == 'raw.githubusercontent.com':
        return any(path.startswith('/'+r+'/') for r in repos) and path.endswith(('.md','.py','.yaml','.yml','.toml','.txt','.lock'))
    if parsed.hostname in ('huggingface.co', 'hf.co'):
        if path.startswith(('/api/models/quentinll/lewm-pusht', '/api/datasets/quentinll/lewm-pusht')):
            return True
        prefixes = ('/quentinll/lewm-pusht/resolve/', '/datasets/quentinll/lewm-pusht/resolve/')
        return path.startswith(prefixes) and path.rsplit('/',1)[-1] in ('README.md','config.json','.gitattributes')
    return False


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Even allowed redirects are not followed implicitly; receipts retain Location.
        return None


def read_bounded(response, limit=MAX_RESPONSE):
    length = response.headers.get('Content-Length')
    if length is not None and int(length) > limit:
        raise ValueError('response Content-Length exceeds cap')
    data = response.read(limit+1)
    if len(data) > limit:
        raise ValueError('response stream exceeds cap')
    return data


class Collector:
    def __init__(self, output):
        self.output = output
        self.records = []
        self.lock = threading.Lock()
        self.reserved = 0
        self.used = 0
        self.requests = 0
        self.opener = urllib.request.build_opener(NoRedirect())

    def fetch(self, label, url):
        if not allowed(url):
            raise ValueError('URL outside frozen public metadata/source allowlist')
        with self.lock:
            if self.requests >= MAX_REQUESTS or self.used+self.reserved+MAX_RESPONSE > MAX_TOTAL:
                raise ValueError('network request/byte budget exhausted')
            self.requests += 1
            request_id = self.requests
            self.reserved += MAX_RESPONSE
        record = dict(id=request_id, label=label, url=url, method='GET',
                      started_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        data = b''
        try:
            request = urllib.request.Request(url, headers={'User-Agent':TAG,
                'Accept':'application/vnd.github+json' if 'api.github.com' in url else 'application/json,text/plain,*/*',
                'Accept-Encoding':'identity'})
            try:
                response = self.opener.open(request, timeout=12)
            except urllib.error.HTTPError as error:
                response = error
            with response:
                record.update(http_status=response.status, response_headers={k:response.headers[k] for k in
                    ('ETag','Last-Modified','Date','Content-Type','Content-Length','Location','X-RateLimit-Remaining','Link')
                    if response.headers.get(k) is not None})
                data = read_bounded(response)
            body = self.output/'responses'/f'{request_id:03d}_{label}.body'
            with body.open('xb') as stream:
                stream.write(data)
            record.update(body_path=str(body), body_sha256=hashlib.sha256(data).hexdigest(),
                          body_bytes=len(data), status='recorded')
        except (OSError, ValueError, urllib.error.URLError) as error:
            record.update(status='unavailable', error_type=type(error).__name__, error=str(error))
        finally:
            record['finished_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
            with self.lock:
                self.used += len(data)
                self.reserved -= MAX_RESPONSE
                self.records.append(record)
                with (self.output/'journal.jsonl').open('a') as stream:
                    stream.write(json.dumps(record, sort_keys=True)+'\n')
            print(json.dumps({k:record.get(k) for k in ('label','http_status','status','body_bytes','error')}), flush=True)
        return record

    def batch(self, queries):
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            return list(pool.map(lambda row:self.fetch(*row), queries))


def parsed(record):
    if record.get('http_status') != 200 or record['status'] != 'recorded':
        return None
    try:
        return json.loads(Path(record['body_path']).read_text())
    except (ValueError, UnicodeError):
        return None


def hf_digest_witness(records):
    candidates = []
    for record in records:
        data = parsed(record)
        if record['label'] != 'hf_pinned_model' or not isinstance(data,dict):
            continue
        for file in data.get('siblings',[]):
            if file.get('rfilename') == 'weights.pt':
                candidates.append(dict(repository_revision=data.get('sha'), lfs=file.get('lfs'),
                                       response_sha256=record['body_sha256']))
    matched = any(v['repository_revision'] == REVISION and isinstance(v['lfs'],dict) and
                  v['lfs'].get('sha256') == WEIGHTS for v in candidates)
    return dict(status='verified_remote_LFS_identity' if matched else 'unknown', witnesses=candidates)


def collect(output):
    from experiments.spectral_reachability import d2p_prerun_source_seal as seal
    if output.exists():
        raise ValueError('exclusive NEW output required')
    sources = [Path(__file__), ROOT/f'tests/test_{TAG}.py', ROOT/f'innerdocs/{TAG}.md', Path(seal.__file__)]
    local = [CHECKPOINT/name for name in ('weights.pt','config.json','README.md')]
    local += [CHECKPOINT/'.cache/huggingface/download'/f'{name}.metadata' for name in ('weights.pt','config.json','README.md')]
    bindings = {str(p):sha(p) for p in sources+local}
    if bindings[str(CHECKPOINT/'weights.pt')] != WEIGHTS:
        raise ValueError('fixed checkpoint weight digest differs')
    metadata = (CHECKPOINT/'.cache/huggingface/download/weights.pt.metadata').read_text().splitlines()
    if metadata[:2] != [REVISION,WEIGHTS]:
        raise ValueError('local pinned HF revision/digest differs')
    output.mkdir(parents=True)
    (output/'responses').mkdir()
    protocol = dict(schema=TAG, P1=False, source_and_local_bindings=bindings,
        checkpoint_sha256=WEIGHTS, hf_revision=REVISION, official_candidate_repository=REPO,
        request_policy=dict(max_requests=MAX_REQUESTS, max_response_bytes=MAX_RESPONSE,
                            max_total_body_bytes=MAX_TOTAL, timeout_seconds=12, workers=4,
                            redirect_following=False, credentials=False, methods=['GET']),
        history_selection='Latest public head plus oldest returned train.py revision (per_page=30, no outcome use). '
                          'Record pagination/truncation, no claim of exhaustive history.',
        authority_boundary='Official-source definitions are candidate historical contracts, not proof the fixed checkpoint used them.',
        forbidden=['weights/data body downloads','training','neural execution','physics','author contact','old seal edits'])
    write_json(output/'protocol.json',protocol)
    snapshot = seal.create_seal(files=sources+[output/'protocol.json'],
        args_json=json.dumps(dict(protocol_sha256=sha(output/'protocol.json'))), output_dir=output/'source_seal')
    collector = Collector(output)
    gh = 'https://api.github.com/repos/'+REPO
    first = collector.batch([
        ('hf_pinned_model',f'https://huggingface.co/api/models/quentinll/lewm-pusht/revision/{REVISION}?blobs=true'),
        ('hf_model_tree',f'https://huggingface.co/api/models/quentinll/lewm-pusht/tree/{REVISION}?recursive=true&expand=true'),
        ('hf_model_history','https://huggingface.co/api/models/quentinll/lewm-pusht/commits/main'),
        ('hf_pinned_card',f'https://huggingface.co/quentinll/lewm-pusht/resolve/{REVISION}/README.md'),
        ('hf_pinned_config',f'https://huggingface.co/quentinll/lewm-pusht/resolve/{REVISION}/config.json'),
        ('hf_dataset_metadata','https://huggingface.co/api/datasets/quentinll/lewm-pusht?blobs=true'),
        ('github_repository',gh), ('github_head',gh+'/commits/main'),
        ('github_releases',gh+'/releases?per_page=20'), ('github_tags',gh+'/tags?per_page=30'),
        ('github_train_history',gh+'/commits?path=train.py&per_page=30'),
        ('github_utils_history',gh+'/commits?path=utils.py&per_page=30'),
        ('github_data_config_history',gh+'/commits?path=config/train/data/pusht.yaml&per_page=30'),
    ])
    by_label = {r['label']:r for r in first}
    head_data = parsed(by_label['github_head'])
    head = head_data.get('sha') if isinstance(head_data,dict) else None
    history = parsed(by_label['github_train_history'])
    revisions = list(dict.fromkeys(v for v in (head, history[-1].get('sha') if isinstance(history,list) and history else None) if v))
    queries = []
    for i,revision in enumerate(revisions):
        queries.append((f'github_tree_{i}', gh+f'/git/trees/{revision}?recursive=1'))
        for path in TEXT_FILES:
            # Optional missing files produce recorded 404s, not invented versions.
            queries.append((f'github_source_{i}_'+path.replace('/','_').replace('.','_'),
                            f'https://raw.githubusercontent.com/{REPO}/{revision}/{path}'))
    second = collector.batch(queries)
    dependency_queries = []
    for i,repo in enumerate(DEPENDENCIES):
        dependency_queries.extend([(f'dependency_head_{i}',f'https://api.github.com/repos/{repo}/commits/main'),
                                   (f'dependency_tags_{i}',f'https://api.github.com/repos/{repo}/tags?per_page=20')])
    dependencies = collector.batch(dependency_queries)
    # Inspect only relevant small current dependency source. No dependency code is executed.
    for i,repo in enumerate(DEPENDENCIES):
        record = next(r for r in dependencies if r['label'] == f'dependency_head_{i}')
        data = parsed(record)
        if not isinstance(data,dict) or 'sha' not in data:
            continue
        files = ('pyproject.toml','stable_worldmodel/data/formats/hdf5.py','stable_worldmodel/data/formats/lance.py') if i == 0 else (
            'pyproject.toml','stable_pretraining/data/transforms.py')
        collector.batch([(f'dependency_source_{i}_'+p.replace('/','_').replace('.','_'),
                          f'https://raw.githubusercontent.com/{repo}/{data["sha"]}/{p}') for p in files])
    records = sorted(collector.records, key=lambda r:r['id'])
    # Make narrow candidate-contract excerpts with exact URL/body hash/line anchors.
    terms = ('normaliz','mean','std','ImageNet','Resize','ToImage','frameskip','pusht','jpeg','codec','hdf5','lance',
             'camera','render','stable-worldmodel','stable-pretraining','weights.pt','checkpoint','commit','wandb')
    excerpts = []
    for record in records:
        if record.get('http_status') != 200 or not ('source' in record['label'] or record['label'] == 'hf_pinned_card'):
            continue
        text = Path(record['body_path']).read_text(errors='replace')
        excerpts.append(dict(url=record['url'],body_path=record['body_path'],sha256=record['body_sha256'],
            lines=[dict(line=i+1,text=line) for i,line in enumerate(text.splitlines()) if any(t.lower() in line.lower() for t in terms)]))
    for path,digest in bindings.items():
        if sha(path) != digest:
            raise ValueError('bound source/local checkpoint changed: '+path)
    seal.verify_seal(output_dir=output/'source_seal',expected_manifest_sha256=snapshot['manifest_sha256'])
    report = dict(schema=TAG,status='bounded_historical_lookup_recorded',P1=False,
        protocol_sha256=sha(output/'protocol.json'),source_seal=snapshot,
        source_and_local_inputs_unchanged=True, requests=records,
        remote_weight_identity=hf_digest_witness(records), candidate_github_revisions=revisions,
        candidate_contract_excerpts=excerpts,
        unresolved=dict(checkpoint_to_training_commit='unknown', checkpoint_to_fitted_action_normalizer='unknown',
                        checkpoint_to_dataset_revision='unknown', checkpoint_to_camera_and_codec='unknown'),
        counts=dict(requests=collector.requests,body_bytes=collector.used,neural_predict=0,neural_encode=0,
                    physics_steps=0,weight_or_dataset_download_bytes=0,author_messages=0),
        claim_boundary='Network failures are unavailable evidence, not absent artifacts. Public implementation '
                       'and dependency main are candidates only until linked to this checkpoint by a run/version witness.')
    write_json(output/'report.json',report)
    print(json.dumps(dict(report=str(output/'report.json'),sha256=sha(output/'report.json'),counts=report['counts'])),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path,required=True)
    collect(parser.parse_args().out_dir.resolve())
