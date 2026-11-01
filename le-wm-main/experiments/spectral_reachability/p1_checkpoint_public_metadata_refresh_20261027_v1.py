"""Approval-gated anonymous public metadata refresh, never weights/data download.

No credentials, environment token lookup, SDK, netrc, redirects or URL discovery.
Training normalizer provenance stays unknown even when public hashes match.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import http.client
import json
import math
import multiprocessing
import os
from pathlib import Path
import re
import stat
import time
from urllib.parse import urlsplit

NAME = 'p1_checkpoint_public_metadata_refresh_20261027_v1'
REPO = 'quentinll/lewm-pusht'
HOSTS = {'official': 'huggingface.co', 'mirror': 'hf-mirror.com'}
API_URLS = {k: f'https://{host}/api/models/{REPO}?blobs=true' for k, host in HOSTS.items()}
SMALL_FILES = ('README.md', 'config.json', 'stats.json', 'action_stats.json',
               'normalizer.json', 'normalizer_stats.json', 'dataset_stats.json')
METADATA_CAP = 2 * 1024**2
SMALL_CAP = 256 * 1024
TIMEOUT_CAP = 15
CHECKPOINT = Path('/home/gbk/.stable_worldmodel/checkpoints/pusht/lewm')
LOCAL_WEIGHTS_CAP = 8 * 1024**3


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def is_digest(value, length):
    return type(value) is str and re.fullmatch('[0-9a-f]{'+str(length)+'}', value) is not None


def git_blob(raw):
    return hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest()


def allowed_url(url):
    # Exact comparison rejects credentials, ports, fragments, extra query params,
    # encoded path escapes, alternate repositories and resolve/LFS endpoints.
    if url in API_URLS.values():
        return True
    return any(re.fullmatch(re.escape(f'https://{host}/{REPO}/raw/') +
                           '[0-9a-f]{40}/(' + '|'.join(map(re.escape, SMALL_FILES)) + ')', url)
               is not None for host in HOSTS.values())


@dataclass(frozen=True)
class FetchResult:
    status: int | None
    body: bytes
    error: str | None
    complete: bool


ERRORS = {'timeout', 'transport_error', 'http_status_not_200', 'body_cap_reached',
          'redirect_not_followed', 'unsupported_encoding', 'truncated_body', 'worker_failed'}


def _network_worker(url, limit, timeout, pipe):
    """Child process: direct anonymous HTTPS, stream partial bytes to parent."""
    connection = None; used = 0
    try:
        parsed = urlsplit(url)
        connection = http.client.HTTPSConnection(parsed.hostname, timeout=timeout)
        target = parsed.path + ('?' + parsed.query if parsed.query else '')
        connection.request('GET', target, headers={
            'User-Agent': NAME, 'Accept': 'application/json,text/plain', 'Accept-Encoding': 'identity'})
        response = connection.getresponse()
        pipe.send(('status', response.status))
        length = response.getheader('Content-Length')
        expected = int(length) if length is not None else None
        if expected is not None and expected < 0:
            raise ValueError('invalid length')
        encoding = response.getheader('Content-Encoding', 'identity').lower()
        eof = False
        while used < limit:
            block = response.read1(min(16384, limit - used))
            if not block:
                eof = True; break
            used += len(block); pipe.send(('body', block))
        complete = eof or (expected is not None and expected == used)
        error = None
        if expected is not None and expected > limit or not complete and used == limit:
            error = 'body_cap_reached'
        elif expected is not None and expected != used:
            error = 'truncated_body'
        elif encoding not in ('', 'identity'):
            error = 'unsupported_encoding'
        elif 300 <= response.status < 400:
            error = 'redirect_not_followed'
        elif response.status != 200:
            error = 'http_status_not_200'
        pipe.send(('done', (error, complete)))
    except TimeoutError:
        pipe.send(('done', ('timeout', False)))
    except http.client.IncompleteRead as exc:
        if exc.partial:
            pipe.send(('body', exc.partial[:limit-used]))
        pipe.send(('done', ('truncated_body', False)))
    except Exception:
        # Never serialize exception strings/headers; they can contain credentials
        # or environment/proxy information. Parent preserves received raw prefix.
        pipe.send(('done', ('transport_error', False)))
    finally:
        if connection is not None:
            connection.close()
        pipe.close()


def fetch_public(url, limit, timeout):
    require(allowed_url(url), 'URL outside fixed public allowlist')
    require(type(timeout) in (int, float) and math.isfinite(timeout) and 0 < timeout <= TIMEOUT_CAP,
            'timeout must be finite and <=15 seconds')
    require(type(limit) is int and 0 < limit <= (METADATA_CAP if url in API_URLS.values() else SMALL_CAP), 'body limit')
    # Parent wall deadline also bounds DNS/TLS/slow reads. No global socket or
    # signal mutation, implicit authentication, retry or proxy environment use.
    context = multiprocessing.get_context('spawn')
    receive, send = context.Pipe(duplex=False)
    process = context.Process(target=_network_worker, args=(url, limit, timeout, send), daemon=True)
    deadline = time.monotonic() + timeout
    body = bytearray(); status = None; error = 'worker_failed'; complete = False
    started = False
    try:
        process.start(); started = True; send.close()
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not receive.poll(remaining):
                error = 'timeout'; break
            kind, value = receive.recv()
            if kind == 'status':
                require(type(value) is int and 100 <= value <= 599, 'worker status')
                status = value
            elif kind == 'body':
                require(type(value) is bytes, 'worker body')
                available = limit - len(body)
                body.extend(value[:available])
                if len(value) > available:
                    error = 'body_cap_reached'; complete = False; break
            elif kind == 'done':
                error, complete = value
                require(error is None or error in ERRORS, 'worker error code')
                break
            else:
                raise ValueError('worker protocol')
    except Exception:
        error = 'transport_error'; complete = False
    finally:
        receive.close(); send.close()
        if started:
            if process.is_alive():
                process.terminate()
            process.join(timeout=0.1)
            if process.is_alive():
                process.kill(); process.join(timeout=0.1)
    return FetchResult(status, bytes(body), error, complete)


class DiskStore:
    def __init__(self, output):
        self.output = Path(output)

    def begin(self):
        # Existing directory, ordinary file and dangling symlink all fail.
        self.output.mkdir(parents=False, exist_ok=False)

    def write(self, name, raw):
        path = self.output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())


def hash_local(path, kind):
    """Hash an explicit existing file; never deserialize weights or load config."""
    path = Path(path)
    cap = LOCAL_WEIGHTS_CAP if kind == 'weights' else SMALL_CAP
    require(not path.is_symlink(), 'local symlink is not a fixed file')
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and before.st_size <= cap, 'local regular file/size cap')
        sha = hashlib.sha256(); blob = hashlib.sha1(f'blob {before.st_size}\0'.encode()); used = 0
        while True:
            block = stream.read(min(1024**2, cap-used+1))
            if not block:
                break
            used += len(block); require(used <= cap, 'local file grew beyond cap')
            sha.update(block); blob.update(block)
        after = os.fstat(stream.fileno()); current = path.stat()
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    require(identity(before) == identity(after) == identity(current) and used == before.st_size,
            'local file changed while hashing')
    return dict(path=str(path), status='hashed_readonly', bytes=used, sha256=sha.hexdigest(),
                git_blob_sha1=blob.hexdigest() if kind == 'config' else None)


def parse_metadata(raw):
    def unique(items):
        out = {}
        for key, value in items:
            require(key not in out, 'duplicate metadata key')
            out[key] = value
        return out
    require(len(raw) <= METADATA_CAP, 'metadata byte cap')
    data = json.loads(raw, object_pairs_hook=unique)
    require(type(data) is dict and data.get('id') == REPO and is_digest(data.get('sha'), 40), 'fixed public repo/commit')
    require(data.get('private') is not True, 'public metadata only')
    siblings = data.get('siblings')
    require(type(siblings) is list and len(siblings) <= 4096, 'bounded siblings required')
    files = {}
    for item in siblings:
        require(type(item) is dict and type(item.get('rfilename')) is str, 'sibling descriptor')
        name = item['rfilename']; require(name not in files, 'duplicate sibling path')
        files[name] = item
    return data['sha'], files


def refresh(output, *, main_approved=False, sources=('official', 'mirror'), small_files=(), timeout=15,
            local_weights=CHECKPOINT/'weights.pt', local_config=CHECKPOINT/'config.json',
            transport=None, local_hasher=None, store_factory=DiskStore):
    require(main_approved is True, 'MAIN review and explicit network approval required')
    require(type(timeout) in (int, float) and math.isfinite(timeout) and 0 < timeout <= TIMEOUT_CAP,
            'timeout must be finite and <=15 seconds')
    require(bool(sources) and len(set(sources)) == len(sources) and all(s in HOSTS for s in sources), 'fixed origins only')
    require(len(set(small_files)) == len(small_files) and all(n in SMALL_FILES for n in small_files), 'fixed small-file allowlist only')
    store = store_factory(output); store.begin()
    transport = transport or fetch_public; local_hasher = local_hasher or hash_local
    protocol = dict(schema=NAME+'.protocol', repository=REPO, sources=list(sources), small_files=list(small_files),
                    api_urls={s:API_URLS[s] for s in sources}, timeout_seconds=timeout,
                    metadata_cap=METADATA_CAP, small_file_cap=SMALL_CAP,
                    max_requests=len(sources)*(1+len(small_files)),
                    max_body_bytes=len(sources)*(METADATA_CAP+len(small_files)*SMALL_CAP),
                    main_network_approval_asserted=True, anonymous=True, follow_redirects=False,
                    training_stats_provenance='unknown', credentials=False, weights_dataset_downloads=False)
    store.write('protocol.json', canonical(protocol))
    local = {}
    for kind, path in (('weights', local_weights), ('config', local_config)):
        try:
            local[kind] = local_hasher(path, kind)
        except Exception:
            local[kind] = dict(path=str(path), status='unknown', sha256=None, git_blob_sha1=None, error='local_hash_unavailable')
    records = []; origins = {}

    def request(source, label, url, cap):
        require(allowed_url(url), 'fixed URL policy')
        started = datetime.now(timezone.utc).isoformat()
        try:
            fetched = transport(url, cap, timeout)
        except Exception:
            fetched = FetchResult(None, b'', 'transport_error', False)
        body = fetched.body[:cap]
        error = fetched.error if fetched.error in ERRORS else ('transport_error' if fetched.error is not None else None)
        complete = fetched.complete is True
        if len(fetched.body) > cap:
            error = 'body_cap_reached'; complete = False
        status = fetched.status if type(fetched.status) is int and 100 <= fetched.status <= 599 else None
        if error is None and status != 200:
            error = 'redirect_not_followed' if status is not None and 300 <= status < 400 else 'http_status_not_200'
        if error is None and not complete:
            error = 'truncated_body'
        name = source+'/'+label+'.body'
        record = dict(source=source, url=url, method='GET', http_status=status, body_sha256=sha256(body),
                      rawbody_path=name, body_bytes=len(body), complete=complete, error=error,
                      started_utc=started, finished_utc=datetime.now(timezone.utc).isoformat())
        # Persist raw bytes (including empty/partial failures) before interpretation.
        store.write(name, body); store.write(source+'/'+label+'.receipt.json', canonical(record))
        records.append(record)
        return record, body

    for source in sources:
        origin = dict(source_kind='official_domain' if source == 'official' else 'mirror_domain_not_official_attestation',
                      commit=None, metadata_error=None, weights_lfs_sha256=None, config_git_blob_sha1=None,
                      weights_lfs_match_local=None, config_git_blob_match_local=None, config_body_sha256_match_local=None,
                      training_stats_provenance='unknown', skipped_files=[], files=[])
        origins[source] = origin
        record, body = request(source, 'metadata', API_URLS[source], METADATA_CAP)
        if record['error'] is not None or not record['complete']:
            origin['metadata_error'] = 'metadata_request_unavailable'; continue
        try:
            commit, files = parse_metadata(body)
        except (ValueError, TypeError, UnicodeError, RecursionError):
            origin['metadata_error'] = 'invalid_public_metadata'; continue
        origin['commit'] = commit
        lfs = files.get('weights.pt', {}).get('lfs')
        weights_sha = lfs.get('sha256') if type(lfs) is dict else None
        if is_digest(weights_sha, 64):
            origin['weights_lfs_sha256'] = weights_sha
            if is_digest(local['weights'].get('sha256'), 64):
                origin['weights_lfs_match_local'] = weights_sha == local['weights']['sha256']
        config_oid = files.get('config.json', {}).get('blobId')
        if is_digest(config_oid, 40):
            origin['config_git_blob_sha1'] = config_oid
            if is_digest(local['config'].get('git_blob_sha1'), 40):
                origin['config_git_blob_match_local'] = config_oid == local['config']['git_blob_sha1']
        for name in small_files:
            descriptor = files.get(name)
            size = descriptor.get('size') if descriptor is not None else None
            if descriptor is None or type(size) is int and size > SMALL_CAP:
                origin['skipped_files'].append(dict(name=name, reason='not_advertised' if descriptor is None else 'advertised_size_exceeds_cap'))
                continue
            url = f'https://{HOSTS[source]}/{REPO}/raw/{commit}/{name}'
            receipt, raw = request(source, name, url, SMALL_CAP)
            item = dict(name=name, url=url, body_sha256=receipt['body_sha256'], error=receipt['error'],
                        metadata_git_blob_sha1=descriptor.get('blobId') if is_digest(descriptor.get('blobId'), 40) else None,
                        body_git_blob_match_metadata=None, training_stats_provenance='unknown')
            if receipt['error'] is None and receipt['complete']:
                if item['metadata_git_blob_sha1'] is not None:
                    item['body_git_blob_match_metadata'] = git_blob(raw) == item['metadata_git_blob_sha1']
                if name == 'config.json' and is_digest(local['config'].get('sha256'), 64):
                    origin['config_body_sha256_match_local'] = sha256(raw) == local['config']['sha256']
            origin['files'].append(item)
    result = dict(schema=NAME+'.report', repository=REPO, protocol_sha256=sha256(canonical(protocol)), local=local,
                  requests=records, origins=origins, cross_origin_independence='unknown',
                  training_stats_provenance='unknown', training_normalizer_certified=False, P1_admission=False,
                  official_export_manufactured=False,
                  remaining_unknowns=['weights_to_training_run_source_dependency_dataset_binding',
                                      'fitted_training_normalizer_export_bound_to_weights', 'historical_camera_codec'],
                  note='Hash equality is byte identity evidence at each origin only, not training stats provenance. Mirror is not an independent official attestation; failures do not prove absence.')
    store.write('report.json', canonical(result))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', required=True, help='exclusive NEW directory; parent must exist')
    parser.add_argument('--main-approved-network', action='store_true', required=True,
                        help='assert MAIN has reviewed and explicitly approved this network run')
    parser.add_argument('--sources', nargs='+', choices=tuple(HOSTS), default=list(HOSTS))
    parser.add_argument('--small-files', nargs='*', choices=SMALL_FILES, default=[])
    parser.add_argument('--timeout', type=float, default=15)
    parser.add_argument('--local-weights', type=Path, default=CHECKPOINT/'weights.pt')
    parser.add_argument('--local-config', type=Path, default=CHECKPOINT/'config.json')
    args = parser.parse_args(argv)
    result = refresh(args.out_dir, main_approved=args.main_approved_network, sources=tuple(args.sources),
                     small_files=tuple(args.small_files), timeout=args.timeout,
                     local_weights=args.local_weights, local_config=args.local_config)
    print(json.dumps(dict(report=str(Path(args.out_dir).absolute()/'report.json'),
                          sha256=sha256(canonical(result)), requests=len(result['requests']),
                          training_stats_provenance='unknown'), sort_keys=True))


if __name__ == '__main__':
    main()
