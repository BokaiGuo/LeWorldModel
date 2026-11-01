"""Offline final-archive verification. Never deserialize, extract or fetch assets."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
import time
import zipfile
import zlib
from dataclasses import dataclass

STEM = 'agent_hilewm_asset_verify_20261028_v1'
ARCHIVE_SIZE = 2299863924
ARCHIVE_MD5 = 'aaf78b8b4fd6f5d1f5e720927951152f'
TAIL_START = 2299798367
TAIL_SHA = '43a0c87d4f1b652caccbc26ac5e1b1ed73c0c6cf97c8e00683ecc1bdd8c16aef'
API_SHA = '2fc221128c12b609b86dfb7810eda46f2399bc796348f36c5d7e016ab25dd773'
CONTRACT_SHA = 'c25edf542f38faa9cf0150df4822198d28753e2cf77a0d243e990ffabe849ded'
URL = 'https://zenodo.org/api/records/21353240/files/package.zip/content'
OBJECT = 'package/checkpoints/pusht/main/pusht_hi_lewm_epoch15_object.ckpt'
CONFIG = 'package/checkpoints/pusht/main/config.yaml'
CHUNK = 1024 * 1024
PENDING = [
    'trusted_pickle_loading_and_dependency_class_review',
    'embedded_low_tensor_config_pairing',
    'original_dataset_and_normalizer_identity',
    'selected_protocol_and_empirical_macro_bank_binding_if_C_variant',
    'third_party_license_scope_and_environment_revision_review',
    'MAIN_authorized_CUDA_interface_validation',
]


class Invalid(ValueError):
    def __init__(self, code, detail):
        super().__init__(detail)
        self.code = code


def need(ok, code, detail):
    if not ok:
        raise Invalid(code, detail)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def safe_name(name):
    need(isinstance(name, str) and bool(name), 'unsafe_path', repr(name))
    trimmed = name[:-1] if name.endswith('/') else name
    need(not name.startswith('/') and '\\' not in name and '\x00' not in name
         and ':' not in name and all(p not in ('', '.', '..') for p in trimmed.split('/')),
         'unsafe_path', repr(name))
    need(not any(ord(c) < 32 or ord(c) == 127 for c in name), 'unsafe_path', repr(name))


def metadata(info, offset=0):
    return {'name': info.filename, 'size': info.file_size, 'packed': info.compress_size,
            'crc32': info.CRC, 'method': info.compress_type, 'flags': info.flag_bits,
            'offset': info.header_offset + offset, 'external_attr': info.external_attr,
            'create_system': info.create_system}


def index(infos, offset=0):
    result = {}
    for i in infos:
        safe_name(i.orig_filename)
        need(i.filename == i.orig_filename, 'unsafe_path', repr(i.orig_filename))
        need(i.filename not in result, 'duplicate_path', i.filename)
        mode = i.external_attr >> 16
        kind = stat.S_IFMT(mode)
        need(kind in (0, stat.S_IFREG, stat.S_IFDIR), 'special_member', i.filename)
        need(kind != stat.S_IFDIR or i.is_dir(), 'directory_type', i.filename)
        need(not i.is_dir() or (i.file_size == 0 and kind != stat.S_IFREG),
             'directory_type', i.filename)
        need(not i.flag_bits & (1 | 64), 'encrypted_member', i.filename)
        need(i.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED),
             'unsupported_compression', i.filename)
        need(i.file_size >= 0 and i.compress_size >= 0, 'negative_size', i.filename)
        result[i.filename] = metadata(i, offset)
    return result


@dataclass(frozen=True)
class Binding:
    size: int
    md5: str
    tail_size: int
    tail_sha256: str
    entries: dict
    texts: dict
    required: tuple
    provenance: dict


def load_binding(project: Path) -> Binding:
    """Only small, pinned local reference files; never touches an archive candidate."""
    inventory = project / 'results/p1_learned_subgoal_remote_inventory_20261027_v1'
    textroot = project / 'results/p1_hilewm_text_contract_20261027_v1'
    raw = (inventory / 'zenodo_api.json').read_bytes()
    need(sha(raw) == API_SHA, 'reference_changed', 'zenodo_api.json')
    api = json.loads(raw)
    files = [f for f in api['files'] if f['key'] == 'package.zip']
    need(api['id'] == 21353240 and len(files) == 1, 'reference_changed', 'record/file')
    row = files[0]
    need(row['size'] == ARCHIVE_SIZE and row['checksum'] == 'md5:' + ARCHIVE_MD5
         and row['links']['self'] == URL, 'reference_changed', 'official archive spec')
    tail = (inventory / 'archive_tail.bin').read_bytes()
    need(len(tail) == 65557 and sha(tail) == TAIL_SHA, 'reference_changed', 'archive_tail.bin')
    with zipfile.ZipFile(io.BytesIO(tail)) as z:
        entries = index(z.infolist(), TAIL_START)
    need(len(entries) == 414 and sum(i['size'] for i in entries.values()) == 2500619520,
         'reference_changed', 'complete directory')
    raw = (textroot / 'contract.json').read_bytes()
    need(sha(raw) == CONTRACT_SHA, 'reference_changed', 'contract.json')
    contract = json.loads(raw)
    texts = {}
    for name, row in contract['sources'].items():
        safe_name(name)
        safe_name(row['local_text'])
        need(row['archive_member'] == name and name in entries,
             'reference_changed', name)
        text = (textroot / row['local_text']).read_bytes()
        need(len(text) == entries[name]['size'] and sha(text) == row['sha256'],
             'reference_changed', name)
        texts[name] = row['sha256']
    need(len(texts) == 33 and CONFIG in texts and OBJECT in entries,
         'reference_changed', 'critical source inventory')
    return Binding(ARCHIVE_SIZE, ARCHIVE_MD5, len(tail), TAIL_SHA, entries, texts,
                   (OBJECT,) + tuple(texts),
                   {'official_url': URL, 'api_sha256': API_SHA, 'tail_sha256': TAIL_SHA,
                    'text_contract_sha256': CONTRACT_SHA, 'record_license': api['metadata']['license'],
                    'author_object_sha256': None, 'immutable_ETag': None})


def new_report(binding):
    return {'schema': STEM + '.report', 'status': 'unverified',
            'layers': {'download_complete': None, 'zip_safely_readable': None,
                       'baseline_runnable': None},
            'scope': 'known_PushT_main_archive_bytes_only_not_pickle_safety',
            'provenance': binding.provenance, 'archive': {}, 'members': [],
            'missing_members': [], 'errors': [], 'baseline_missing_evidence': list(PENDING),
            'unpickled': False, 'weights_loaded': False, 'extracted': False,
            'network_calls': 0, 'neural_calls': 0, 'physics_calls': 0,
            'scientific_admission': False}


def inspect_stream(stream, binding, report, tick):
    """Seekable RAM/file seam. Caller owns immutable input and exception reporting."""
    tick()
    stream.seek(0, 2)
    actual_size = stream.tell()
    report['archive']['bytes'] = actual_size
    need(actual_size == binding.size, 'archive_size_mismatch',
         f'expected={binding.size}, actual={actual_size}, missing={max(0, binding.size-actual_size)}')
    stream.seek(0)
    md5, h, n = hashlib.md5(), hashlib.sha256(), 0
    while True:
        tick()
        block = stream.read(CHUNK)
        if not block:
            break
        n += len(block)
        need(n <= binding.size, 'archive_changed', 'size grew during read')
        md5.update(block)
        h.update(block)
    report['archive'].update(md5=md5.hexdigest(), sha256_measured=h.hexdigest())
    need(n == binding.size, 'archive_changed', 'size changed during read')
    need(md5.hexdigest() == binding.md5, 'archive_md5_mismatch', 'official full-package checksum')
    report['layers']['download_complete'] = True
    stream.seek(-binding.tail_size, 2)
    need(sha(stream.read(binding.tail_size)) == binding.tail_sha256,
         'tail_mismatch', 'frozen directory/tail differs')
    stream.seek(0)
    with zipfile.ZipFile(stream) as z:
        infos = z.infolist()
        actual = index(infos)
        missing = sorted(set(binding.entries) - actual.keys())
        extra = sorted(actual.keys() - set(binding.entries))
        report['missing_members'] = missing
        need(not missing and not extra, 'inventory_mismatch', json.dumps({'missing': missing, 'extra': extra}))
        need(all(name in actual for name in binding.required), 'critical_member_missing', 'required set')
        for name, expected in binding.entries.items():
            need(actual[name] == expected, 'member_metadata_mismatch', name)
        for i in infos:
            tick()
            name, count, crc, h = i.filename, 0, 0, hashlib.sha256()
            # Even directory local headers are checked. No extract or nested checkpoint parsing.
            with z.open(i, 'r') as f:
                while True:
                    tick()
                    block = f.read(CHUNK)
                    if not block:
                        break
                    count += len(block)
                    need(count <= i.file_size, 'member_size_mismatch', name)
                    h.update(block)
                    crc = zlib.crc32(block, crc)
            need(count == i.file_size, 'member_size_mismatch', name)
            need(crc & 0xffffffff == i.CRC, 'member_crc_mismatch', name)
            text_matched = None
            if name in binding.texts:
                text_matched = h.hexdigest() == binding.texts[name]
                need(text_matched, 'source_config_mismatch', name)
            report['members'].append({'name': name, 'bytes': count, 'crc32': f'{crc & 0xffffffff:08x}',
                                      'sha256_measured': h.hexdigest(), 'source_text_matched': text_matched})
        report['zip_summary'] = {'entries_verified': len(infos),
                                 'expanded_bytes_verified': sum(i.file_size for i in infos),
                                 'source_texts_matched': len(binding.texts),
                                 'all_member_CRC_checked': True}
    report['layers']['zip_safely_readable'] = True
    report['status'] = 'archive_verified_baseline_unestablished'


def snapshot(s):
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def diagnose_download_receipt(path: Path, expected_sha256: str, project: Path):
    """Read a terminal JSON receipt, never its filename_effective/partial payload."""
    b = load_binding(project)
    need(re.fullmatch('[0-9a-f]{64}', expected_sha256 or '') is not None,
         'receipt_sha_required', 'MAIN actual terminal report SHA required')
    need(path.is_absolute() and path.suffix == '.json', 'receipt_path', 'absolute terminal JSON path required')
    with path.open('rb') as f:
        raw = f.read(2 * 1024 * 1024 + 1)
    need(len(raw) <= 2 * 1024 * 1024, 'receipt_cap', 'terminal JSON exceeds 2MiB')
    need(sha(raw) == expected_sha256, 'receipt_sha_mismatch', str(path))
    saved = json.loads(raw)
    need(saved.get('schema') == 'MAIN.hilewm_whole_archive_result.v1'
         and saved.get('status') == 'failed_preserved', 'receipt_schema', 'need supported terminal failure')
    transfer = saved['transfer']
    need(transfer['url_effective'] == URL, 'receipt_origin', 'not the fixed official source')
    actual = saved['actual_saved_bytes']
    need(type(actual) is int and 0 <= actual < b.size, 'receipt_size', 'not an incomplete whole archive')
    need(transfer['size_download'] == actual and saved['curl_exit_code'] == transfer['exitcode'],
         'receipt_inconsistent', 'curl byte/exit counts disagree')
    http, exitcode = transfer['http_code'], saved['curl_exit_code']
    require_missing = b.size - actual
    if http == 200 and exitcode == 18:
        match = re.fullmatch(r'transfer closed with (\d+) bytes remaining to read', transfer['errormsg'])
        need(match is not None and int(match[1]) == require_missing,
             'receipt_inconsistent', 'early EOF missing-byte receipt mismatch')
        diagnosis = 'http200_early_EOF_truncated_not_403'
    elif http == 403:
        diagnosis = 'http403_forbidden_not_classified_as_EOF'
    else:
        diagnosis = 'incomplete_transfer_cause_unknown'
    r = new_report(b)
    r.update(status='rejected', receipt_sha256=expected_sha256,
             receipt_path=str(path), source_sha256=sha(Path(__file__).read_bytes()),
             download_diagnosis=diagnosis, payload_opened=False, automatic_retry=False)
    r['layers']['download_complete'] = False
    r['layers']['zip_safely_readable'] = False
    r['archive'] = {'expected_bytes': b.size, 'reported_saved_bytes': actual,
                    'missing_bytes': require_missing, 'sha256_measured': None,
                    'reported_http_status': http, 'reported_curl_exit': exitcode,
                    'reported_transfer_seconds': transfer.get('time_total')}
    r['errors'] = [{'code': diagnosis, 'detail': 'Do not unpack partial archive; preserve, no automatic retry.'}]
    return r


def verify_archive(archive: Path, project: Path, *, final_file=False, wall_seconds=900):
    binding = load_binding(project)
    report = new_report(binding)
    report['source_sha256'] = sha(Path(__file__).read_bytes())
    report['archive']['path'] = str(archive)
    start = time.monotonic()

    def tick():
        need(time.monotonic() - start < wall_seconds, 'wall_cap', 'offline verification wall budget')

    try:
        need(final_file, 'final_file_ack_required', 'MAIN must attest download is closed/final')
        need(0 < wall_seconds <= 900, 'wall_cap_invalid', 'allowed 0 < seconds <= 900')
        need(archive.is_absolute(), 'absolute_path_required', str(archive))
        need(not any(s in ('.part', '.partial', '.tmp', '.download') for s in archive.suffixes),
             'active_partial_forbidden', str(archive))
        # No stat/open of partial names above. Parent symlinks are not a trust boundary.
        fd = os.open(archive, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, 'rb') as f:
            before = os.fstat(f.fileno())
            need(stat.S_ISREG(before.st_mode), 'regular_file_required', str(archive))
            try:
                inspect_stream(f, binding, report, tick)
            finally:
                after = os.fstat(f.fileno())
                need(snapshot(before) == snapshot(after), 'archive_changed', 'input mutated during verification')
                need(snapshot(os.stat(archive, follow_symlinks=False)) == snapshot(after),
                     'archive_changed', 'path replaced during verification')
        tick()
    except Exception as exc:
        report['status'] = 'rejected'
        report['layers']['zip_safely_readable'] = False
        if getattr(exc, 'code', None) == 'archive_changed':
            report['layers']['download_complete'] = None
        report['errors'].append({'code': getattr(exc, 'code', type(exc).__name__), 'detail': str(exc)})
    report['elapsed_seconds'] = time.monotonic() - start
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['plan', 'verify', 'receipt'])
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--acknowledge-final-file', action='store_true')
    parser.add_argument('--output', type=Path, help='new exclusive report; parent must already exist')
    parser.add_argument('--wall-seconds', type=float, default=900)
    parser.add_argument('--download-report', type=Path, help='terminal JSON only; never follows its partial path')
    parser.add_argument('--download-report-sha256', default='')
    args = parser.parse_args(argv)
    try:
        if args.mode == 'plan':
            b = load_binding(args.project)
            report = {'schema': STEM + '.plan', 'archive_bytes': b.size, 'official_md5': b.md5,
                      'members': len(b.entries), 'expanded_bytes': sum(r['size'] for r in b.entries.values()),
                      'source_texts': len(b.texts), 'provenance': b.provenance,
                      'baseline_missing_evidence': PENDING, 'archive_candidate_opened': False}
        elif args.mode == 'receipt':
            need(args.download_report is not None, 'receipt_path', '--download-report required')
            report = diagnose_download_receipt(args.download_report, args.download_report_sha256, args.project)
        else:
            need(args.archive is not None, 'archive_required', '--archive final file required')
            report = verify_archive(args.archive, args.project,
                                    final_file=args.acknowledge_final_file, wall_seconds=args.wall_seconds)
        raw = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n'
        need(len(raw.encode()) <= 2 * 1024 * 1024, 'report_cap', '2MiB output cap')
        if args.output is not None:
            with args.output.open('x', encoding='utf-8') as f:
                f.write(raw)
        else:
            print(raw, end='')
        return int(report.get('status') == 'rejected')
    except Exception as exc:
        print(json.dumps({'schema': STEM + '.error', 'code': getattr(exc, 'code', type(exc).__name__),
                          'detail': str(exc)}))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
