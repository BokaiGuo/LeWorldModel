"""One bounded whole-archive transfer, independent of failed range probes.

Only retrieves public bytes and verifies publisher size/MD5. No model loading,
pickle parsing, extraction, training, or scientific admission.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_SIZE = 2299863924
EXPECTED_MD5 = 'aaf78b8b4fd6f5d1f5e720927951152f'
API_SHA = '2fc221128c12b609b86dfb7810eda46f2399bc796348f36c5d7e016ab25dd773'
URL = 'https://zenodo.org/api/records/21353240/files/package.zip/content'


def write_new(path, value):
    payload = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    with path.open('xb') as stream:
        if stream.write(payload) != len(payload):
            raise OSError('short metadata write')
        stream.flush()
        os.fsync(stream.fileno())


def main():
    api_path = ROOT / 'results/p1_learned_subgoal_remote_inventory_20261027_v1/zenodo_api.json'
    api_raw = api_path.read_bytes()
    if hashlib.sha256(api_raw).hexdigest() != API_SHA:
        raise ValueError('primary cached API changed')
    api = json.loads(api_raw)
    files = [row for row in api['files'] if row['key'] == 'package.zip']
    if len(files) != 1 or files[0]['size'] != EXPECTED_SIZE or files[0]['checksum'] != 'md5:' + EXPECTED_MD5:
        raise ValueError('publisher size/checksum mismatch')
    directory = ROOT / 'results/hilewm_archive_MAIN_20261028_v1'
    if directory.exists() or shutil.disk_usage(directory.parent).free < EXPECTED_SIZE + 512 * 1024**2:
        raise ValueError('exclusive new directory and archive+512MiB reserve required')
    directory.mkdir()
    command = ['curl', '--fail', '--location', '--max-redirs', '3', '--proto', '=https',
        '--proto-redir', '=https', '--connect-timeout', '20', '--max-time', '1800',
        '--max-filesize', str(EXPECTED_SIZE), '--retry', '0', '--silent', '--show-error',
        '--output', str(directory / 'archive.part'), '--write-out', '%{json}', URL]
    request = dict(schema='MAIN.hilewm_whole_archive_request.v1', url=URL,
        expected_bytes=EXPECTED_SIZE, expected_publisher_md5=EXPECTED_MD5,
        cached_primary_api=str(api_path), cached_primary_api_sha256=API_SHA,
        transfer_limit_seconds=1800, attempts=1, retry=0, range_request=False,
        unpickled=False, extracted=False, scientific_admission=False, command=command)
    write_new(directory / 'request.json', request)
    print(json.dumps(dict(status='whole_archive_transfer_started', directory=str(directory))), flush=True)
    started = time.monotonic()
    report = dict(schema='MAIN.hilewm_whole_archive_result.v1', status='failed_preserved',
        errors=[], unpickled=False, extracted=False, pairing_verified=False,
        macro_bank_verified=False, scientific_admission=False)
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=1830)
        report['curl_exit_code'] = result.returncode
        report['curl_stderr'] = result.stderr[-8000:]
        try:
            report['transfer'] = json.loads(result.stdout)
        except ValueError:
            report['transfer_stdout'] = result.stdout[-8000:]
        if result.returncode:
            raise ValueError('whole archive curl did not finish')
        archive = directory / 'archive.part'
        if archive.stat().st_size != EXPECTED_SIZE:
            raise ValueError('archive exact byte count mismatch')
        md5, sha = hashlib.md5(), hashlib.sha256()
        with archive.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                md5.update(chunk)
                sha.update(chunk)
        report.update(measured_archive_md5=md5.hexdigest(), measured_archive_sha256=sha.hexdigest())
        if md5.hexdigest() != EXPECTED_MD5:
            raise ValueError('publisher archive MD5 mismatch')
        report['status'] = 'publisher_archive_bytes_verified_only'
    except Exception as exc:
        report['errors'].append(type(exc).__name__ + ': ' + str(exc))
    report['wall_seconds'] = time.monotonic() - started
    archive = directory / 'archive.part'
    report['actual_saved_bytes'] = archive.stat().st_size if archive.exists() else 0
    write_new(directory / 'report.json', report)
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    print(json.dumps(report, sort_keys=True), flush=True)
    return int(report['status'] != 'publisher_archive_bytes_verified_only')


if __name__ == '__main__':
    raise SystemExit(main())
