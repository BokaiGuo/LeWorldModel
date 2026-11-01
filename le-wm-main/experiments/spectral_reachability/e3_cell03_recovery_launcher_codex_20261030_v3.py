"""One-shot recovery candidate. Preparation is CPU-only; run is explicitly gated."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import stat
import subprocess
import sys

from experiments.spectral_reachability import e3_cell03_recovery_manifest_codex_20261030_v2 as draft

ROOT = draft.ROOT
MANIFEST = ROOT / 'results/e3_cell03_recovery01_launcher_manifest_codex_20261030_v3.json'
CLAIM = ROOT / 'results/e3_cell03_recovery01_launcher_codex_20261030_v3.claim'
JOURNAL = ROOT / 'results/e3_cell03_recovery01_launcher_codex_20261030_v3.jsonl'
OUTPUT = ROOT / 'results/agent_e3_chunked_identity_host_20261029_v1_CELL03_RECOVERY01_V3'
LOCK = draft.LOCK
CONFIRM = 'CELL03_RECOVERY01_P2_COORDINATED'
ENV_KEYS = ('HOME', 'PATH', 'DISPLAY', 'XAUTHORITY', 'XDG_RUNTIME_DIR',
            'LD_LIBRARY_PATH', 'CUDA_VISIBLE_DEVICES')
CHILD = draft.prior.parent.CHILD


def child_env():
    result = {key: os.environ[key] for key in ENV_KEYS if key in os.environ}
    result.update(PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1',
                  HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    return result


def prepare_body():
    candidate = draft.build()
    return dict(schema='e3_cell03_recovery_launcher_manifest_v3',
                attempt='CELL03_RECOVERY01_V3', candidate=candidate,
                launcher_sha256=draft.prior.file_sha(Path(__file__).resolve()),
                child_utf8_sha256=draft.digest(CHILD.encode('utf-8')),
                child_environment=child_env(), cwd=str(ROOT),
                claim=str(CLAIM), journal=str(JOURNAL), output=str(OUTPUT),
                lock=str(LOCK), cooperative_only=True,
                external_native_and_transient_imports_proved=False)


def verify(manifest_sha):
    data = MANIFEST.read_bytes()
    if hashlib.sha256(data).hexdigest() != manifest_sha:
        raise ValueError('manifest SHA mismatch')
    saved = json.loads(data)
    # draft includes a full process-environment digest that is relevant to its
    # CPU snapshot, not the explicitly restricted child runtime environment.
    current = prepare_body()
    saved_candidate = saved['candidate']
    current_candidate = current['candidate']
    saved_candidate.pop('environment_sha256')
    current_candidate.pop('environment_sha256')
    if saved != current:
        raise ValueError('manifest/source/child environment drift')
    return manifest_sha


def append_event(stream, stage, **fields):
    stream.write(draft.raw(dict(stage=stage, **fields)))
    stream.flush()
    os.fsync(stream.fileno())


def run(manifest_sha, confirm):
    if confirm != CONFIRM:
        raise ValueError('explicit P2 coordinated recovery required')
    if os.environ.get('CUDA_VISIBLE_DEVICES') in ('', '-1'):
        raise ValueError('CUDA disabled')
    if Path.cwd().resolve() != ROOT:
        raise ValueError('wrong cwd')
    verify(manifest_sha)
    if any(os.path.lexists(p) for p in (CLAIM, JOURNAL, OUTPUT)):
        raise FileExistsError('attempt path already exists')
    fd = os.open(LOCK, os.O_RDWR | os.O_NOFOLLOW)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
            raise ValueError('invalid lock inode')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current = os.stat(LOCK, follow_symlinks=False)
        if (st.st_dev, st.st_ino) != (current.st_dev, current.st_ino):
            raise ValueError('lock replaced')
        # Revalidate before claiming, while holding the cooperative lock.
        verify(manifest_sha)
        if any(os.path.lexists(p) for p in (CLAIM, JOURNAL, OUTPUT)):
            raise FileExistsError('attempt path already exists')
        claim = dict(schema='e3_cell03_recovery_claim_v3', manifest_sha256=manifest_sha,
                     predecessor=draft.prior.EXPECTED, output=str(OUTPUT),
                     pid=os.getpid(), hostname=socket.gethostname(),
                     lock_device=st.st_dev, lock_inode=st.st_ino,
                     cooperation_only=True)
        draft.write_exclusive(CLAIM, claim)
        jfd = os.open(JOURNAL, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(jfd, 'wb') as journal:
            append_event(journal, 'claimed', manifest_sha256=manifest_sha)
            append_event(journal, 'child_intent')
            try:
                result = subprocess.run(
                    [sys.executable, '-B', '-c', CHILD, str(draft.prior.parent.AUTHORITY),
                     draft.prior.parent.AUTHORITY_SHA, str(OUTPUT)],
                    cwd=ROOT, env=child_env(), pass_fds=(fd,), check=False)
            except BaseException as exc:
                append_event(journal, 'child_start_failure', error_type=type(exc).__name__,
                             message=str(exc))
                raise
            append_event(journal, 'child_exit', returncode=result.returncode)
            if result.returncode:
                raise RuntimeError('child exited nonzero: ' + str(result.returncode))
            try:
                report = OUTPUT / 'report.json'
                completion = OUTPUT / 'completion.json'
                report_sha = draft.prior.file_sha(report)
                completion_sha = draft.prior.file_sha(completion)
                inspected = draft.precheck.host.inspect(str(OUTPUT), report_sha, completion_sha,
                                                        str(draft.prior.parent.AUTHORITY),
                                                        draft.prior.parent.AUTHORITY_SHA)
                if inspected['base']['status'] == 'partial':
                    raise ValueError('consumer reports partial')
            except BaseException as exc:
                append_event(journal, 'saved_inspect_failure', error_type=type(exc).__name__,
                             message=str(exc))
                raise
            append_event(journal, 'saved_inspect_passed', report_sha256=report_sha,
                         completion_sha256=completion_sha, full_saved_strict=False)
            return dict(complete=True, report_sha256=report_sha, completion_sha256=completion_sha,
                        scientific_admission=False)
    finally:
        # Close, do not explicitly unlock: inherited child descriptors retain
        # the lock if this parent dies while the child is still running.
        os.close(fd)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare')
    r = sub.add_parser('run')
    r.add_argument('--manifest-sha256', required=True)
    r.add_argument('--confirm-p2-coordination', required=True)
    a = p.parse_args()
    if a.command == 'prepare':
        if Path.cwd().resolve() != ROOT:
            raise ValueError('wrong cwd')
        sha = draft.write_exclusive(MANIFEST, prepare_body())
        print(json.dumps(dict(manifest=str(MANIFEST), sha256=sha, approved=False)))
    else:
        print(json.dumps(run(a.manifest_sha256, a.confirm_p2_coordination)))


if __name__ == '__main__':
    main()
