"""One-shot cell03 host entry; no GPU or native execution on import."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

from experiments.spectral_reachability import e3_cell03_host_cpu_precheck_codex_20261030_v1 as precheck

ROOT = Path('/home/gbk/code/LeWorldModel/le-wm-main')
AUTHORITY = ROOT / 'results/e3_cell03_feedback_prepare_codex_20261030_v1_MAIN01.json'
AUTHORITY_SHA = '1facc17c2bb088c5c2af1ce4f035bc1a0c2742ab3d81ccd9d2830382fedf76e4'
OUTPUT = ROOT / 'results/agent_e3_chunked_identity_host_20261029_v1_CELL03_MAIN01'
CLAIM = ROOT / 'results/e3_cell03_exclusive_host_codex_20261030_v1_MAIN01.claim'
CONFIRM = 'HOST_CUDA_COORDINATED_CELL03_ONCE'
RUNNER_MODULE = 'experiments.spectral_reachability.agent_e3_chunked_identity_host_20261029_v1'
# -c has no project-backed __main__.__file__; the parent wrapper and its
# precheck imports therefore cannot contaminate the runner's loaded census.
CHILD = """import json, sys
from experiments.spectral_reachability import agent_e3_chunked_identity_host_20261029_v1 as runner
result = runner.run(sys.argv[1], sys.argv[2], sys.argv[3], runner.launch.AUTH)
print(json.dumps(result, sort_keys=True), flush=True)
sys.exit(0 if result.get('complete') is True else 1)
"""


def run(confirmation):
    if confirmation != CONFIRM:
        raise ValueError('explicit host GPU coordination confirmation required')
    if os.environ.get('CUDA_VISIBLE_DEVICES') in ('', '-1'):
        raise ValueError('CUDA is explicitly disabled')
    # All read-only checks precede the irreversible claim.
    precheck.check(str(AUTHORITY), AUTHORITY_SHA)
    if os.path.lexists(OUTPUT):
        raise FileExistsError(f'output already exists: {OUTPUT}')
    record = dict(schema='e3_cell03_one_shot_claim_v1', pid=os.getpid(),
                  hostname=socket.gethostname(), authority=str(AUTHORITY),
                  authority_sha256=AUTHORITY_SHA, output=str(OUTPUT),
                  meaning='attempt claimed; not evidence of execution or scientific admission')
    fd = os.open(CLAIM, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(record, stream, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(CLAIM.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    # The original runner creates its output exclusively and performs its own
    # guards. Never import the wrapper or precheck in this fresh interpreter.
    result = subprocess.run([sys.executable, '-B', '-c', CHILD, str(AUTHORITY),
                             AUTHORITY_SHA, str(OUTPUT)], check=False)
    if result.returncode:
        raise SystemExit(result.returncode if result.returncode > 0 else 128 - result.returncode)
    return dict(complete=True, child_exit_code=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--confirm-host-gpu-coordinated', required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.confirm_host_gpu_coordinated), sort_keys=True))


if __name__ == '__main__':
    main()
