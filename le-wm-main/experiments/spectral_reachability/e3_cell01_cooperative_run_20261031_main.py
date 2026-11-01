"""Single-host, cooperative cell01 exploratory feedback run; not a global fence."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time

ROOT = Path("/home/gbk/code/LeWorldModel/le-wm-main")
RESULTS = ROOT / "results"
SOURCE = RESULTS / "agent_e3_cell01_feedback_prepare_20261029_v1_MAIN01.json"
AUTHORITY = RESULTS / "e3_cell01_startup_authority_20261031_main_RUN01.json"
CLOSURE = RESULTS / "e3_cell01_startup_closure_20261031_main_RUN01.json"
AUTH_SHA = "0e525436e311dc8ffcb4cf7ec04fd0ffcbccafedc3ea52391fe9e888bb5105f8"
CLOSURE_SHA = "337db3b7f8abe5336ecd74739f2ac8e88136e1155196c6dea3940ef043dafc4a"
OUTPUT = RESULTS / "e3_cell01_cooperative_native_20261031_main_RUN01"
CLAIM = RESULTS / "e3_cell01_cooperative_20261031_main_RUN01.claim"
JOURNAL = RESULTS / "e3_cell01_cooperative_20261031_main_RUN01.jsonl"
LOCK = RESULTS / "e3_p2_host_cuda_cooperative.lock"
CONFIRM = "CELL01_COOPERATIVE_EXPLORATORY_ONCE"
CHILD = """import sys
from experiments.spectral_reachability import agent_e3_chunked_identity_host_20261029_v1 as r
x = r.run(sys.argv[1], sys.argv[2], sys.argv[3], r.launch.AUTH)
sys.exit(0 if x.get('complete') is True else 1)
"""


def precheck_command():
    return [sys.executable, "-B", "-m", "experiments.spectral_reachability.agent_e3_startup_closure_20261029_v1",
            "precheck", "--source", str(SOURCE), "--source-sha256", AUTH_SHA,
            "--closure", str(CLOSURE), "--closure-sha256", CLOSURE_SHA,
            "--authority", str(AUTHORITY), "--authority-sha256", AUTH_SHA]


def exclusive_write(path, value):
    data = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as file:
        file.write(data)
        file.flush()
        os.fsync(file.fileno())
    dfd = os.open(RESULTS, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def run(confirm):
    if confirm != CONFIRM or Path.cwd().resolve() != ROOT or os.environ.get("CUDA_VISIBLE_DEVICES") in ("", "-1"):
        raise ValueError("confirmation, cwd or CUDA")
    if any(os.path.lexists(p) for p in (OUTPUT, CLAIM, JOURNAL)):
        raise FileExistsError("attempt already exists")
    fd = os.open(LOCK, os.O_RDWR | os.O_NOFOLLOW)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
            raise ValueError("lock inode")
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current = os.stat(LOCK, follow_symlinks=False)
        if (current.st_dev, current.st_ino) != (st.st_dev, st.st_ino):
            raise ValueError("lock replaced")
        subprocess.run(precheck_command(), cwd=ROOT, check=True, timeout=900)
        if any(os.path.lexists(p) for p in (OUTPUT, CLAIM, JOURNAL)):
            raise FileExistsError("attempt appeared during precheck")
        exclusive_write(CLAIM, {"cell": 1, "authority_sha256": AUTH_SHA, "closure_sha256": CLOSURE_SHA,
                                "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                "lock_device": st.st_dev, "lock_inode": st.st_ino,
                                "cooperative_only": True, "global_fence": False})
        jfd = os.open(JOURNAL, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(jfd, "w") as journal:
            def record(stage, **fields):
                journal.write(json.dumps({"stage": stage, **fields}, sort_keys=True) + "\n")
                journal.flush()
                os.fsync(journal.fileno())
            record("child_intent", cell=1, scientific_admission=False)
            try:
                proc = subprocess.run([sys.executable, "-B", "-c", CHILD, str(AUTHORITY), AUTH_SHA, str(OUTPUT)],
                                      cwd=ROOT, pass_fds=(fd,), timeout=1800, check=False)
                record("child_exit", returncode=proc.returncode)
                if proc.returncode != 0:
                    raise RuntimeError("native child failed")
            except BaseException as exc:
                record("failure", type=type(exc).__name__, message=str(exc)[:1000])
                raise
        return {"cell": 1, "child_returncode": 0, "saved_consumer_required": True,
                "scientific_admission": False, "global_fence": False}
    finally:
        os.close(fd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm", required=True)
    print(json.dumps(run(parser.parse_args().confirm)), flush=True)
