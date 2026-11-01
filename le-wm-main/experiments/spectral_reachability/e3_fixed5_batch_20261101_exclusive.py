"""Sealed, serial E3 fixed-five launcher. Importing this module never runs a model."""
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import stat
import sys

from . import e3_uncovered_fixed_inputs_20261101_exclusive as fixed
from . import agent_e3_startup_closure_20261029_v1 as startup

ROOT = fixed.ROOT
RESULTS = ROOT / "results"
LOCK = RESULTS / "e3_p2_host_cuda_cooperative.lock"
CELLS = (5, 6, 7, 14, 15)
SHAS = (
    "23bedf07b17e40bf03d10e276de528f126d915317b76b111d0cd41706f78340b",
    "0add0b56b7e3f8f09864e36b2447b8df26ff776a296a4fee525f1803c714cae1",
    "5d72b7a0f95c407bccf6edc2e523de761f22f07edd8332b6372fd1cf31e70fff",
    "7ad242cfcaa49150091e4aeaa66a64111c3d8dbe42d78750fbde80f2dcc68da0",
    "304276bbf13684d293c84215abdaf29f5a332e64ab801da5ecfec1c6b4e84aaf",
)
CONFIRM = "E3_FIXED5_REAL_MODEL_PHYSICS"
SCHEMA = "e3_fixed5_batch_20261101_exclusive.v1"
STOPS = frozenset(("native_stopped", "native_stopped_warmup", "native_stopped_prefix",
                   "native_initial_success_control5"))
CHILD = """import sys
from experiments.spectral_reachability import agent_e3_chunked_identity_host_20261029_v1 as h
r=h.run(sys.argv[1],sys.argv[2],sys.argv[3],h.launch.AUTH)
sys.exit(0 if r.get('complete') is True else 1)
"""


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_new(path, value):
    data = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    dfd = os.open(Path(path).parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)
    return hashlib.sha256(data).hexdigest()


def batch_path(path):
    p = Path(path)
    if not p.is_absolute() or p.parent != RESULTS or p.resolve() != p or not p.name.startswith("e3_fixed5_batch_20261101_exclusive_"):
        raise ValueError("batch must be a canonical, exclusively named results directory")
    return p


def seal(path):
    """Create CPU-only startup closures and freeze all five inputs before execution."""
    p = batch_path(path)
    p.mkdir(mode=0o700)
    rows = []
    for cell, digest in zip(CELLS, SHAS):
        source = RESULTS / f"e3_uncovered_fixed_inputs_20261101_exclusive_cell{cell:02d}_RUN01.json"
        fixed.verify(cell, str(source), digest)
        closure = p / f"cell{cell:02d}_closure.json"
        authority = p / f"cell{cell:02d}_authority.json"
        c = startup.collect(str(source), digest, str(closure))
        a = startup.prepare(str(source), digest, str(closure), c["sha256"], str(authority))
        rows.append(dict(cell=cell, source=str(source), source_sha256=digest,
                         closure=str(closure), closure_sha256=c["sha256"],
                         authority=str(authority), authority_sha256=a["sha256"]))
    manifest = dict(schema=SCHEMA, cells=rows, launcher_sha256=sha(__file__),
                    python=str(Path(sys.executable).resolve()), scientific_admission=False,
                    execution="serial_stop_on_failure_or_native_stop", cooperative_lock=str(LOCK))
    digest = write_new(p / "seal.json", manifest)
    return dict(seal=str(p / "seal.json"), sha256=digest, cuda_executed=False, physics_executed=False)


def load_seal(p, expected):
    manifest_path = p / "seal.json"
    if sha(manifest_path) != expected:
        raise ValueError("seal hash mismatch")
    m = json.loads(manifest_path.read_bytes())
    if (m["schema"] != SCHEMA or m["launcher_sha256"] != sha(__file__)
            or m["python"] != str(Path(sys.executable).resolve())
            or m["scientific_admission"] is not False
            or m["cooperative_lock"] != str(LOCK)
            or [x["cell"] for x in m["cells"]] != list(CELLS)):
        raise ValueError("seal identity mismatch")
    for row, cell, digest in zip(m["cells"], CELLS, SHAS):
        expected_source = RESULTS / f"e3_uncovered_fixed_inputs_20261101_exclusive_cell{cell:02d}_RUN01.json"
        if (row["source"] != str(expected_source) or row["source_sha256"] != digest
                or row["closure"] != str(p / f"cell{cell:02d}_closure.json")
                or row["authority"] != str(p / f"cell{cell:02d}_authority.json")):
            raise ValueError("sealed path or source mismatch")
        for key in ("source", "closure", "authority"):
            if sha(row[key]) != row[key + "_sha256"]:
                raise ValueError("sealed input drift: " + key)
    return m


def attempts(p, cell):
    paths = sorted(p.glob(f"cell{cell:02d}_attempt[0-9][0-9][0-9]"))
    if any(x.is_symlink() or not x.is_dir() or x.name != f"cell{cell:02d}_attempt{i:03d}"
           for i, x in enumerate(paths, 1)):
        raise ValueError("attempt sequence or directory invalid")
    return paths


def latest(p, cell):
    paths = attempts(p, cell)
    if not paths:
        return None
    result = paths[-1] / "result.json"
    if not result.is_file():
        return {"status": "interrupted_or_unrecorded", "attempt": paths[-1].name}
    data = json.loads(result.read_bytes())
    if data.get("cell") != cell or data.get("attempt") != paths[-1].name:
        raise ValueError("attempt result identity")
    return data


def verify_saved(p, row, prior, seal_sha):
    attempt = p / prior["attempt"]
    intent = json.loads((attempt / "intent.json").read_bytes())
    if intent != dict(cell=row["cell"], attempt=attempt.name, seal_sha256=seal_sha,
                      authority_sha256=row["authority_sha256"]):
        raise ValueError("attempt intent drift")
    native = attempt / "native"
    for key, name in (("report_sha256", "report.json"), ("completion_sha256", "completion.json")):
        if sha(native / name) != prior[key]:
            raise ValueError("saved result drift")
    inspected = startup.host.inspect(str(native), prior["report_sha256"], prior["completion_sha256"],
                                     row["authority"], row["authority_sha256"])
    if (inspected["base"]["status"] != prior["status"]
            or inspected["binding_chunks_complete"] != prior["binding_chunks_complete"]
            or inspected["full_saved_strict"] != prior["full_saved_strict"]):
        raise ValueError("saved result mismatch")


@contextlib.contextmanager
def gpu_lock():
    fd = os.open(LOCK, os.O_RDWR | os.O_NOFOLLOW)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
            raise ValueError("invalid cooperative lock")
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current = os.stat(LOCK, follow_symlinks=False)
        if (st.st_dev, st.st_ino) != (current.st_dev, current.st_ino):
            raise ValueError("cooperative lock replaced")
        yield fd
    finally:
        os.close(fd)


def execute_one(p, row, seal_sha, lock_fd):
    cell = row["cell"]
    existing = attempts(p, cell)
    if len(existing) >= 999:
        raise ValueError("attempt limit")
    attempt = p / f"cell{cell:02d}_attempt{len(existing)+1:03d}"
    attempt.mkdir(mode=0o700)
    write_new(attempt / "intent.json", dict(cell=cell, attempt=attempt.name, seal_sha256=seal_sha,
                                             authority_sha256=row["authority_sha256"]))
    result = dict(cell=cell, attempt=attempt.name, scientific_admission=False)
    try:
        startup.precheck(row["source"], row["source_sha256"], row["closure"], row["closure_sha256"],
                         row["authority"], row["authority_sha256"])
        load_seal(p, seal_sha)
        write_new(attempt / "precheck.json", {"passed": True, "cuda_executed": False, "physics_executed": False})
        write_new(attempt / "child_intent.json", {"physical_execution_may_have_started": True})
        with (attempt / "child.log").open("xb") as log:
            proc = subprocess.run([sys.executable, "-B", "-c", CHILD, row["authority"],
                                   row["authority_sha256"], str(attempt / "native")],
                                  cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=1800, check=False,
                                  pass_fds=(lock_fd,))
        result["returncode"] = proc.returncode
        if proc.returncode != 0:
            raise RuntimeError("native child failed")
        native = attempt / "native"
        report_sha = sha(native / "report.json")
        completion_sha = sha(native / "completion.json")
        inspected = startup.host.inspect(str(native), report_sha, completion_sha,
                                         row["authority"], row["authority_sha256"])
        status = inspected["base"]["status"]
        if status != "completed20" and status not in STOPS:
            raise ValueError("unexpected saved status: " + str(status))
        result.update(status=status, report_sha256=report_sha, completion_sha256=completion_sha,
                      binding_chunks_complete=inspected["binding_chunks_complete"],
                      full_saved_strict=inspected["full_saved_strict"])
    except BaseException as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc)[:1000])
        write_new(attempt / "result.json", result)
        raise
    write_new(attempt / "result.json", result)
    return result


def run(path, seal_sha, confirm, retry_cell=None, continue_after_stop=False):
    if confirm != CONFIRM or Path.cwd().resolve() != ROOT or os.environ.get("CUDA_VISIBLE_DEVICES") in ("", "-1"):
        raise ValueError("explicit confirmation, project cwd and visible CUDA required")
    p = batch_path(path)
    with gpu_lock() as lock_fd:
        manifest = load_seal(p, seal_sha)
        if retry_cell is not None and retry_cell not in CELLS:
            raise ValueError("retry outside fixed scope")
        if retry_cell is not None:
            candidate = latest(p, retry_cell)
            if not candidate or candidate["status"] == "completed20" or candidate["status"] in STOPS:
                raise ValueError("retry cell did not identify an unresolved attempt")
        retried = False
        for row in manifest["cells"]:
            cell = row["cell"]
            prev = latest(p, cell)
            if prev and prev["status"] == "completed20":
                verify_saved(p, row, prev, seal_sha)
                continue
            if prev and prev["status"] in STOPS:
                verify_saved(p, row, prev, seal_sha)
                if not continue_after_stop:
                    return {"status": "paused_native_stop", "cell": cell}
                continue
            if prev:
                if retry_cell != cell:
                    return {"status": "paused_unresolved_attempt", "cell": cell, "previous": prev}
                retried = True
            result = execute_one(p, row, seal_sha, lock_fd)
            if result["status"] in STOPS:
                return {"status": "paused_native_stop", "cell": cell, "result": result}
        if retry_cell is not None and not retried:
            raise ValueError("retry cell did not identify an unresolved attempt")
        return {"status": "all_cells_recorded", "scientific_admission": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("seal")
    s.add_argument("--batch", required=True)
    r = sub.add_parser("run")
    r.add_argument("--batch", required=True)
    r.add_argument("--seal-sha256", required=True)
    r.add_argument("--confirm", required=True)
    r.add_argument("--retry-cell", type=int, choices=CELLS)
    r.add_argument("--continue-after-stop", action="store_true")
    a = parser.parse_args()
    value = seal(a.batch) if a.command == "seal" else run(
        a.batch, a.seal_sha256, a.confirm, a.retry_cell, a.continue_after_stop)
    print(json.dumps(value, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
