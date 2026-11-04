"""Mount one decision input and its exclusive output for an isolated worker."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/pc_jepa_v13_contact_feedback_dev_20261104_codex"


def main():
    if len(sys.argv) != 2:
        raise ValueError("decision directory required")
    base = Path(sys.argv[1]).resolve()
    if base.parent != RUN or base.name not in {f"{a}_{s}" for a in ("response", "cem") for s in range(3)}:
        raise ValueError("unexpected decision directory")
    inputs = [ROOT / "experiments", ROOT / ".venv", ROOT / "external/jepa-wms",
              ROOT / "external/dinov2_official_7764ea0f912e53c92e82eb78a2a1631e92725fc8",
              ROOT / ".cache/dinov2/dinov2-small", ROOT / "data/jepa_wm_checkpoints/jepa_wm_pusht.pth.tar",
              base / "input"]
    lock, output = ROOT / "results/e3_p2_host_cuda_cooperative.lock", base / "output"
    if any(not path.exists() for path in [*inputs, lock, output]):
        raise FileNotFoundError("missing isolated input")
    cmd = ["bwrap", "--die-with-parent", "--unshare-pid", "--unshare-net", "--ro-bind", "/", "/",
           "--tmpfs", "/home/gbk", "--tmpfs", "/tmp", "--proc", "/proc", "--dev-bind", "/dev", "/dev"]
    dirs = set()
    for path in [*inputs, lock, output]:
        parent = path.parent
        while str(parent).startswith("/home/gbk/"):
            dirs.add(parent)
            parent = parent.parent
    for path in sorted(dirs, key=lambda p: (len(p.parts), str(p))):
        cmd.extend(["--dir", str(path)])
    for path in inputs:
        cmd.extend(["--ro-bind", str(path), str(path)])
    for path in (lock, output):
        cmd.extend(["--bind", str(path), str(path)])
    cmd.extend(["--chdir", str(ROOT), "--clearenv", "--setenv", "PATH", "/usr/bin:/bin",
                "--setenv", "HOME", "/tmp", "--setenv", "PYTHONPATH", str(ROOT),
                "--setenv", "PYTHONDONTWRITEBYTECODE", "1", "--setenv", "CUDA_VISIBLE_DEVICES", "0",
                str(ROOT / ".venv/bin/python"), "-m", "experiments.pc_jepa_v13_contact_feedback_worker_20261104_codex",
                str(base)])
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
