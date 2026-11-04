"""Launch the isolated development worker with an explicit filesystem view."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/pc_jepa_v7_isolated_input_dev_20261104_codex"


def main():
    if sys.argv[1:] not in ([], ["--probe"]):
        raise SystemExit("usage: launcher [--probe]")
    sources = [
        ROOT / "experiments",
        ROOT / ".venv",
        ROOT / "external/jepa-wms",
        ROOT / "external/dinov2_official_7764ea0f912e53c92e82eb78a2a1631e92725fc8",
        ROOT / ".cache/dinov2/dinov2-small",
        ROOT / "data/jepa_wm_checkpoints/jepa_wm_pusht.pth.tar",
        BASE / "input",
    ]
    lock = ROOT / "results/e3_p2_host_cuda_cooperative.lock"
    output = BASE / "output"
    for path in [*sources, lock, output]:
        if not path.exists():
            raise FileNotFoundError(path)
    cmd = ["bwrap", "--die-with-parent", "--unshare-pid", "--unshare-net",
           "--ro-bind", "/", "/", "--tmpfs", "/home/gbk", "--tmpfs", "/tmp",
           "--proc", "/proc", "--dev-bind", "/dev", "/dev"]
    directories = set()
    for path in [*sources, lock, output]:
        parent = path.parent
        while str(parent).startswith("/home/gbk/"):
            directories.add(parent)
            parent = parent.parent
    for directory in sorted(directories, key=lambda p: (len(p.parts), str(p))):
        cmd.extend(["--dir", str(directory)])
    for path in sources:
        cmd.extend(["--ro-bind", str(path), str(path)])
    for path in (lock, output):
        cmd.extend(["--bind", str(path), str(path)])
    cmd.extend(["--chdir", str(ROOT), "--clearenv", "--setenv", "PATH", "/usr/bin:/bin",
                "--setenv", "HOME", "/tmp", "--setenv", "PYTHONPATH", str(ROOT),
                "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
                "--setenv", "CUDA_VISIBLE_DEVICES", "0"])
    if sys.argv[1:] == ["--probe"]:
        code = ("from pathlib import Path; import numpy; "
                f"p=Path({str(ROOT / 'data/jepa_wm_osf')!r}); "
                f"q=Path({str(BASE / 'input/context.npz')!r}); "
                "assert not p.exists() and q.is_file(); print('isolated CPU probe passed')")
        cmd.extend([str(ROOT / ".venv/bin/python"), "-c", code])
    else:
        cmd.extend([str(ROOT / ".venv/bin/python"), "-m",
                    "experiments.pc_jepa_v7_isolated_worker_20261104_codex"])
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
