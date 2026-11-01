"""Run the preregistered centered four-arm panel serially in fresh processes."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from . import p1_saved_h4_parfit_20261101 as saved

ROOT = Path(__file__).resolve().parents[2]
MODULE = "experiments.spectral_reachability.p1_centered_h4_parfit_20261101_launcher"
NAME = "p1_centered_h4_parfit_20261101_launcher"
ARMS = ("structured", "native", "random", "immediate_progress")
SOURCE_SHA = "73bc21ae08adf0382f2e7e17c9f66a8526572a1178125ab64b5f5f646693b275"


def run(tag):
    if not tag.isalnum():
        raise ValueError("simple output tag")
    path = ROOT / "experiments/spectral_reachability" / (NAME + ".py")
    if hashlib.sha256(path.read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError("frozen launcher")
    summary_path = ROOT / "results" / f"p1_centered_panel_20261101_main_{tag}.json"
    if summary_path.exists():
        raise ValueError("exclusive summary")
    results = []
    for arm in ARMS:
        prefix = ROOT / "results" / f"{NAME}_{arm}"
        prepared = Path(str(prefix) + f"_PREP_{tag}.json")
        output = Path(str(prefix) + f"_RUN_{tag}")
        log = Path(str(prefix) + f"_RUN_{tag}.log")
        if prepared.exists() or output.exists() or log.exists():
            raise ValueError("exclusive arm artifacts")
        print(json.dumps(dict(arm=arm, status="preparing")), flush=True)
        command = [sys.executable, "-m", MODULE, "--arm", arm]
        response = subprocess.run(command + ["prepare", "--output", str(prepared)],
                                  cwd=ROOT, check=True, text=True, capture_output=True)
        receipt = json.loads(response.stdout.strip().splitlines()[-1])
        with log.open("x") as stream:
            stream.write(response.stdout + response.stderr)
            stream.flush()
            print(json.dumps(dict(arm=arm, status="running_host_cuda", log=str(log))), flush=True)
            subprocess.run(command + ["run", "--prepared", str(prepared),
                "--sha256", receipt["sha256"], "--output", str(output),
                "--authorization", "MAIN_AUTHORIZES_COMMON_SUFFIX_RGB_CUDA_AND_PHYSICS",
                "--authorize-host-cuda"], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        report_path = output / "report.json"
        report = json.loads(report_path.read_bytes())
        row = report["rows"][0]
        digest = saved.sha(report_path)
        verification = (saved.check(output, digest) if row["full_horizon"] and row["control_returned"] == 20
                        else saved.status_only(output, digest))
        entry = dict(arm=arm, report=str(report_path), report_sha256=digest,
                     status=row["status"], steps=row["control_returned"],
                     native_success=row["native_success"], first_error=row["first_error"],
                     verification=verification)
        with Path(str(prefix) + f"_CHECK_{tag}.json").open("x") as stream:
            json.dump(entry, stream, indent=2, allow_nan=False)
        results.append(entry)
        print(json.dumps({k: entry[k] for k in ("arm", "status", "steps", "native_success", "first_error")}), flush=True)
        if row["first_error"]:
            raise RuntimeError("infrastructure error; stop and diagnose before further arms")
    with summary_path.open("x") as stream:
        json.dump(dict(rows=results, P1_admission=False, independent_confirmation=False,
                       launcher_sha256=SOURCE_SHA, driver_sha256=saved.sha(Path(__file__))),
                  stream, indent=2, allow_nan=False)
    print(json.dumps(dict(status="panel_complete", summary=str(summary_path))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--authorize-host-cuda", required=True, action="store_true")
    args = parser.parse_args()
    run(args.tag)
