"""Read-only saved physical-pair consumer in the isolated historical view."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

from . import p2_b0_isolated_consume_20261102_exclusive as isolation

ROOT = isolation.ROOT
OUTPUT = ROOT / "results/p2_b0_isolated_physical_consume_20261102_exclusive_run01.json"
CONTRACT = ROOT / "results/p2_shared_rgb_physical_bridge_20261102_exclusive_CONTRACT_B0_01.json"
CONTRACT_SHA = "c0397aeed4754a29e93beba303a8af8157c71f525c414847a66f40da0b6dbdc2"
PHYSICAL = ROOT / "results/codex_p2_multistart_joint_physical_20261030_v1_shared_rgb_B0_20261102_01"
REPORT_SHA = "a858d9fbe89ee997461391e6d4ea5171cfed326e010d6dc3ddd3dbceb5bba1ae"
COMPLETION_SHA = "b61f33104bfd34f70f9c17932d536f42a629732869a96dc03a2dd11cd3567fce"
CHILD = (
    "import json; from pathlib import Path; "
    "from experiments.spectral_reachability import p2_shared_rgb_physical_bridge_20261102_exclusive as b; "
    f"c=b.verify_contract(Path({str(CONTRACT)!r}), {CONTRACT_SHA!r}); "
    f"print(json.dumps(b.audit_output(c,Path({str(PHYSICAL)!r}),{REPORT_SHA!r},"
    f"{COMPLETION_SHA!r}),sort_keys=True,allow_nan=False))"
)


def command(historical):
    cmd = isolation.command(historical)
    cmd[-1] = CHILD
    return cmd


def run():
    source = isolation.historical_bytes()
    before, error = isolation.recovery.checked_file(isolation.recovery.TARGET)
    if error:
        raise ValueError("current source unreadable: " + error)
    with tempfile.TemporaryDirectory(prefix="p2_b0_physical_readonly_") as directory:
        historical = Path(directory) / "historical_p0.py"
        with historical.open("xb") as stream:
            stream.write(source)
            stream.flush()
            os.fsync(stream.fileno())
        historical.chmod(0o444)
        try:
            child = subprocess.run(command(historical), stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, timeout=300, check=False)
            code, stdout, stderr, timed_out = (child.returncode, child.stdout.decode("utf-8", "replace"),
                                                child.stderr.decode("utf-8", "replace"), False)
        except subprocess.TimeoutExpired as exc:
            code, stdout, stderr, timed_out = (None, (exc.stdout or b"").decode("utf-8", "replace"),
                                                (exc.stderr or b"").decode("utf-8", "replace"), True)
    after, error = isolation.recovery.checked_file(isolation.recovery.TARGET)
    if error or before != after:
        raise ValueError("working tree target changed during verification")
    result = None
    if code == 0:
        result = json.loads(stdout)
        if result.get("status") != "verified_known_root_physical_not_admitted":
            raise ValueError("unexpected physical consumer result")
    return {
        "schema": "p2_b0_isolated_physical_consume_20261102_exclusive_v1",
        "status": "consumer_returned_success" if code == 0 else "consumer_failed_or_timed_out",
        "return_code": code, "timed_out": timed_out, "stdout": stdout, "stderr": stderr,
        "consumer_result": result, "historical_source_sha256": isolation.digest(source),
        "working_tree_sha256_before_after": before, "contract_sha256": CONTRACT_SHA,
        "physical_report_sha256": REPORT_SHA, "physical_completion_sha256": COMPLETION_SHA,
        "readonly_root": True, "minimal_dev": True, "cuda_visible_devices": "",
        "network_namespace_isolated": False, "new_physical_execution": False,
        "new_model_inference": False, "independent_confirmation": False,
        "P1_admission": False, "P2_admission": False,
        "limitations": ["Saved-data checks reuse producer-related validation code; no fresh physical replay.",
                        "Known root and repeated commands are not independent confirmation or equal-budget proof."],
    }


def main():
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise FileExistsError(OUTPUT)
    value = run()
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with OUTPUT.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    print(f"created {OUTPUT} sha256={isolation.digest(raw)} status={value['status']}")


if __name__ == "__main__":
    main()
