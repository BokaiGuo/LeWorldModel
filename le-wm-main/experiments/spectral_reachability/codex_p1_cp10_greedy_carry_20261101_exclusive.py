"""Known cp10 spectral8 greedy-prefix plus real-RGB own carry replanning."""
import argparse
import json
from pathlib import Path

from . import codex_p1_common_structured_cp10_host_20261101_exclusive as cp10
from . import codex_p1_greedy_structured_host_20261101_exclusive as greedy

NAME = "codex_p1_cp10_greedy_carry_20261101_exclusive"


def configure():
    cp10.previous = greedy
    cp10.configure()
    host = cp10.host
    old_build, old_protocol = host.build, host.protocol

    def build():
        value = old_build()
        path = Path(__file__).resolve()
        value["sources"][str(path)] = host.sha(path)
        return value

    def protocol():
        value = old_protocol()
        value.update(experiment=NAME, retrospective_known_root=True,
                     P1_admission=False, independent_confirmation=False)
        return value

    host.build, host.protocol, host.NAME = build, protocol, NAME
    return host


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    if not args.tag.isalnum():
        raise ValueError("alphanumeric tag required")
    host = configure()
    prefix = host.ROOT / "results" / NAME
    prepared = Path(str(prefix) + "_PREP_" + args.tag + ".json")
    output = Path(str(prefix) + "_RUN_" + args.tag)
    if prepared.exists() or output.exists():
        raise ValueError("exclusive outputs")
    seal = host.prepare(str(prepared))
    print(json.dumps({"prepared": seal}), flush=True)
    with cp10.locking.gpu_lock(host.ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        result = host.run(str(prepared), seal["sha256"], str(output), host.AUTH)
    print(json.dumps({"status": result["status"], "output": str(output)}), flush=True)
    return 0 if result["status"] == "recorded_known_root_not_admitted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
