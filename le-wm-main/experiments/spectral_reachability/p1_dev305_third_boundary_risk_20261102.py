"""Apply the unchanged risk projector to third-boundary model candidates."""
import argparse
import importlib.util

from . import p1_dev305_contact_third_boundary_20261102 as source
from . import p1_dev305_second_boundary_risk_20261102 as template

base, need, run, ROOT = source.base, source.need, source.run, source.ROOT
NAME = "p1_dev305_third_boundary_risk_20261102"
AUTHORITY = "DEV305_THIRD_BOUNDARY_RISK_CUDA"
PREP_SHA = "08cb1cb1554d515e30eeb74b247dcad716e7d7c0dd558dcf1f26fd405a787ec0"
PLAN_SHA = "bd244019e1f24d4dc960bd4cc51ee99b5bf7cde3d45bbff05f5c72d20b5af342"

# Load an isolated module instance: the archived source chain itself imports the
# canonical template and must not observe altered source-contract constants.
module_spec = importlib.util.spec_from_file_location(
    "experiments.spectral_reachability.p1_dev305_third_boundary_risk_engine", template.__file__)
engine = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(engine)
engine.source = source
engine.NAME = NAME
engine.AUTHORITY = AUTHORITY
engine.PREP_SHA = PREP_SHA
engine.PLAN_SHA = PLAN_SHA
original_protocol = engine.protocol


def protocol():
    result = original_protocol()
    result["wrapper_sha256"] = base.sha(__file__)
    result["scope"] = "post-outcome known-root continuation; fixed uncalibrated terminal heuristic, not safety certification"
    return result


engine.protocol = protocol

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--tag", required=True)
    parser.add_argument("--contract-sha")
    parser.add_argument("--authority")
    args = parser.parse_args()
    need(args.tag.isascii() and args.tag.isalnum(), "tag")
    path = ROOT / "results" / f"{NAME}_PREP_{args.tag}.json"
    if args.command == "prepare":
        print(base.write_new(path, protocol()))
    else:
        with run.coherent.base.first.engine.base.locking.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
            print(engine.execute(path, args.contract_sha, ROOT / "results" / f"{NAME}_RUN_{args.tag}", args.authority))
