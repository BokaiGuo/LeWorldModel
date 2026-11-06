"""Run the audited v25 paired harness with a prefix-preserving seed ablation."""
import itertools
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace

from experiments import pc_v25_random_refinement_e0_dev_20261104_codex as previous
from experiments.spectral_reachability import pc_v25_prefix_random_refinement_dev as candidate

OUT = previous.ROOT / "results/pc_v25_prefix_random_refinement_e0_dev_20261104_codex"
PROTOCOL = previous.ROOT.parent / "docs/PC_V25_PREFIX_RANDOM_REFINEMENT_E0_DEV_PROTOCOL_20261104_CODEX.md"


def tasks():
    return [{"start": [-.5, y, theta], "goal": [.5, gy]}
            for y, theta, gy in itertools.product((-.135, .135), (-1.13, 1.13), (-.07, .07))]


def isolated(function):
    globals_copy = dict(function.__globals__)
    sealed_harness = SimpleNamespace(
        capture=lambda files: previous.seal.capture([*files, Path(previous.__file__)]),
        verify=previous.seal.verify)
    globals_copy.update({"candidate": candidate, "OUT": OUT, "PROTOCOL": PROTOCOL,
                         "tasks": tasks, "__file__": __file__, "seal": sealed_harness})
    return FunctionType(function.__code__, globals_copy, function.__name__,
                        function.__defaults__, function.__closure__)


def run():
    if any(task in previous.tasks() for task in tasks()):
        raise ValueError("overlap with prior random-refinement panel")
    isolated(previous.run)()


def audit():
    isolated(previous.audit)()


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("run", "audit"):
        raise SystemExit("choose run or audit")
    {"run": run, "audit": audit}[sys.argv[1]]()
