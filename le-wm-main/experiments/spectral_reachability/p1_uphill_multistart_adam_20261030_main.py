"""Apply the fixed common multistart Adam planner to the uphill RGB bundle."""
import argparse
from pathlib import Path

from . import codex_p1_adam_cp5_20261030_main as adam
from . import p1_uphill_cp5_bundle_20261030 as bundle

NAME = "p1_uphill_multistart_adam_20261030_main"
BUNDLE_SHA = "376bb8276e1544010ea2767d9c46f141c2665e223154a12b2e6c28786636676a"
PROTOCOL = bundle.ROOT.parent / "docs/D2P_P1_UPHILL_MULTISTART_ADAM_PROTOCOL_20261030_MAIN.md"


def configure():
    base = adam.prior.base
    if base.sha(bundle.OUT) != BUNDLE_SHA:
        raise ValueError("uphill input digest")
    original_bindings = base.bindings

    def bindings():
        result = original_bindings()
        for path in (Path(__file__).resolve(), PROTOCOL, Path(bundle.__file__).resolve()):
            result[str(path)] = base.sha(path)
        return result

    base.BUNDLE, base.BUNDLE_SHA = bundle.OUT, BUNDLE_SHA
    base.bindings = bindings
    adam.NAME = NAME


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    configure()
    print("Plan all fixed uphill prefixes from common random starts on host CUDA.", flush=True)
    adam.run(Path(args.output))


if __name__ == "__main__":
    main()
