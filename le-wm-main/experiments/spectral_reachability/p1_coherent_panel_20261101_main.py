"""Serial fixed four-arm coherent-initializer development comparison."""
import argparse
import json
from pathlib import Path

from . import p1_centered_panel_20261101_main as panel

NAME = "p1_centered_h4_coherent_20261101_launcher"
LAUNCHER_SHA = "e4de84f9505a179baa40cd95a4b83b578a671872e7e72c3e23fe27dca66d96a4"


def run(tag):
    if not tag.startswith("COHERENT") or not tag.isalnum():
        raise ValueError("exclusive coherent tag")
    panel.NAME = NAME
    panel.MODULE = "experiments.spectral_reachability." + NAME
    panel.SOURCE_SHA = LAUNCHER_SHA
    panel.run(tag)
    summary = panel.ROOT / "results" / f"p1_centered_panel_20261101_main_{tag}.json"
    binding = panel.ROOT / "results" / f"p1_coherent_panel_20261101_main_{tag}_binding.json"
    with binding.open("x") as stream:
        json.dump(dict(summary=str(summary), summary_sha256=panel.saved.sha(summary),
                       wrapper_sha256=panel.saved.sha(Path(__file__)),
                       launcher_sha256=LAUNCHER_SHA, P1_admission=False,
                       independent_confirmation=False), stream, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--authorize-host-cuda", required=True, action="store_true")
    args = parser.parse_args()
    run(args.tag)
