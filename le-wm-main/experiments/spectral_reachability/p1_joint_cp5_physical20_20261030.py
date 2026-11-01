"""Use the fixed 24-row physical runner for the 20-step CUDA plan."""
import sys
from pathlib import Path

from . import p1_joint_cp5_physical_20261030 as physical
from . import p1_joint_cp5_cuda20_20261030 as cuda20

NAME = "p1_joint_cp5_physical20_20261030"
original_build = physical.build


def configure():
    cuda20.configure()
    physical.engine.NAME = NAME


def build():
    value = original_build()
    for path in (Path(__file__).resolve(), Path(cuda20.__file__).resolve(), cuda20.PROTOCOL):
        value["bindings"][str(path)] = physical.sha(path)
    return value


def main():
    physical.NAME = NAME
    physical.PLAN_DIR = physical.ROOT / "results/p1_joint_cp5_cuda20_20261030_RUN01"
    physical.MANIFEST_SHA = "60a3bb54f121ab31527d14d2df52525914be59d374235cc5f3502a312cea5b93"
    physical.REPORT_SHA = "69d7920a5909e7f5c3cb93f7f25f3de9b95122b9d006691b2b5a922ebbae3ccf"
    physical.configure, physical.build = configure, build
    return physical.main()


if __name__ == "__main__":
    sys.exit(main())
