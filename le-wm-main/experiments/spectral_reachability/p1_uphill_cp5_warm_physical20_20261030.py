"""Execute the fixed 24 warm-initialized cp5 suffix plans."""
from pathlib import Path

from . import p1_uphill_cp5_physical20_20261030 as physical
from . import p1_uphill_cp5_warm_cuda20_20261030 as warm

NAME = "p1_uphill_cp5_warm_physical20_20261030"
original_build = physical.build


def configure():
    warm.configure()
    physical.engine.NAME = NAME


def build():
    result = original_build()
    for path in (Path(__file__).resolve(), Path(warm.__file__).resolve(), warm.PROTOCOL):
        digest = physical.sha(path)
        physical.need(str(path) not in result["bindings"] or result["bindings"][str(path)] == digest,
                      "warm source collision")
        result["bindings"][str(path)] = digest
    return result


def main():
    physical.NAME = NAME
    physical.PLAN_DIR = physical.ROOT / "results/p1_uphill_cp5_warm_cuda20_20261030_RUN01"
    physical.MANIFEST_SHA = "b9fd1ba1b6233132b2ad3ba9b8013fa5f88faf598ee81dba53b20d9ece0f6bad"
    physical.REPORT_SHA = "0334d3485c6631b357b7aada1855767b221800d684955d0b932cd7d4c7c00688"
    physical.configure, physical.build = configure, build
    return physical.main()


if __name__ == "__main__":
    raise SystemExit(main())
