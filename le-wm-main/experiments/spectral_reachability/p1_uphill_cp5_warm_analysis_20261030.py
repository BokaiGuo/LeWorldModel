"""Run the fixed physical analysis against the warm-start experiment."""
from . import p1_uphill_cp5_warm_physical20_20261030 as warm
from . import p1_uphill_cp5_analysis_20261030 as analysis


def main():
    physical = warm.physical
    physical.NAME = warm.NAME
    physical.PLAN_DIR = physical.ROOT / "results/p1_uphill_cp5_warm_cuda20_20261030_RUN01"
    physical.MANIFEST_SHA = "b9fd1ba1b6233132b2ad3ba9b8013fa5f88faf598ee81dba53b20d9ece0f6bad"
    physical.REPORT_SHA = "0334d3485c6631b357b7aada1855767b221800d684955d0b932cd7d4c7c00688"
    physical.configure, physical.build = warm.configure, warm.build
    return analysis.main()


if __name__ == "__main__":
    main()
