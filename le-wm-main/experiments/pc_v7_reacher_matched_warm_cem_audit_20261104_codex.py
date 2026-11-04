"""Apply the established two-start RGB record audit to the matched run."""

from experiments import pc_v4_reacher_two_start_cem_rgb_audit_20261104_codex as audit
from experiments import pc_v7_reacher_matched_warm_cem_dev_20261104_codex as configured

if __name__ == "__main__":
    audit.configuration = configured
    audit.OUT = configured.PREFIX.with_name(configured.PREFIX.name + "_audit.json")
    audit.main()
