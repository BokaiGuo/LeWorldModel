"""Same-backend physical replay and model prediction audit for matched CEM."""
from experiments import pc_v4_pusht_rgb_feedback_audit_20261104_codex as audit
from experiments import pc_v7_pusht_matched_warm_cem_dev_20261104_codex as run

if __name__ == "__main__":
    run.configure()
    audit.OUT = run.PREFIX.with_name(run.PREFIX.name + "_audit.json")
    audit.main()
