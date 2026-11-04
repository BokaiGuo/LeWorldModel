"""Same-backend physical and model replay audit for V5 PushT RGB."""
from experiments import pc_v4_pusht_rgb_feedback_audit_20261104_codex as audit
from experiments import pc_v5_pusht_rgb_feedback_dev_20261104_codex as run

if __name__ == "__main__":
    run.configure()
    audit.OUT = run.PREFIX.with_name(run.PREFIX.name + "_audit.json")
    audit.main()
