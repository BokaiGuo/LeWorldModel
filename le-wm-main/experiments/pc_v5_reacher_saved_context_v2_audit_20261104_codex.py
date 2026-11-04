"""Reference rollout audit for the authorized-device retry."""
from experiments import pc_v4_reacher_saved_context_audit_20261104_codex as audit
from experiments import pc_v5_reacher_saved_context_v2_20261104_codex as run

if __name__ == "__main__":
    run.configure()
    audit.OUT = run.PREFIX.with_name(run.PREFIX.name + "_audit.json")
    audit.main()
