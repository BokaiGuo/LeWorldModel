"""Reference packet unroll audit of V5 saved PushT contexts."""
from experiments import pc_v4_pusht_saved_context_adapter_audit_20261104_codex as audit
from experiments import pc_v5_pusht_saved_context_20261104_codex as run

if __name__ == "__main__":
    run.configure()
    audit.OUT = run.PREFIX.with_name(run.PREFIX.name + "_audit.json")
    audit.main()
