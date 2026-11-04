"""Apply the existing independent saved-record checks to the v4 execution."""
from experiments import pc_response_temporal_v3_reacher_rgb_live_audit_20261104_codex as audit
from experiments import pc_v4_reacher_rgb_live_20261104_codex as run


if __name__ == "__main__":
    run.configure()
    audit.OUT = run.PREFIX.with_name(run.PREFIX.name + "_audit.json")
    audit.main()
