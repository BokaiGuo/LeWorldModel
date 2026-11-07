"""Independent physical replay using the existing geometric audit implementation."""
from experiments import pc_geometric_contact_equivalence_audit_20261106_codex as audit
from experiments import pc_osfal_pusht_heldout_geometry_20261107_codex as extension


def main():
    extension.audit_preconditions()
    extension.configure()
    audit.OUT = extension.PREFIX.with_name(extension.PREFIX.name + "_replay.json")
    audit.main()


if __name__ == "__main__":
    main()
