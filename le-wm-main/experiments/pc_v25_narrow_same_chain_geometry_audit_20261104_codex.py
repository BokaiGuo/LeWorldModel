"""Run the established geometry audit on the narrow prospective panel."""
from types import FunctionType, SimpleNamespace

from experiments import pc_v25_e0_prospective_same_chain_dev_20261104_codex as original_panel
from experiments import pc_v25_narrow_prospective_same_chain_dev_20261104_codex as narrow
from experiments import pc_v25_same_chain_geometry_audit_20261104_codex as original_audit


def main():
    namespace = dict(original_audit.main.__globals__)
    namespace["panel"] = SimpleNamespace(
        OUT=narrow.OUT, seal=original_panel.seal, journal=original_panel.journal,
        journal_audit=original_panel.journal_audit, fixed=original_panel.fixed, OLD=original_panel.OLD)
    check = FunctionType(original_audit.main.__code__, namespace, original_audit.main.__name__,
                         original_audit.main.__defaults__, original_audit.main.__closure__)
    return check()


if __name__ == "__main__":
    main()
