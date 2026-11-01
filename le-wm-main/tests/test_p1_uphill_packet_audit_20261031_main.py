import numpy as np
import pytest

from experiments.spectral_reachability import p1_uphill_packet_audit_20261031_main as audit


def test_action_binding_and_stop_count():
    actions = np.arange(20, dtype="<f4").reshape(10, 2)
    events = [{"step": i + 1, "actual_bits": actions[i].tobytes().hex()} for i in range(8)]
    audit.verify_actions(actions, events, 8)
    events[5]["actual_bits"] = actions[6].tobytes().hex()
    with pytest.raises(Exception, match="executed action mismatch"):
        audit.verify_actions(actions, events, 8)
    with pytest.raises(Exception, match="returned count"):
        audit.verify_actions(actions, events, 10)
