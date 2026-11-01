# E3 single-capture native packet

One actual env.step followed by the original single-render `capture` and native
success/reward validation per returned step. Saves action intent before dispatch,
actual raw bits, observations and native return; stops immediately on native
termination/truncation, and an exception leaves prior returns plus pending intent.
Actual meter hooks, source/goal authority, output durability and episode cleanup
remain responsibilities of the not-yet-implemented launcher/strict consumer.

The earlier `agent_e3_feedback_native_packet_20261029_v1` uses allmode.observe,
which renders off/on/off three times and therefore is NOT the E3 32-render
contract. Do not integrate it as the E3 packet adapter. This replacement has only
fixture testing, not production, and no scientific admission.
