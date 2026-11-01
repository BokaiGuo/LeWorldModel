# E3 unified native meter: bounded component, not E3 strict

This new per-cell meter uses the unified publication Store's existing 1800 s
clock, 512 MiB total, 64 MiB final reserve and 200 MiB floor. The caller must
start that Store clock before authority verification. It does not alter or
extend the frozen prior meter's independent 600 s contract.

The native subclass is the original `metered_class`; only its meter and the
close-only environment proxy are new. Constructor, reset, setters, goal
render/bind, explicit render, public step, physics step and close use the
following predispatch per-cell caps: respectively 1, 2, 4, 3, 2, 1, 32, 30,
304 and 1. These are caps, not assertions that early-stopped or failed jobs
performed the full counts. Encoders, predictions, AD and their separate axes
are not charged by this physical meter.

Every admitted call uses the unified native-meter intent/return/failure
events with a literal episode, region, unit and matching token. Nesting is
LIFO, including parent failure after a child failure; attempts are not
refunded. Exception type, message, repr and traceback are retained with
explicit bounds of 2048/4096/4096 characters for message/repr/traceback.
If recording fails, the original exception remains primary with a note.

`Meter.cleanup` attempts the actual close even if the clock or journal guard
prevents recording its intent. Such a path is necessarily incomplete and
cannot earn a complete ledger. A failure inside the original constructor may
trigger the original subclass's internal cleanup; without a returned env it
is not fabricated as an independently metered close.

`consume(output,report_sha,completion_sha,episode)` first verifies unified
publication and then independently reconstructs native counts, nesting,
region restrictions, stop-before-next-step and close. Early stop is not
padded. A completed ledger requires complete transport, no outstanding or
failed native operation, one successful constructor and one successful
close. This is **not** a native-state/RGB/encoder/numeric semantic strict,
nor does it authenticate the episode's source/goal. Unknown work stays
unknown. Existing publication cannot certify an over-deadline completion;
this reader does not claim recovery of unpublishable timeout output.

The exclusive `agent_e3_authorized_episode_20261029_v2.py` composes the new
meter with the existing authority/model/setup lifecycle and the original
native factory. It is an integration component, not yet a complete prepare,
production CLI, or independent experiment consumer. MAIN must review and
authorize any eventual host run. No production, model training or physical
execution was performed by these fixtures; all scientific gates remain false.

Fixture command from `/home/gbk/code/LeWorldModel/le-wm-main`:

```
.venv/bin/python -m pytest -q tests/test_agent_e3_native_meter_20261029_v1.py
```
