# E1 six-slot physical: MAIN review candidate, not authorized production

The saved-only asset preflight reconstructs the entire frozen six-slot CUDA
consumer, compares it to MAIN's persisted receipt SHA
`1ca8e949cc4b219b9b03adb3d34ee859e924e0078f9f81c519be2c03ed77be3d`, reads
and authenticates the real selected plan/raw artifacts, and joins all six plans
to the original two moving sourcepack roots/goal/task through the original
task bridge. No physical outcome enters selection. This is not a physical
receipt and does not authorize dispatch.

The new RAM-tested branch driver uses the existing sealed native branch,
operation meter, tracking receipts and cleanup. It retains all six rows, checks
the root and selected-plan seals before dispatch, and records per-resource
attempt/return counts. A new independent ledger checker validates nested
intents/returns and all six plan identities; the saved checker also validates
frames, original source history, goals and transitions.

The fixed full-horizon native operation ceiling is six constructors/closes,
12 resets, 24 set-state calls, 12 goal renders, six goal binds, 120 public
steps, 1,224 physics steps and 132 explicit renders/PNGs. Early stop is not
filled or used to remove a branch. The 864 controller and 24 extra replay calls
per model axis are inherited separately, not physical costs or equal FLOPs.
No training, new neural inference, retry, or scientific admission.

The implementation now contains prepare/verify/run/consume entry points and
an independent saved checker. A temporary-disk fixture exercises the real
Store with a fake environment, all six branches and 132 PNGs, including
partial and early-stop examples. These are fixture evidence only. Old four-arm
producer/consumer hard-code two
learned unknowns and cannot be represented as the new six-arm receipt. Do not
run production until MAIN reviews the following storage and completion
boundaries, independently tests the source closure and accepts the protocol.

The new Store partitions the unchanged 33,554,432-byte output cap as follows:
up to 132 stored render artifacts with a shared 21,626,880-byte allocation
(132 × 163,840, not a per-frame truncation); journal 6,291,456;
execution binding 262,144; report 4,194,304; emergency marker 32,768. Sum is
32,407,552 and slack is 1,146,880 bytes. This is an integer bound on admitted
stored bytes, not a proof that arbitrary native outputs fit those allocations.
The journal includes the intent/return records, frames' metadata, branch rows,
and the initial prepare; the report duplicates operation receipts and rows,
and its entire serialized size is charged to the report allocation. The real
temporary fixture exercised six fake branches, 132 PNGs and all journal/row
copies. Separate temporary-disk tests exercise real 224×224×3 high-entropy PNG,
every allocation at its exact byte ceiling, and oversized writes. Ceiling-test
bytes are synthetic transport payloads, not native evidence. An oversized
payload exceeding its aggregate allocation is rejected without modifying an existing artifact. These limits are
engineering stop conditions, not scientific thresholds; they do not establish
that native exception messages or malformed renders have a finite size.

The Store keeps the existing 209,715,200-byte free-space floor and inherited
600-second pre-dispatch/ordinary-write wall. Terminal writes and cleanup are
exempt as in the inherited implementation; inclusive completion wall is not
certified. The emergency allocation is held separately from the report. When
journal/report publication fails or an exception escapes `run`, a small
`emergency.json` records partial status, denominator, journal hash and chain
tail when the filesystem remains writable. It is not a full six-slot receipt;
an interruption, I/O failure or unexpected oversized output can leave intent
without return, and no software reserve guarantees a disk that has failed.
Saved consumer validation requires the complete journal and rejects that case.
No successful production execution is implied by the fixture.

MAIN-only CLI after review and final freezing, from this project root:

```
.venv/bin/python -m experiments.spectral_reachability.agent_e1_six_slot_physical_20261029_v1 prepare --output /ABS/NEW_PREPARE.json
.venv/bin/python -m experiments.spectral_reachability.agent_e1_six_slot_physical_20261029_v1 verify --prepared /ABS/NEW_PREPARE.json --prepare-sha PREPARE_SHA
.venv/bin/python -m experiments.spectral_reachability.agent_e1_six_slot_physical_20261029_v1 run --prepared /ABS/NEW_PREPARE.json --prepare-sha PREPARE_SHA --output /ABS/NEW_OUTPUT --authorize-physics
.venv/bin/python -m experiments.spectral_reachability.agent_e1_six_slot_physical_20261029_v1 consume --output /ABS/NEW_OUTPUT --report-sha REPORT_SHA --receipt /ABS/NEW_RECEIPT.json
```

Prepare and verify rebuild the upstream saved consumer; neither invokes physics
or neural inference. MAIN must provide the actual hashes and new paths. No
worker production, scientific admission or claim of fresh independent roots.
