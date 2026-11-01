# P5 instrumented CUDA supervision

This is a new, independently scoped production wiring, not permission to run.
It leaves the earlier runner, consumer, supervisor and checkpoint baseline
unchanged. The production child strictly loads the real checkpoint through the
pinned original CUDA backend. It never performs a CPU neural forward. The new
wrapper captures module names/classes/training flags, every parameter's
shape/dtype/device/requires-grad and value hash, normalization tensors, and
PyTorch precision/autocast flags. It compares them to the real CPU-loaded
checkpoint baseline; CUDA tensor copies to CPU are for hashing only.

Capture points are inside the actual child call path: immediately after the
first encode attempt is journaled and before the first encode, and immediately
after the final (branch 163, step 5) CUDA transition returns and synchronizes,
before its return event is journaled. Each snapshot is exclusively created,
fsynced, bound to the input baseline and producer source SHA, and anchored to
its corresponding attempt event hash and sequence number. Pre-state must match
the frozen checkpoint and post-state must equal pre-state; failure stops the
run without retry. A timeout or crash can leave no snapshot, one snapshot, or a
truncated/incomplete journal; preserve those bytes as partial.

The new supervisor uses the existing UUID/memory admission and external
process-group watchdog. It starts exactly one fixed child, reviews only after
confirmed child exit, and marks complete only for exit zero, a complete strict
164-branch journal, both valid snapshots, unchanged pinned sources and
deadline compliance. The offline `accept` command additionally requires
independently supplied supervisor and journal SHA values and writes a new
exclusive acceptance file. It repeats all reviews and reports off/on separately
(82 each). No future labels are opened; no scores or scientific admissions
are produced.

Limitations: the admission free-memory check is not a reservation or GPU memory
cap. SIGKILL cannot guarantee prompt termination of uninterruptible tasks, nor
handle supervisor SIGKILL or host failure. An external service watchdog remains
necessary. Snapshot/log hashes are self-attestation, not independent proof of
hardware, kernel precision, or absence of transient between-snapshot state
changes. Autocast is checked outside model calls, not dynamically traced inside
them; the existing backend uses inference mode for the calls. Training versus
panel rendering distributions are not established. A complete receipt is not
a scientific pass.

The production entry point is `launch`, with the same explicit UUID, memory and
time choices as the preceding supervisor plus `--authorization
MAIN_THREAD_APPROVED_INSTRUMENTED_164_RGB_CUDA` and a unique `--run-suffix`.
It is not executed by tests. After a stopped complete run, `accept` requires
`--run-suffix`, `--supervisor-sha`, `--journal-sha`, and an exclusive
`--output-name` beginning with this module name plus `_accept_` and ending in
`.json`. The prior acceptance tool expects the older, uninstrumented launch
command, so the new `accept` path is required for this wiring.

Unit and integration tests use fixed real metadata plus fake tensor records and
a data-only fake backend; the watchdog's own tests use ordinary child processes.
No GPU, simulator, CPU neural forward, future label, or reserved root is used.
