# Timed execution: exact-Wall wiring diagnostic

This one-task diagnostic checks the new common metered execution entry point
with real exact-Wall dynamics. It is not a performance, feasibility, PC2, or
PC3 experiment. Use start `[-0.5, 0.317, -1.17]`, goal `[0.5, -0.283]`,
horizon 15, packet length 5, cap 1,920 tokens, seed 2029311000 and the
existing v13 nominal bank. Run exactly once per arm: v17 and four-start
shared-bank CEM. Do not tune, retry, or select the task based on outcome.
The task is checked against the locally enumerated recent panels; wider
provenance is unknown.

Before saving traces, stop the synchronized planning clock. Save all actions,
features and scores with the call ledger. Independently recompute every
endpoint using the existing alternate implementation and verify per-call
and total token counts, complete-chain selection and common first bank.
Timings include trace capture and Python overhead and exclude serialization;
they do not include visual encoding or GPU cost and support no PC3 cost claim.
