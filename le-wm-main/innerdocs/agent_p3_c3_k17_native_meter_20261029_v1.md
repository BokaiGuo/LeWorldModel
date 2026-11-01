# K17 six-slot native meter seam

The fixed prospective warm10/first5 ceiling is six constructors/close, 12
resets, 24 set-state, 18 goal-setter, 12 goal-render, six goal-bind, 90 public
steps, 924 physics steps, and 306 explicit renders. Wall is 900 seconds.
The meter journals intent durably before each callback, counts attempted calls
without refund on callback failure, records returns/failures and allows close
as cleanup. A journal failure prevents a new non-cleanup callback. Cleanup may
run even when intent logging failed, recording the loss as a secondary error.

This is not the six-slot outer runner or a complete saved consumer. It does
not bind sources, validate six-slot identities, publish a final failure record,
or prove wall through publication. Actual native errors may exceed metadata
quotas, in which case retention depends on the separately bounded failure
reserve and successful I/O. Do not use this seam as production authorization.
