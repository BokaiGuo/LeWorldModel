# P3 rejected-decision capture V2

The main thread reports the prior V2 host job finished with four common-gate
rejections and two signal slots completed. This note does not independently
certify those results, infer rejected decision values, or reinterpret them.
The original gate requires `continue_expanding=True`, both types `progress`,
and an integer horizon of 1 or 3 for every policy. No mapping from R to
progress and no change to that gate is made here.

The separate callback seam
`experiments/spectral_reachability/p3_decision_rejection_capture_20261030_codex_v2.py`
records `decision_gate_rejected` when the original common gate raises
`ValueError`. Before propagating it, the event contains the original decision,
complete spectral diagnostic, scheduling node, dataclass signal fields,
full runtime signal receipt and both responses, root/context digests, policy,
stage, original gate error, and the resource receipt at that instant. With the
existing durable `Store.event`, the event is fsynced before returning. The slot
then remains failed, retains attempted work and dispatches no new panel,
replay, packet or encoder call for that decision. A later slot can still be
attempted according to the existing six-slot lifecycle. If the event write
fails, the write error takes precedence; no claim is made that the rejected
payload survived or that the outer journal can be completed.

The gate receipt is a point-in-time snapshot; the slot-final failure receipt is
taken later, so `wall_ns` may increase during persistence even with identical
resource operations. Neither time is FLOPs nor a physical work proof.

This is a new **callback seam**, not a production driver or a new scientific
result. The existing V1 decision store starts with a V1 contract and requires a
V1 roster schema at terminal; its existing consumer rejects the new event and
schema. A fixture can test durable event bytes with that store, but cannot
produce a valid V2 terminal by calling its old `finish`. Do not connect this
seam to the sealed driver or claim a complete V2 saved certificate.

Before a future authorized host run, implement a versioned durable journal and
independent saved consumer that bind this exact source and new contract, audit
the event against schedule resource return, full inputs, gate and terminal
failure, reject missing/reordered/altered records, and preserve fixed six-slot
denominators and partials. Separately pin the changed adapter/driver closure,
resources and supervision; coordinate GPU and simulation explicitly.

For a scientifically meaningful new contract, decide *before* collecting new
data whether to retain the present supported-decision population or to
implement and prospectively specify a genuine R-capable intervention, including
its candidate generation, executable action semantics, equal-budget accounting,
stopping rules and failure denominators for all arms. This would be a new
protocol, not a post-hoc conversion of failed decisions or a repair of P1.
MAIN01/MAIN02 observations remain historical and non-independent for any
future choice informed by them.
