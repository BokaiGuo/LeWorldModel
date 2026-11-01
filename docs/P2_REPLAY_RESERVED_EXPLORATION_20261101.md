# P2 replay-reserved CUDA exploration protocol

This is a proposed known-root exploratory engineering run, not a P2
confirmation. No CUDA or physics is authorized by this document. The four
roots are the ordered cases in the pinned frontier prepared protocol; they
have already been used and are not independent. Do not select roots based on
previous outcomes. For each root, use the original compiled goal, context,
action domain, two arms and 80/40 candidate denominators. Construct arm
provenance only after the actual compiled candidate bank is available.

Per-root, per-arm common limits: predict 168, action_encoder 168, Jacobian
41, VJP directions 7872, SVD 42, solve 41. Four predict and four action
encoder attempts per arm are held for replay. Shared compilation is charged
to both arms, but actually performed once. Limits are not equal expenditure,
FLOPs, or a full resource meter. Failed attempts are retained, not refunded.
Report every root/arm, failures, attempted resources and missing quantities;
no post-hoc threshold, sample removal, or winner claim.

Before any CUDA initialization a reviewed launcher must verify the pinned
inventory, inherited seals, checkpoint and compiled data, acquire the same
protected stable advisory lock used by cooperating E3 jobs, and hold it until
CUDA work and cleanup finish. This is not hardware exclusivity: other users,
processes, hosts and noncooperating launchers are outside that lock. Require
operator coordination and independent device census. A new exclusive output
and durable failure record are required for every attempt. Freeze and recheck
model state and sources; record actual device and resource usage. If any check
fails, stop; do not silently reseal or resume under the same attempt.

The source inventory is a hash snapshot, not proof of complete transitive
closure, independent temporal registration, scientific validity or physical
authority. It binds the available prepared sources, selected runtime files,
checkpoint files, and compilation report. Additional imports, runtime
environment and dependencies still require independent closure review.
The current physical bridge deliberately refuses dispatch because a new-plan
root/warmup/source authority is not integrated. No execution or comparison
using physical outcomes is authorized here.

This document was written after earlier frontier outcomes had been observed,
but before this new replay-reserved exploration. It is not preregistration of
those earlier outcomes. Independent timestamping/review and a launcher
binding this exact protocol and inventory remain necessary before calling the
proposed run independently preregistered. P1/P2 admission stays false.
