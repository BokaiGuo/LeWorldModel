# Durable shadow closed loop: frozen development contract

New files and output prefix only; retain all previous results, scripts and
failures. Reuse the preceding two exposed source branches, same frozen
checkpoint, RGB-only feedback, prefix, CEM seeds/budget and same-action shadow
and legacy control. Acquire the same advisory GPU cooperative lock
`results/e3_p2_host_cuda_cooperative.lock` before model loading and retain it
through all model calls. No hidden simulator state enters planner inputs.

Before each bridge reset and each bridge action, an immutable exclusive-create
snapshot must be flushed, `os.fsync`ed, closed, and its parent directory
`os.fsync`ed. Action snapshots contain the bound action, action clock,
source/phase and RGB feedback context; CEM snapshots additionally contain the
candidate trace, scores, seed and selected packet. A reset snapshot identifies
the immutable source and start index, not a state exposed to the planner.
For the paired action, a single snapshot may be shared by both bridges, but
its content hash is checked before each dispatch. Each bridge separately has
an `attempt` event synchronously flushed and fsynced **before** calling it,
and a `returned` event synchronously flushed and fsynced **after** its
returned RGB has itself been saved with file and parent-directory fsync.
Journal creation is directory-fsynced. If any snapshot, journal, fsync, call,
hash or clock check fails, do not dispatch the next bridge or action and do
not retry. Explicit per-bridge pending attempts remain uncertain if the
process dies between the external effect and its returned event.

Test real fsync invocation, file/directory ordering, failed fsync with no
dispatch, and abrupt process termination after shadow return and during
legacy dispatch, using separate test processes and fake bridges. Tests do not
prove storage hardware power-loss behavior. Then run a separate real host
two-root smoke under the GPU lock, report source hashes, event/snapshot counts,
per-bridge commands and RGB feedback; verify all 13 prefix frames against
prior sources for each root. Preserve incomplete event logs and artifacts;
do not call this statistical evaluation or evidence of physical success.
