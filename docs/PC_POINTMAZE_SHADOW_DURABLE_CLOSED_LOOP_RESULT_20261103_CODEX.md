# Durable shadow closed-loop development smoke

The new isolated executor completed the frozen two-root replay with the
existing CUDA cooperative advisory lock, unchanged model, same seeds and the
same shadow-only RGB feedback as the earlier smoke. The prior executor,
results and negative attempts were not edited.

For each of the two resets and 34 action pairs, an exclusive-create snapshot
was written, flushed, file-fsynced, closed and parent-directory-fsynced before
dispatch; each bridge checks the snapshot hash before its own attempt. The
journal records a separately fsynced attempt and return for each bridge:
72 attempts and 72 returns including four bridge resets. Every returned RGB
has its own file- and directory-fsynced artifact, bound to its event by hash.
One reset snapshot may be shared across its two bridges; likewise each
same-action pair shares its action snapshot.

The run performed 500 scored CEM candidates in 40 batches, 13 goal/feedback
encodes, four bridge resets and 68 high-level physical-simulator commands.
Both roots passed 13/13 shadow and legacy prefix checks; model hashes matched
before/after. The main result records 6.66 s elapsed and approximately
4.5 microseconds waiting for the cooperative lock. The two branches' complete
actions, shadow frames and legacy frames happen to be byte-identical to the
earlier smoke. This is a repeated development run, not independent evidence
or a performance comparison.

Fourteen tests passed, including the previous saved-result audit, verified
`fsync` calls that delegate to the real system call, snapshot/response
binding, reset and per-bridge ordering. Injecting EIO at snapshot or attempt
fsync prevents dispatch; EIO persisting a returned shadow RGB prevents the
legacy action. Separate child processes terminate abruptly after shadow return
and during legacy dispatch; their surviving journals distinguish
shadow-returned/legacy-not-attempted from legacy-attempted/no-return.
The independently saved intentional-crash journals, snapshots and receipt
are under `results/pc_pointmaze_durable_fault_receipt_20261103_CODEX/`.

Limits: no software journal can prove whether a command executed if a process
dies after the external effect but before its return event is fsynced. Such an
attempt remains explicitly uncertain and is not retried. Tests exercise real
fsync syscalls and abrupt process exit, not power interruption or storage
hardware guarantees; a final journal line may be torn on an actual crash.
This run has only two exposed roots, no physical-position observation supplied
to the planner or evaluator, and no general control or success claim. The
planner still receives shadow RGB and RGB-derived proprio only.
