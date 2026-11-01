# P2 replay-reserved cooperative CUDA launcher

`experiments.spectral_reachability.p2_replay_exclusive_cuda_20261101` creates
a new attempt directory, verifies the externally pinned inventory and inherited
seals, and re-verifies them under the existing E3/P2 cooperative lock. It
requires explicit operator coordination and a physical GPU UUID, then records
an independent `nvidia-smi` device/compute-process census before initializing
CUDA. It runs the four pinned roots in order with the protocol's per-arm caps,
and constructs provenance from each actual compiled candidate bank.

Source verification includes the checkpoint files before and after execution;
the loaded model is hashed and checked for frozen evaluation state after work.
CUDA synchronization and cache release happen while the cooperative lock is
held. Each root has its own runtime output and resource journal. The parent
attempt records completion or, on ordinary exceptions, best-effort failure
details; abrupt termination, full disks or filesystem failure can prevent the
final record. Never reuse or resume an attempt directory.

The lock is advisory, not hardware exclusivity. The census observes compute
processes at one instant, not all GPU users or other hosts; operator
coordination remains essential. Source hashes do not establish transitive
closure, independent registration, or scientific authority. The launcher
itself is new and is not included in the existing source inventory; its
independent review and binding are still required before claiming a reviewed
or independently preregistered launch. No execution is authorized by this
file. No physical bridge is invoked, and P1/P2 admission remains false.

This integration has CPU tests only; no CUDA or physical run was performed as
part of development.
