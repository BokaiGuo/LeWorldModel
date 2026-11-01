# Independent P2 program library HOST CUDA runner

The supervisor locks the existing cooperative inode before the child starts
and passes its open descriptor through child exit. The child verifies the
inherited lock before importing torch, reconstructs the fixed prepare, binds
the device UUID and model tensor identity, then independently optimizes both
arms at each of four pinned cp15 contexts. Each branch records nested model
entry attempts/returns, SVD, scores, literal action packets, and context-chain
events in a durable hash-chain. Branch failures are retained; unstarted slots
stay explicit. The supervisor does not acquire a global GPU reservation.

The saved reader is a separate process, but reuses the branch validator and
reconstructs recorded content, not independent neural reexecution. It checks
the complete roster, model/device identity, source hashes, supervisor/child
journals, branch data and per-axis counts. The source list cannot prove a full
dynamic closure. Equal ceilings do not prove equal FLOPs, wall, energy, model
loading cost, or equal total compute. Known contexts are not independent
confirmation; no scientific gate is modified.

No physical bridge or physical result is produced by this module. A future
physical binding must consume only authenticated winner raw bits and exact
native cp15 references, preserve early stop/failure, and never mislabel a
fixture or partial journal as a complete physical receipt.
