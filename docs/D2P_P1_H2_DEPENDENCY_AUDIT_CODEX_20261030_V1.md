# H2 host dependency candidate and unresolved isolation gates

The adjacent `D2P_P1_H2_DEPENDENCY_INVENTORY_CODEX_20261030_V2.json`
is an exploratory, non-authorizing inventory from the current sandbox view.
It is **not** an executable wrapper manifest and is not a minimal runtime
closure. Its generator reads only reachable Python source text, the explicitly
identified blind bundle, the two offline model files, and library/device
metadata. It does not import the H2 driver, load weights as a model, initialize
CUDA, run physics, or read historical outcomes.

## Observed candidates

- The static walk finds 89 project Python files reachable through syntactic
  imports, including imports inside unexecuted functions and conditional
  branches. This is an overapproximation, not a list of proven necessary
  files. Relative and dynamic import sites are flagged and are not resolved;
  dependencies outside the `experiments` package are only named, not closed.
  Reduce and independently review the list before staging it.
- The blind bundle and `weights.pt`/`config.json` hashes agree with the
  model-file binding in that bundle. This establishes byte equality at scan
  time only; it does not establish immutable storage or an independent trust
  authority. The wrapper must grant only the exact files, not the results or
  model directory recursively.
- `.venv/bin/python` resolves to `/usr/bin/python3.12`. The inventory records
  PyTorch/NVIDIA packaged `.so` candidates and `/usr` driver-library links;
  inclusion is not proof that any particular library is loaded or necessary.
  Python's `.pth` startup file is separately hashed because it executes code.
  Library bytes, transitive ELF dependencies, wheel metadata and symlink
  targets have not been fully sealed or audited.
- The scanned sandbox cannot see `/dev/nvidia0`, `/dev/nvidiactl`,
  `/dev/nvidia-uvm`, `/dev/nvidia-uvm-tools` or `/dev/nvidia-modeset`.
  This is not a claim about the host or about CUDA/model failure. Those
  candidates are not a complete device inventory and must be checked by an
  authorized operator on the real host without initializing CUDA.

## Required before a coordinated run

1. Construct a clean immutable source/runtime snapshot from a reviewed
   dependency closure, audit every recursively allowed directory and `.pth`
   startup file, and seal code, binaries, libraries, bundle and model bytes.
   The present wrapper's recursive grants, including a possible broad `/usr`,
   are not a proven least-privilege runtime. Review dynamic imports and all
   loader dependencies; fail closed on unexpected accesses.
2. On the actual host, identify the coordinated GPU, device nodes, driver
   libraries and host-specific CUDA paths, and verify access under the final
   policy. Statting a node and seeing a library do not prove CUDA execution.
   Confirm immutable snapshots and recheck hashes around the run.
3. Run denial tests against representative historical outcome, future-frame
   and physical-state paths without reading their contents. Landlock allows
   metadata operations and does not revoke previously opened descriptors.
   Audit inherited FDs, `/proc`, other same-user processes, shared memory,
   Unix sockets, device operations and other IPC separately; do not claim a
   complete process boundary from Landlock alone.
4. Provide and verify external network isolation. Offline environment variables
   are not enforcement, and the existing wrapper does not isolate network.
   Establish GPU exclusivity, resource limits and an atomic cross-entry claim.
5. Independently review the H2 science and output consumer. There are 16 known
   selected/zero rows and eight unknown greedy conditions; total-budget parity
   and physical suffix execution are not established. Earlier adapter tests
   read historical outcomes, so the method is exploratory and has no
   preregistration or clean-blinding guarantee.
