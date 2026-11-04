# P2 joint versus conditional: sealed host development protocol

Known exposed B0 only; the fixed 16-candidate bank is consumed in order. No
physical result is used to select candidates. This is a model-only development
comparison, not an independent confirmation, mechanism result or P2 admission.
The 20-dimensional joint Gauss–Newton control is not historical SRA. Its
`sra_initialization` ledger category is solely a legacy category label; the
previously failed historical SRA source binding remains failed.

Both arms invoke the existing native objective and optimizer with the same
candidate bank, 16 initializers, 144 scores and 128 explicit optimizer
gradients per arm. Conditional Jacobians are 192×10 and joint Jacobians
192×20. Equal call counts do not imply equal FLOPs, equal backward graph size,
equal total compute, energy, or equal inherited setup cost. Each Jacobian
requests 192 output directions; actual internal autograd engine invocations
are not measured.

The launcher binds current source hashes, authenticated inherited input
handoff, pair, context, goal, bank bytes, checkpoint file and tensor identities,
normalization, device UUID and cooperative lock. This is an explicit manifest,
not a complete transitive environment or original-provenance closure; in
particular it does not repair or authenticate the historical SRA binding.
CUDA entry requires explicit authorization, exactly one UUID-visible device,
device UUID verification, frozen model, and postchecks. The 900-second limit
is checked at dispatch boundaries, not a preemptive kernel timeout. Failures
retain partial ledger/report; abrupt process termination can leave only partial
files and must not be treated as completion. No retries or replacement.

The fsynced hash-chained ledger records attempts, returns, failures, timing,
model entrypoint calls, transition input shapes and optimizer receipts. Each
real transition records three input context tokens and one selected successor
token; these are program-level tokens, not all internal model token operations.
Synchronized inclusive wall timings include nested work, synchronization and
ledger overhead and cannot be added across nested calls or treated as GPU
kernel time. Explicit optimizer gradient calls are counted; internal Jacobian
backward calls remain unknown. Model load, upstream bank production and I/O
are not equated between methods.

The saved consumer validates the plan/current source binding, saved report and
ledger hashes, resource census, timing and token events, initializer dimensions,
fixed bank and conditional prefix, optimizer receipts, winner action hashes and
model-score tie-breaking. It cannot reconstruct unrecorded gradients or prove
the mathematical correctness of a compromised model/runtime. Fixture tests
exercise the same interfaces with analytical functions and are not CUDA or
physical evidence.
