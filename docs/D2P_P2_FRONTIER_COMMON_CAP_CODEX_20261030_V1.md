# P2 native frontier with a common pre-dispatch cap

`codex_p2_frontier_shared_cap_gate_20261030_v1.run_native` is a new,
independent execution path, not a modification of the existing runner. It
reuses the existing native joint compiler, the 80-candidate no-composition
frontier, the 40-candidate own-context conditional-suffix composition
frontier, both winner replays, the existing native model-entry guards, and
derivative-array persistence. It does not load a model or perform physics.

The same common per-arm limits are checked *before* the guarded predict,
action-encoder, Jacobian, and SVD/solve attempts. Shared joint compilation is
charged to both standalone arms, but performed once; failed and pending
reservations are not refunded. An attempt or journal failure poisons the
gate. The program reserves a new output directory and does not retry.

Illustrative API shape, **not authorization to execute**:

```python
run_native(model=frozen_host_cuda_model, device=coordinated_cuda_device,
           ready=externally_sealed_ready, compiled=externally_sealed_compiled,
           provenance_by_arm=externally_sealed_provenance_by_arm,
           output="/absolute/new_unique_attempt",
           caps={"predict": 168, "action_encoder": 168,
                 "jacobian": 41, "vjp_directions": 7872,
                 "svd": 42, "solve": 41})
```

The values illustrate a **common cap**, not equal usage: with all suffix
solves active, the original frontier attributes per arm 168 predict and 168
action-encoder calls, but 1 versus 41 Jacobian requests and 192 versus 7872
mathematical VJP directions. SVD/solve usage can vary with zero-radius
suffixes. Eighty versus forty candidate evaluations also means this is not the
older fixed-equal-slot P2 contract. No padding, refund or lowering of the
scientific gate is performed. A common cap is not a matched resource vector.

## Required before host use

This code does not itself authenticate the `ready`, `compiled` and provenance
objects, prove source/checkpoint/goal seals or independent pre-outcome freezing,
reserve a GPU, implement a complete source census, synchronize/measure all
device work, capture immutable model state before/after, enforce hard wall or
storage caps, or deduplicate across other entrypoints. It cannot enforce or
measure complete FLOPs, memory peaks, I/O, visual encoding, model loading,
physical steps, numerical-library internals, or costs outside its hooks.
Journal callbacks must remain synchronous and durable; events written before
a failure are retained, while failure-report writing is best effort.
All these omissions remain blocking for a full fair-budget claim and deployment.

The candidate bank is limited to the frontier's existing supported root IDs;
using them is known-root calibration, not a fresh independent cohort.
Main-thread review must verify the complete method, per-arm/candidate
denominators, scientific gates, source and input seals and total resource
instrumentation. The code never permits CPU neural fallback. No CUDA,
neural model, or physical execution was performed for this delivery.

CPU tests exercise real existing frontier candidate loops with synthetic
arithmetic replay and the native entry guard with synthetic metadata. They do
not establish real CUDA execution, model validity or physical effects.
