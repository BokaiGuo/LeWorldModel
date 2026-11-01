# P2 future per-prefix recompile exploration

This new design was written after H25 results were known; it is not
preregistered and cannot turn those results into independent confirmation.
In the saved H25 banks, each joint arm contains 16 distinct first packets
and 16 distinct second packets; each library arm contains four of each,
evaluated as the 4x4 Cartesian product. The original library's second
packets are fixed from the original cp15 basis, not re-optimized at each
resulting first-packet context. Each original arm has its own cp15
Jacobian/SVD and 16 full10 candidates; these facts do not make the candidate
supports identical or establish equal total compute.

For every fixed cp15 case, each arm receives its own resource meter and the
same caps: 40 packet and nested model calls, five Jacobians, five SVDs,
17 scores and 450 seconds. Calls are upper bounds, not targets, and actual
cost axes must be kept separate.

Joint evaluates the existing 16 full10 root spectral candidates. The new
prefix-chain evaluates four root-derived first packets; after each actual
first packet it separately computes a 192x10 next-packet Jacobian and local
SVD, and evaluates two local directions with both signs, for 16 full
candidate paths. Each independently derived second packet stays within the
same two-block logit norm cap. Both arms select the first lowest finite
terminal latent MSE, then replay the selected full path from the same
initial context. There is no feedback replanning after physical execution.

The chain spends more Jacobians/SVDs and can reuse a prefix across four
continuations; the joint search can use the same ceiling but need not waste
operations. Different candidate supports and compute patterns mean this is
an exploratory algorithm comparison, not isolation of composition or equal
total compute. Model load, dynamic source closure, VJP engine calls, FLOPs
and energy remain separate or unknown.

The module is a callback kernel plus CUDA-only adapter, not an authorized
production entry: a host driver, fixed source preparation, lock lifetime,
durable journal, resource checks, independent saved consumer, and new
physical-action binding remain necessary before any experiment. No neural
inference or physical run has been performed for this new contract.

Offline fixture tests and inherited kernel tests: 19 passed. They check four
distinct prefix-context Jacobian requests, separate caps, early failure with
no retry, winner replay mismatch, CPU-device rejection, and model-free
synthetic execution. Fixtures are not experiment evidence.
