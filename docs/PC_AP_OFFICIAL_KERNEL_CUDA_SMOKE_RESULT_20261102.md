# Pinned AP CEM kernel: recorded-RGB CUDA smoke result

The predeclared single-decision development run in
`PC_AP_OFFICIAL_KERNEL_CUDA_SMOKE_PROTOCOL_20261102.md` completed on the
host RTX 5060 Ti. The frozen parameter hash before and after is
`3ec019f167066551122f2df0016c629394cb30efa72ba84e74488daa78484da6`.
The callback recorded 34 predicted blocks in three batches, as fixed by the
16-by-two CEM budget plus two final means. The retrieved record is memory
episode 0, start 5, horizon 25. The refined mean's predicted target cost
was 371.442596 versus 374.981262 for the initial prior, so the fixed rule
chose the refined mean. The measured 0.081 seconds includes the CUDA/NumPy
bridge and planning but excludes memory encoding and retrieval construction.

The previous paper-based NumPy implementation used the same saved query
RGB hashes, retrieved the same record, and recorded 34 predicted blocks; it
selected a different normalized block and reported predicted cost
371.644814. The random generators and standard-deviation rules differ, so
this is an engineering observation rather than a controlled attribution to
one difference or evidence of physical superiority. No physical action
was executed. Memory provenance, official checkpoint-to-training binding,
total compute and official AP end-to-end parity remain uncertified.

Result report SHA-256:
`2e0df092eac6fdb69af2e3747d597b01e6c5660cf82f3461b459cbb9d547991c`.
Runner SHA-256:
`9b3098abeaa0c2e1bce8938e8a6b93def366fe88061e3233da7ce1d799593b59`.
Kernel SHA-256:
`6ddfa22a333a5f899794085dc30604157541dcb9a8d995c33e5e3110fbb1f129`.
