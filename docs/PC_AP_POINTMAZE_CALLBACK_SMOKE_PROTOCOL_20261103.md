# AP-compatible PointMaze callback: frozen-model smoke

Use exposed root 1888 and the saved initial/goal RGB, task mask and first
16 one-packet actions from the existing causal-CEM development source. Use
the same frozen checkpoint, camera projection and dataset normalizer as the
existing PointMaze proprio scorer. Compare the new AP callback's predicted
proprio costs against that scorer on exactly the same raw actions after one
normalization and fivefold packet expansion. Also compare callback output
when scored as one batch and in two halves. Assert model parameters stay
unchanged, record source/script hashes, exact counts and discrepancies.

This is a known-root host-CUDA adapter smoke, with no AP memory retrieval,
action selection, physical execution, or independent confirmation. It is not
an official AP reproduction or an equal-budget performance experiment.
