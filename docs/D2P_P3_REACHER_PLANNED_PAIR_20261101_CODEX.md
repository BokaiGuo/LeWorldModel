# P3 Reacher planned single-chunk exploratory pair

Predefined development comparison: same root seed 20261101, same CEM
configuration (8 samples, 2 iterations, 2 elites), same frozen Reacher model
and action normalization, fixed five-step versus dynamic five/ten/fifteen-step
execution. Dynamic thresholds are min improvement 0.01 and max disagreement
1.0; these values were used in the prior zero-action wiring probe and are not
tuned to this run's physical outcomes.

The existing exclusive planned-chunk executor is called in two fresh host CUDA
processes. It seals its protocol, checkpoint and sources, uses real CEM and
records physical actions and resource counters. The pair consumer checks
returned root identities and the complete selected raw action bank before
labeling a pair matched; otherwise it reports mismatch without pooling.
Comparisons are descriptive because separate GPU planning may differ, actual
rollout lengths may differ, and resource usage is not matched. This is not a
P3 gate, success-rate study, independent task cohort, full FLOPs measurement or
P1 admission; failed attempts remain on disk.
