# P2 intersection v2: saved-plan preparation slice

This new slice is **not a physical executor or an independent physical-result
consumer**. It prepares all six fixed slots using the actual v2 strict receipt
and source report, verifies the original prepare through the producer API, and
independently recomputes joint30 ball, float64 tanh to float32 raw15 and all
three latent/action history shifts. It retains raw bit identity, producer
accounts, original contexts, goal/Q, native root metadata and explicit nulls.
Receipt authentication inherits MAIN's strict receipt SHA; it does not rerun
producer full consume or independently authenticate NN/AD.

Fixed order is root0 joint/composition/grasp, root1 grasp/composition/joint;
two known off roots only, no outcome selection, no additional model/physics,
no equal-FLOPs or scientific admission. Prior producer compute accounts are
kept intact rather than relabeled as physical costs. The already reviewed
common-resource physical executor is **not** reused as an own-schema adapter:
its required producer/plan schemas differ. A new native driver and complete
saved event/PNG/state/reward verifier remain required before any physical run.

RAM tests only:

```
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p2_intersection_physical_prepare_20261029_v1.py
```

MAIN may review before using prepare; `verify` takes the same arguments plus
`--prepare-sha` and reconstructs the entire saved preparation. This prepare
does not authorize or execute physics.

```
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_intersection_physical_prepare_20261029_v1 prepare --receipt RECEIPT --receipt-sha RECEIPT_SHA --producer-output PRODUCER_OUTPUT --report-sha REPORT_SHA --prepared results/agent_p2_intersection_physical_prepare_20261029_v1_MAIN01.json
```
