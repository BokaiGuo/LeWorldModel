# v23 PointMaze live-feedback development entry point

`experiments.pc_v23_pointmaze_live_feedback` uses the durable v3 RGB
dispatch journal and the v23 response/shared-CEM adapter. It requires a
separate, predeclared JSON task contract:

```json
{
  "scope": "development",
  "root_id": 1,
  "start": [0.0, 0.0],
  "goal_sha256": "<digest of the existing frozen goal RGB>",
  "output_name": "pc_v23_unique_experiment_name"
}
```

The displayed start and root are schema examples, not an authorized or
validated task. Before execution, verify the chosen start is physically
valid, not in any stopped panel, and satisfies a separately frozen
feasibility and stopping protocol. The entry point only rejects the known
`(0.85, 0.85)` v22 stopped root; it does **not** certify other roots or
source independence. `output_name` is restricted to a single filename under
the results directory; preexisting outputs are rejected. The contract and
declared code dependencies are hashed before execution and checked again
afterward.

Both arms receive RGB feedback and their own previous action tail. Each has a
1920 prediction-token cap per decision, with remaining horizons 15/10/5.
The shared optimizer does not itself certify total compute, feasible tasks,
physical success, or PC4. The actual trace and the dispatch journal must be
audited after a run. No experiment is authorized merely by the existence of
this contract schema.
