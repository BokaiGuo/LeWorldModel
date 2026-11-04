# v11 JEPA constructed-feasible feedback diagnostic

The fixed protocol constructed a goal from a predeclared legal 15-command
trajectory following the common warmup. The witness was replayed exactly
before planning. This task establishes feasibility only for this constructed
goal; it does not establish feasibility for the stopped random goal or a
representative PushT task distribution. Neither planner received the witness
as an input or candidate.

Both arms completed three real native feedback decisions with RGB plus
proprioception. The selected v11 actions came from the same compiler at each
decision, and 15/10/5-step plans dispatched five physical commands each.
The initial joint XY error was 28.309. At the fixed terminal time:

| Development arm | Joint XY error | Object XY error | Wrapped angle error | Success |
|---|---:|---:|---:|---|
| v11 response | 11.336 | 11.205 | 0.06161 | yes |
| warm-start CEM | 3.520 | 2.793 | 0.00630 | yes |

Response used 1893/1916/1916 prediction tokens; CEM used 1920 per
decision. Equal token caps do not equate complete cost. On this one constructed
task CEM was better on those terminal errors; neither a general superiority
claim nor a mechanism claim follows.

The independent saved-data audit checked source/decision hashes, actual
observations, action history, carry, packet counts, final selections, dispatched
actions and physical metrics. A second arithmetic audit verified that every
scored complete chain was considered for the incumbent, that the two later
decisions re-evaluated the previous tail, and that paired abstention used the
final incumbent. A separate-process same-backend replay matched both arms'
26 saved boundaries including transitions. It is not independent neural
reproduction or a source-independence certification.

Raw records: `le-wm-main/results/pc_jepa_native_v11_feedback_dev_20261104_codex/`
(`manifest.json`, `summary.json`, `audit.json`, `incumbent_audit.json`,
`physical_replay.json`, decision records and trajectories).
The normalizer and checkpoint training membership remain unverified. No PC
gate has been passed by this diagnostic. In particular, automatic non-greedy
discovery and fair independent advantage remain open.
