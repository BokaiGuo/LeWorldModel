# Known-root four-arm action and accounting boundary

CPU-only source audit of `codex_four_arm_host_cuda_20261030_PREP01.json`,
file SHA-256 `b8044f55943b34c5a56a0b5db99cfc5105c5180cdf0f2139926fbb9c82d2b5ec`.
This is not a GPU run, physical execution, training provenance certification,
or permission to run either. Slot 0 root hash is
`6812ab89db47612390424790b001921168d0102dde47b39f07669bfeba6cd0a1`.

## Action chain

1. The task binding contains action-domain SHA
   `a6136cdc5e6a7330ec9879c1a117c323db5af30682cf7463f5e6d8298096c055`.
   `agent_p3_runtime_contract_20261028_v1.py:90-105` validates it against
   its explicit domain: raw bounds [-1,1], five steps, two dimensions, one
   float32 subtract/divide with no /100 or clipping. Source SHA
   `72f076106fb8f6c2aa73386c06fecb912f033e81a5108aa1f36885bd0342c2bc`.
2. `pusht_d2p_packet_contract.py:30-40,50-61,75-87,90-115` validates a
   whole finite [5,2] packet without clipping, defines five separate
   `env.step` calls, and flattens time-major after `(raw-mean)/std`. SHA
   `eda631348e929b45591807dac0adcbb72d10d84b7d48a7a558a492eab9c04aef`.
   The four-arm entry clamps the *generated proposal* to this raw box; that
   is not permission for an executor to silently clip a saved action.
3. Installed `stable_worldmodel/envs/pusht/env.py:29-49,72-75,312-330`
   declares relative=True by default, Box [-1,1] shape (2,), action_scale
   100, and maps each command to `agent.position + 100 * action` before PD
   control and simulation steps. Its target clip is commented out. SHA
   `c6d2104adc368ff1c352edc0e35ac74da9f07fda3a748ddbef0ba6b7a2d1daa2`.
   Bounded commands do not prove bounded agent state or visible RGB.
4. The fixed entry's raw `[N,3,5,2]` candidate is three ordered packets;
   each modeled token is standardized with the task's two coordinates and
   flattened to 10, while the two input history tokens are already normalized.
   A full hypothetical physical H3 path would involve 15 public steps, not
   three. The entry itself has no physical executor and authorizes none.

Important conflict: `experiments/run_pusht_counterfactual.py:38-45,206`
(SHA `0a4eb4592951139e596b35becede17d175df7b48a2fde16679ab0aac7c4dce34`)
contains a different historical live-action `/100` normalizer. The present
entry follows the explicitly bound P3/P2 packet contract, not that function.
This resolves the *local engineering convention*, not independent authority
for the checkpoint's training data, split, or true training normalizer. Those
remain unverified. Source hashes of the environment and packet contract were
not part of the existing four-arm prepare's direct source list; this document
does not retroactively bind them or establish runtime environment identity.

## Per-arm accounting

`codex_four_arm_durable_20261030.py:62-99` reserves before dispatch, records
intent and then return/failure, blocks excess before callback, and fails closed
on nested dispatch. `codex_four_arm_leaf_suffix_20261030.py:70-110` explicitly
meters outer encode, action_encoder, predict, and backward calls; it does not
intercept hidden internal module calls. `codex_four_arm_final_replay_20261030.py`
meters each candidate replay marker, leaf calls, and scalar selection.
Structured initialization also charges one Jacobian marker, 192 gradient/VJP
rows and one solve; immediate-progress screening charges three one-token
replays and selections. All arms have the same fixed caps, not equal usage:

| Arm | action-encode / predict calls | replay / selection | backward rows | solve |
| --- | ---: | ---: | ---: | ---: |
| structured | 30 | 9 | 192 | 1 |
| native / random | 27 | 9 | 0 | 0 |
| immediate_progress | 30 | 12 | 0 | 0 |

These are source-derived complete-run counts for the fixed schedule, not
measurements. Prediction batch is one, so predict samples match calls.
Counts include both CEM generations and a fresh final-generation replay.
Wall and peak-memory checks are cooperative; they are not hard preemption.
Input context/goal were produced upstream and their collection and encoding
are not charged to the individual arms, despite the existing contract's
`including_attributable_setup` label. Loading, random sampling, tensor
arithmetic, matrix multiplications, I/O and internal model operations are not
completely accounted for. Physical axis is zero and has no hook here.
Thus neither full standalone cost nor matched total compute is demonstrated.
Source subset hashing and `closed_unverified` journal status are not complete
source closure or proof of hook coverage. Partial failures need not preserve
individual scores already computed inside an incomplete generation.

Before an actual comparison, bind and independently check the executing
environment/wrappers, exact source and action bytes, normalized history and
checkpoint training provenance; instrument attributable setup and all required
cost axes; and separately authorize any guarded physical protocol. Existing P1
and RGB/canvas gates remain unchanged.
