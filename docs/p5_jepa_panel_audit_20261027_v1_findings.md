# Actual three-root physical and v12 saved-array findings

Status: saved_panel_physics_audited (1168checks) and saved_panel_fidelity_audited
(1205checks), all true. Tests:19passed in0.23s. Zero env/neural calls, no model
or checkpoint decoding. Source/data hashes matched before/after both audits.
96total frames=3seeds x2roles x16frames;48exact frame pairs;906substeps=.01s.
Do not misread96frames as96per seed: each seed contributes32frames/16pairs/302steps.

## Object/agent/coverage, all fixed roots retained

| Seed | Agent path(px), rolea | Block path(px) | Block angle change(rad) | Coverage, step1=step15 | Contact points |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2026102702 | 46.500137490730005 | 0 | 0 | 0 | 0 |
| 2026102703 | 46.500137490729784 | 0 | 0 | 0.14313007581993942 | 0 |
| 2026102704 | 46.50013749072991 | 0 | 0 | 0.0702660571546306 | 0 |

All16full-public block positions/angles per role are EXACTLY constant, not merely
zero endpoint displacement. Both roles' trajectories/metrics agree exactly.
Coverage is EXACTLY constant through all15steps for each root, not improving.
Reset coverage was not returned: do not invent an initial zero. The nonzero
coverage values are existing overlap, not an action-induced gain. All saved
contact counts are zero; no contact-mediated block response is evidenced.
This panel shows agent-only motion, NOT C2/object response or control benefit.
Full per-frame trajectories/rewards/contact/coverage are in the saved JSON.

Coverage/reward consistency is independently checked numerically against stored
source-native info and public history. It is NOT an independent polygon-geometry
recomputation. Public snapshots do not serialize private contact caches.

## v12 independent global patch MSE against factual frame15

| Seed | actual | normalized_zero | negative_normalized | reverse_steps |
| --- | ---: | ---: | ---: | ---: |
| 2026102702 | 0.029574311045191957 | 0.033828127899972533 | 0.0460346590305085 | 0.031523759340368156 |
| 2026102703 | 0.0271594248332792 | 0.031295775703499354 | 0.04695091141272556 | 0.023681225286205837 |
| 2026102704 | 0.0412833249115294 | 0.0442765085686301 | 0.055346926279836506 | 0.04108635073688046 |

All15full original float32 array artifacts loaded with bounded header/shape/dtype/
finite/SHA/payload checks, no pickle, read-only mmap. Independent float64 MSE over
98304coordinates matches CUDA receipts with maximum absolute error
6.938893903907228e-18. Actual-minus-control signs: actual lower than native-mean
and negative in3/3; lower than reverse in1/3, higher than reverse in2/3. No roots
selected/replaced. No threshold or tolerance used to relabel ordering.

All3 root identities, factual paths, RGB/normalized-proprio/packet hashes,
all9consumer arrays per role, command timing/units and last-packet-only controls
match independent reconstruction. Normalizedzero is native action mean, not
rawzero. Verified attempted/completed CUDA call receipts1encoder/1proprio/4predictor
per root (3/3/12total), frame counts4/3/12per root, strict-loader/source/unchanged
frozen/eval/CUDA receipts. They remain authenticated receipts, not neural reruns.

Only factual successor labels exist; no physical control branches were executed.
Thus these numbers do NOT show counterfactual control benefit or object-model
accuracy under manipulation, and do not admit C2/P5. Official/cache-weight
provenance stays conditional; no new HF/official equivalence claim. No current
observed target may be repackaged as a prospective control target.

## Artifacts, frozen source/tests/protocol

Directory: `le-wm-main/results/p5_jepa_panel_audit_20261027_v1/`

| File | SHA256 |
| --- | --- |
| manifest_physics_frozen.json | ee7d6b5b5afb9feb5603f3c624ae637d50e272d7a6042ec54d7271cb8ff1a0c1 |
| manifest_fidelity_frozen.json | 51de60b227ce5f66bd7c86c613e8fa15c9771d5f7fbec69633e8f00efddce37c |
| physics_audit_saved_v1.json | 15b25541625555ea611a2bebb888cef72a40a006c0f7e37d14ccebbeb77eb13b |
| fidelity_audit_saved_v1.json | 09656b9a08df6437971d3128ebd98ba56ef533d855593b9c78fbfb1d8b1617c2 |

Prepared manifests freeze source/tests/protocol plus input closure. No further
edits to their sealed files. Fidelity prepare CLI requires absolute CUDA manifest/
report paths for its strict absolute-artifact parent check (relative arguments
fail closed). No existing sealed file was edited. The only actions run by agent
were fixture tests, SHA/JSON/array reads and arithmetic; all actual physics/CUDA
receipts came from main.
