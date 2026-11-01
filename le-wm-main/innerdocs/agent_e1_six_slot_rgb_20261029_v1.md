# E1 six-slot actual RGB bridge: review candidate

Fixed 2 known roots × three E1 arms × cp0/5/10 = 18 positions. The original physical report SHA is `77cd42de14e1c7d75f8e7b5b4b3d9ad58fb87e94a77c209333ec8a5fc739c5ce`, and MAIN complete physical strict receipt SHA is `6644156f48a79a30f5118d9c56c3c3a06c3bc818e44edb55e6fe8bf4c29d38d4`. Prepare rebuilds the entire native consumer and compares the entire receipt, rebuilds native prepare, binds original tasks/model/sources/artifacts and exact frame hashes, and reads the original six raw10 paid replay outputs. It requires selected raw commands and replay raw equal. No selection is based on native outcomes. Original controller receipt/source are authenticated transitively by native's prepare/consume.

The prediction is **paid raw10 whole replay**, not plan `predicted_contexts` or `model_successor`; segmented replay is separately retained with both full contexts and bitwise diagnostics. The actual controller MAIN receipt reports 6×2 latent histories and action histories bitwise equal, max ULP 0. This is a fact about that execution, not a generic equality requirement; no tolerance is used. At cp0, original task source pixels and fresh native pixels must have the same pixel SHA, and new encoder latent must be byte-identical to the original root latent in saved consume.

Run uses the frozen P2 RGB encoder/projector and its original read-only hooks/store/guard (no global monkeypatch): one encoder and projector per available image, no dedup; no new predict, AD, physics or goal encode. It persists initial roster before encoding, per-event module inputs/outputs, native seal, report and completion. Saved consumer reconstructs full ordered files, arrays, module chain, events, costs and model/source before/after guards, and calculates f64 tracking Q against each target, task-goal Q and paid-prediction versus encoded MSE. Missing checkpoints remain null, not endpoint substituted. The reported GPU caps are allocated 3 GiB/reserved 4 GiB; disk 256 MiB with 32 MiB terminal reserve and 200 MiB floor; 900s cooperative wall includes prepare verification through report fsync/hash, not the completion acknowledgement. A killed process or failed disk need not have a terminal report. Original physical/controller resources are inherited, not added to new inference counts or converted to FLOPs. No fresh-root, equal-generation-cost, equal-FLOPs, scientific or P2 mechanism claim.

Source-first MAIN review and actual execution only; no production CUDA or fullstrict was run while developing. Real saved ABI checks only (no production prepare): six raw10 replay joins and 18 available fixed cells. Fixture tests are not CUDA/model equivalence. E1 target source times may differ and are not a fairness claim.

```sh
cd /home/gbk/code/LeWorldModel/le-wm-main
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_e1_six_slot_rgb_20261029_v1.py
MOD=experiments.spectral_reachability.agent_e1_six_slot_rgb_20261029_v1
P=results/agent_e1_six_slot_rgb_20261029_v1_prepare_MAIN01.json
O=results/agent_e1_six_slot_rgb_20261029_v1_cuda_MAIN01
.venv/bin/python -B -m "$MOD" prepare --native-output results/agent_e1_six_slot_physical_20261029_v1_MAIN01 --native-report-sha 77cd42de14e1c7d75f8e7b5b4b3d9ad58fb87e94a77c209333ec8a5fc739c5ce --native-receipt results/agent_e1_six_slot_physical_20261029_v1_consume_MAIN01.json --native-receipt-sha 6644156f48a79a30f5118d9c56c3c3a06c3bc818e44edb55e6fe8bf4c29d38d4 --output "$P"
# Set PH to the actual prepare SHA printed above.
.venv/bin/python -B -m "$MOD" verify --prepared "$P" --prepare-sha "$PH"
CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m "$MOD" run --prepared "$P" --prepare-sha "$PH" --output "$O" --main-authorized
# Set RH to the actual run report SHA.
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m "$MOD" consume --rgb-output "$O" --report-sha "$RH" --output results/agent_e1_six_slot_rgb_20261029_v1_consume_MAIN01.json
```
