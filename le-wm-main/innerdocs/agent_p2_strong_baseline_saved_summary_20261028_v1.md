# Saved-only 强基线汇总

本版本只消费三份已封存本地资产：kinodynamic matched-domain report、no-composition consume、sampling physical report。不会加载 torch/Gym，不运行神经、物理、网络或训练。所有作者报告值和本地运行值分别存放；没有明确绑定的作者原始报告时，作者栏为 `unavailable`，绝不以本地值代替。

## 已绑定本地资产

- kinodynamic：`results/agent_kinodynamic_matched_domain_20261028_v1_cuda_MAIN01/report.json`，SHA `d4976aeca357b5324c74be8b2b7e3bfc254e82b834b3acfe1b0ab90a1ef4614c`；2 source slots，matched-domain H2 beam20x6，finite D2P ball，资源288 predict/API等，`equal_total_compute=false`。
- no-composition：`results/p1_moving_no_composition_baseline_cuda_20261027_v1_consume_main01.json`，SHA `a05aa97f6c2688d7ae7a1fd8b961a23ccba1f4e45b14d95159cca63c5ba23167`；2 rows，H2 spectral packet mapped，资源含176 predict/AE、J/SVD/VJP/solve，`equal_total_compute_claim=false`。
- sampling：`results/p1_sampling_physical_20261027_v1_main01/report.json`，SHA `c3eef328bee57ce5d42b25fcf5a9935af2d2282c9de3f8caf46f2b35b53a2ef6`；2 roots×CEM/IID，40 public steps、8 resets、44 renders，`equal_compute=false`，真实 physical calibration。

三者协议并不相同，汇总中的 pairwise comparisons 全部是 `protocol_mismatch_no_direct_comparison`。不会把模型成本、physical final state、sampling official evaluation、作者数值放进同一差值。

## 作者值与本地值

CLI 可选 `--author-report PATH` 显式绑定作者原始报告；保存为独立 `author` 栏并记录 SHA。它不会覆盖 `local`。若不提供，输出 `author.status=unavailable`。作者值只有在后续独立确认任务、根/任务/单位/指标/时间窗/协议完全一致时才可比较；否则仍只分栏。

## 命令与验证

```bash
P2_BASELINE_MODULE=experiments.spectral_reachability.agent_p2_strong_baseline_saved_summary_20261028_v1
P2_BASELINE_OUT=results/agent_p2_strong_baseline_saved_summary_20261028_v1_MAIN01.json
.venv/bin/python -B -m "$P2_BASELINE_MODULE" build --output "$P2_BASELINE_OUT"
P2_BASELINE_SHA=$(sha256sum "$P2_BASELINE_OUT" | cut -d' ' -f1)
.venv/bin/python -B -m "$P2_BASELINE_MODULE" verify --summary "$P2_BASELINE_OUT" --sha "$P2_BASELINE_SHA"
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p2_strong_baseline_saved_summary_20261028_v1.py
```

该汇总不会宣称 P1/P2 admission、等算力、同 horizon、同 search coverage 或科学收益。sampling 的 physical 结果与另外两个 model-only/plan-only 结果即使共享 source slots，也不能直接作因果比较。
