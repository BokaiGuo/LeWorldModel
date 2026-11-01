# P3 H3 真实上下文后缀物理推进

已有 52 条真实上下文重规划已完成主机 CUDA 生成；本轮开始把
固定新后缀在 cp15 真实状态和 RGB 守卫后实际派发。第 0、1 两个
固定 batch 已完成，每批 6 条 old/new 配对，共 12 条；两批均通过
saved semantics 检查，`recorded_not_admitted`，原生成功 0/12。
每条动作均在 manifest 中固定，不按结果挑选。剩余 batch 继续按
同一合同独立执行。

本轮实际证据只说明新后缀的物理执行链和 cp15 接线可以工作，不能
把 12 条当独立任务，也不能宣称 P3 调度收益、P1 机制或公平基线
胜出。结果仍来自已知根；删失、早停和不完整批次不补值。

- Manifest：`le-wm-main/results/p3_h3_real_context_physical_20261031_manifest_RUN01.json`
  SHA-256 `2912800417ace34400c7f858a7d971d2427b2e094319fefd341d6f9721e1f5ce`
- Batch 0 RUN02：prepare `0a69e40ad7623d3f1877d3c5850385cafb8cb6fa962c3ee21665358d53816918`，
  report `6c2c612f98a1c46976f3637c982dd210bc439de1fe34892b6261ba3d7148218d`
- Batch 1 RUN02：prepare `2c7bd8be99a5eb6476145442e680dc79df530521ba4d3d3a0973a62ebb78686f`，
  report `dfce6f83de767fd24d52d03080084a58f4a68c1f91aed0e9758d41246ef0f44f`

两个旧 RUN01 目录已有历史产物，程序拒绝覆盖；未删除、未混用。
