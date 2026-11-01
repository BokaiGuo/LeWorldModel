# 已见 root0：共同节点 cap 的 structured / CEM 收据复核

独立 CPU 保存记录审计；不运行神经网络或物理环境。两臂同一 cp10 目标
RGB/latent、root 和 frozen 模型，初始实际状态及 RGB 像素一致；归一化、
投影、决策边界和每包长度一致。两份运行合同的完整内容、实现和实际
预算使用**并不相同**：此处的“共同”仅指这些匹配项及逐节点 cap。

| 指标 | structured | variable CEM |
|---|---:|---:|
| 首次原生成功控制步 | 13 | 17 |
| 启动的决策边界 | 0、5、10 | 0、5、10、15 |
| RGB 编码次数 | 3 | 4 |
| predict / action-encode 账面预留 | 252 / 252 | 775 / 775 |
| replay / selection 账面预留 | 42 / 42 | 548 / 548 |
| backward / VJP 账面预留 | 624 / 624 | 0 / 0 |
| Jacobian / SVD 账面预留 | 3 / 3 | 0 / 0 |

上限为每节点 predict/action-encode/predict-samples 各 195，
replay/selection 各 193，backward/VJP 各 240，Jacobian/SVD 各 1。
两臂早停不同，未启动节点没有填零；structured 的优化器每节点
仍在 H3 工作，反馈选择使用 H3/H2/H1/H1，CEM 则按 H3/H2/H1/H1
规划。两方法的 replay/selection 事件实现也不同。共同 cap 不等于
相同实际使用、FLOPs、耗时或能耗；视觉、共享准备、模型加载和
环境开销不包含在节点账面汇总中，不能把不同轴直接相加。

审计重新核对两报告和 prepare SHA、5298 个去重来源、两臂逐节点
ledger 哈希链、合同和 intent/return、caps 及预留总数；检查 CEM
候选/精英/winner 重算和报告锚定日志，检查 structured 初次选择、
反馈 carry 与实际 RGB、计划和包封存；检查逐步命令、返回顺序、
所有控制期 RGB 文件与像素哈希、原生首次成功。未重新计算神经
模型输出或物理轨迹；ledger 保持 `closed_unverified`，不能声称
GPU 内部调用覆盖或真实算力认证。

这些均为已见根上的探索性成果，structured cp10 prepare 还绑定了已
完成的 CEM 报告；不是盲态实验、独立确认或 P1 准入。不能把旧
structured `RUN_CODEX01` 的不同目标结果混入。

可复算收据：
`le-wm-main/results/p1_common_cp10_pair_audit_20261101_leibniz_RUN01.json`，
SHA-256 `cfc93c70012bb6a310dc2df522d4047b8c8321b61c580202aab23c3b57adec49`。
7 项测试通过，包括目标、模型、预算、归一化、实际动作和账本篡改。

```sh
cd /home/gbk/code/LeWorldModel/le-wm-main
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_common_cp10_pair_audit_20261101_leibniz
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_common_cp10_pair_audit_20261101_leibniz.py
```
