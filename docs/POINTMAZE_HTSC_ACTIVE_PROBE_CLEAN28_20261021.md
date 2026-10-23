# PointMaze HTSC active-probe clean28：协议修复与机制结果

日期：2026-10-21

## 目的

本批次检验文献启发的闭环修复：

```text
RGB short probe -> re-observe -> local action-effect update -> rerank
```

比较五个 RGB-only arm：`visual_dynamics`、`effect_adapter_rgb`、
`effect_adapter_rgb_htsc`、`effect_adapter_rgb_online_verified_local` 和
`effect_adapter_rgb_htsc_consistency`。所有 arm 使用相同 frozen JEPA/DINOv2
checkpoint、相同 candidate bank 和相同总物理 block budget。

## 运行与数据契约

正式数据集只有 `data/point_maze_tail1800_1899/obses/episode_1800.pth` 到
`episode_1899.pth`。此前准备的跨数据集 root 列表包含 `episode_009.pth`、
`episode_1901.pth` 等不存在的文件，第一次启动在数据读取处失败；该失败不计为
科学结果。随后改为使用可用且未被既有运行使用的 28 个 root：

- visual calibration：1840--1847（8 roots）；
- gate calibration：1848--1855（8 roots）；
- held-out test：1856--1867（12 roots）；
- 额外排除旧 active-probe 使用的 1868--1899；
- 排除 V6.9 renderer chart roots 1800--1839。

完整 root 与来源 manifest 摘要见：
`results/pointmaze_htsc_active_probe_strict_20261021_clean28_cuda/root_provenance.json`。

运行前通过了真实 CUDA recurrent preflight：RTX 5060 Ti、CUDA 13.0、PyTorch
`2.12.0+cu130`、cuDNN `92000`。作业通过
`experiments/run_with_torch_cudnn.sh` 启动，未使用 CPU fallback。

## 结果

held-out 结果如下；最终 active probe 配置由 gate calibration 冻结，不能读取
held-out outcome：

| arm | success | mean final distance | mean charged blocks |
|---|---:|---:|---:|
| visual dynamics | 0.5833 | 0.37695 | 18.83 |
| RGB effect adapter | 0.5000 | 0.42903 | 19.25 |
| RGB + HTSC | 0.5833 | 0.38504 | 19.25 |
| RGB + HTSC consistency | 0.5833 | 0.37604 | 19.25 |
| online verified local probe | 0.3333 | 1.13748 | 22.08 |

相对 HTSC fallback，online verified local probe 的 paired 差异为：

- final distance：`+0.75245`（越小越好），bootstrap 95% CI
  `[+0.34734, +1.20006]`；
- success：`-0.25`，bootstrap 95% CI `[-0.50, 0.00]`；
- paired roots：12。

probe schedule 在 calibration 中被拒绝：6-direction candidate 相对 HTSC 的
success delta 为 `-0.50`，4-direction candidate 为 `-0.625`。因此最终状态是
`calibration_rejected_diagnostic_only`，不是 deployment authorization。

## 机制定位

active arm 确实执行了 probe，平均每个 held-out root 执行 6 个 probe blocks，
但：

- HTSC non-release：55 次；
- support-failure non-release：55 次；
- local verified release：0 次；
- local rerank override：0 次；
- `repair_mechanism_exercised=false`。

所以本批次不能证明 local ranker 本身已被否证，也不能声称 active probing 有收益。
它把失败定位为更具体的两个候选机制：

1. 探针产生的局部 chart 没有进入 action-support gate 的可接受区域；或
2. 探针消耗了物理 block budget 并改变了起始状态，但没有产生足够可靠的局部
   transfer 证据。

在当前协议下，`probe -> local update` 还没有形成一次真实的 local rerank，因而
不能把“probe 后性能下降”解释成 within-support action-effect ranking 已被严格
检验。下一轮若继续，必须单独记录 support gate 的失败原因、probe 后 chart 的
有效样本数和 observed/predicted effect residual，并设置一个不改变物理预算的
短脉冲验证对照。

## 代码与协议修复

本批次同步完成：

1. active-probe 的 GPU preflight 增加 `--recurrent-smoke`，在运行前检测真正的
   cuDNN recurrent kernel，避免系统 cuDNN 9.13 覆盖 PyTorch bundled cuDNN 9.20；
2. root provenance 自动记录来源 manifest digest、旧设计排除集和当前 split；
3. runner 拒绝与显式历史 runtime roots 重叠的 split；
4. claim boundary 改为读取实际 held-out root 数量，不再硬编码 12；
5. 第一次错误数据路径启动保留为协议诊断，不覆盖 clean28 正式 artifact。

正式机器可复现报告：

`results/pointmaze_htsc_active_probe_strict_20261021_clean28_cuda/report.json`

## 结论边界

本批次支持的结论是：在 frozen RGB PointMaze、固定 action bank、等物理预算和
root-disjoint calibration/test 协议下，校准冻结的 observed-effect probe 没有
获得部署授权，且其 held-out 运行没有触发任何 verified local rerank；HTSC
fallback 与 visual baseline 的收益接近，而 active probe 明显更差。

本批次不支持：普适 world model、物理因果 latent、conditional coverage、
universal transfer certificate，或 local observed-effect repair 已经成功。
