# P3 Reacher feedback-cap v2 RUN_MAIN02：独立保存复核

日期：2026-11-01。仅只读检查运行报告、账本、根快照和当前来源；没有运行模型、CUDA、渲染或物理重放。此结果是单一已知根开发对照，不是独立 P1/P3 admission。

## 来源及复核范围

目录 `le-wm-main/results/p3_reacher_feedback_cap_v2_20261101_codex_RUN_MAIN02/`：

| 文件 | 本次计算 SHA-256 |
|---|---|
| report.json | `37b4fb29c07459bd91c4e478b322de305a7c404e2d3775ea06ee6a311c1bf45f` |
| protocol.json | `9a90c692bcd116ac6709f4d954b640dbc693eea2bdd640504ea2e4974c2d7c6f` |
| root_audit.json | `b9c37ab7d0bff733f742251ba2e70952bd75a9f9fcd880c50a524a433aef79cb` |
| root_snapshots.json | `b1195d45986de6b63cf0ceedeae0f157069e90cf272a74bc3e3732dd7fc10093` |
| root_snapshot_fixed.json | `26a47e87a5cfd1d10157bcd28a06a23ec3771e60106cc5c387274b7594b5397b` |
| root_snapshot_dynamic.json | `dbbf4633aed1211c0ddccb0ba031a4d857275e2ad094aec96b8ca458f19b9da8` |
| fixed.json | `0251b37396dbd9b3963554a2a877a45117238df5c104821f5c39239db0306c17` |
| dynamic.json | `50abbb135961e3c19a409bf59a0e212579b6c1d38c61cb70ee7f1f7a94c09a4c` |
| runtime_manifest.json | `453fd383684b44b385ae72d6345a1ef0bb6d5ce8d2d4ea1287f96ff36d5b50d5` |

报告声明的 root audit、汇总及两单臂 snapshot 哈希均与文件一致；单臂文件与报告对应 row 逐结构相等。protocol 所列 23 个来源和 runtime manifest 所列 3137 个项目 Python 文件，当前文件哈希与记录均一致。runtime manifest 只覆盖规定时点已加载的项目文件，并不完整覆盖晚加载和第三方依赖。report 没有将 protocol、runtime manifest、单臂文件独立顶层哈希封存；上表是本次读取的指纹，不是外部签名。

报告记录模型前后摘要一致，为 `3a9c7f7cf8ffa7c325072cb4756b94445439a94eec69b6da88d5f63a77784000`；本次未加载模型复算。源码使用 cooperative lock 包裹实验，但保存文件不含可独立验证的全程持锁事件。

## 根状态

两侧完整保存 snapshot 逐结构相等，包括物理积分状态及规格、qpos、qvel、physics state、模型二进制摘要、目标 qpos、target marker 和压缩 goal RGB。各自的 root、goal、history RGB 与 warmup 动作哈希也相同；汇总中的两行分别与单臂快照一致，root audit 与 fixed 行相应字段一致。解压保存的 goal RGB 得到 150528 字节，其 SHA 与快照和 goal 摘要一致。

两臂根 qpos 均为 `[-1.0713275505406263, -1.5631624782400513]`，目标 qpos 均为 `[-0.16484263736695937, -2.191093943030594]`；root 几何 qpos 和目标与各自 snapshot 精确一致。这是对保存状态的独立比较，不是独立物理重放；root/history 原始图像和 warmup 原始动作未保存在这些文件中，只能比较其摘要，不能从内容重算。

## 逐步物理几何

两臂各 15 次 env.step 尝试、15 个返回动作，均 `completed15`，无 terminated/truncated。各实际动作与其 chunk 的 `candidate_raw` 对应前缀逐项一致。每臂 root 加 15 步共 16 条几何记录；逐条独立从保存 qpos 和目标向量重算未 wrap 的差、L2 和最大绝对差，均与记录一致，且目标一直与该臂 root snapshot 相同。

以下为重算的原始关节坐标差 L2，单位为该环境 qpos 坐标的弧度；这是描述性几何，不是环境原生成功门：

| 步数 | fixed | dynamic |
|---:|---:|---:|
| 0 | 1.102730 | 1.102730 |
| 1 | 1.181866 | 1.181866 |
| 2 | 1.257352 | 1.257352 |
| 3 | 1.316621 | 1.316621 |
| 4 | 1.327611 | 1.327611 |
| 5 | 1.320916 | 1.320916 |
| 6 | 1.320644 | 1.313066 |
| 7 | 1.336740 | 1.352167 |
| 8 | 1.367524 | 1.372233 |
| 9 | 1.287188 | 1.418279 |
| 10 | 1.192656 | 1.432477 |
| 11 | 1.180745 | 1.476891 |
| 12 | 1.190389 | 1.468999 |
| 13 | 1.212001 | 1.502776 |
| 14 | 1.317367 | 1.588061 |
| 15 | 1.494039 | 1.687844 |

按此未 wrap 指标，两臂末步均大于根值；fixed 末步小于 dynamic。这仅是本根、此指标的观测，不证明一般性能排序。实际 task 为 `hard`，goal wrapper 给其附加 target_qpos，但该任务没有这个 qpos 目标的原生成功门；全部记录的 `native_success`、`native_termination`、`native_qpos_threshold` 均为 null，不能将 null 说成失败或成功。两臂记录的 reward 总和均为 0，也不能单独作为成功判定。

## 决策和成本

fixed 在 0、5、10 重规划并各执行 5 步；dynamic 在 0 重规划，依据模型内信号接受 5→10→15 扩展并执行 15 步。dynamic 预测成本依次为 2.1697850、2.0384405、1.9789457；两次 disagreement 为 0.6332406、0.5415369。它们不是实际物理几何收益。

独立逐事件累计调用数和 samples，均等于每臂账本，事件状态均为 attempted，各前缀均未超共同上限：

| 预算轴 | fixed | dynamic | 每臂 cap |
|---|---:|---:|---:|
| predict_calls | 18 | 14 | 48 |
| predict_samples | 144 | 56 | 192 |
| encoder_calls | 18 | 6 | 48 |
| encoder_samples | 36 | 12 | 192 |
| action_encoder_calls | 27 | 17 | 64 |
| action_encoder_samples | 153 | 59 | 256 |

dynamic 的 14 次 predict 含 planning 6 次、scheduling 8 次，不可将后者重复相加。共同上限和本次相同 15 步物理时域不意味着等实际计算；入口 attempted 也不是逐调用成功证明或 GPU 运行时间/能耗测量。

## 与 MAIN01 的关系及结论边界

v2 RUN_MAIN01 报告 SHA `eb2137c7e6582a6488b47927e27e6ac11a1b3328990e5e6f0d26f50300cc1525`，两臂均在 root geometry 因错误假定 task 类型而失败，零规划动作；它的 root audit 和双侧快照摘要与 RUN_MAIN02 相同。MAIN02 是修复后的重跑，不是把 MAIN01 失败改写为成功。

另与**旧版 v1 的成功 RUN_MAIN01** 比较：root audit SHA 相同，fixed 和 dynamic 各自保存的 15 个实际动作、RGB checkpoint 摘要和完整预算事件账本逐项相同。因此 v2 几何为这条已有执行提供新记录和可复核性，不应计作独立根、独立重复或额外有效样本。也不能由 RGB 哈希独立复原图像、或由报告独立验证动作执行与物理状态的因果关系；需单独物理重放才能检验后者。两臂无独立根和重复种子、计算实际消耗不同，P1/P3 仍未 admission。
