# P3 Reacher feedback-cap RUN_MAIN01：保存产物独立复核

日期：2026-11-01。复核只读取保存文件与当前源码，未执行模型、CUDA 或物理环境。该单已知根开发对照不构成 P1 或 P3 admission，也不证明任务成功。

## 来源与根状态

运行目录 `le-wm-main/results/p3_reacher_feedback_cap_20261101_codex_RUN_MAIN01/`。本次读取计算的 SHA-256：

| 文件 | SHA-256 |
|---|---|
| report.json | `16cf54606a8c0d3cbec91d2026e9ba14aee5c6759a16d50a0b7ca0d058ffd663` |
| protocol.json | `8f70b93e0aed07b31f8cf339705b652915602241436dae528d27446737866833` |
| root_audit.json | `b9c37ab7d0bff733f742251ba2e70952bd75a9f9fcd880c50a524a433aef79cb` |
| runtime_manifest.json | `a34a24ffe57193be0508d14feaad0b660d1428e550c6a531c60cbfebf7011315` |
| fixed.json | `5e21435eb49f153998ccaf0aa1632470b1e8f419711bb256acdab46b4c8d3e40` |
| dynamic.json | `d429c8dc04304eac9352bb58e7abbef47b8289f32ce5be9dfabbce43c2b15250` |

`report.root_audit_sha256` 与实际 root audit 一致；两个单臂 JSON 与 report 中相应 row 逐结构完全相等。protocol 所列 18 个来源、runtime manifest 所列 3137 个已加载项目 Python 文件，当前文件均存在且 SHA 与各自记录一致。这是当前文件对记录的比对，不是外部签名；runtime manifest 自述不覆盖晚加载和所有第三方模块。报告未对 protocol、runtime manifest 和单臂文件提供独立顶层哈希封存，本表是本次复核时的指纹。

运行源码中的 `audit_roots` 要求两次独立 reset/warmup 的物理积分状态及其规格、模型二进制哈希、目标 qpos/marker、goal RGB 相等，也要求 root/history/goal RGB 和 warmup 动作逐项相等；每臂执行前还检查实际环境状态。保存的 root audit 只含左侧 snapshot 和摘要，并无右侧 snapshot：因此可核对运行时的检查设计和报告中的存档绑定，**不能仅凭保存文件独立重算两臂根状态相等**。

运行源码采用项目 cooperative GPU lock，锁覆盖整个实验回调；锁实现也在来源哈希内。但存档没有持锁时间线，不能独立证明运行全程实际持锁。

## 实际动作和 15 步结果

报告 `complete`，fixed 和 dynamic 均 `completed15`：各记录 15 次 env.step 尝试、15 个返回动作，无 terminated/truncated。每个 chunk 的已执行动作，与对应保存的 `candidate_raw` 从头取 `actual_returned` 条逐项完全一致。

| 臂 | 重规划边界 | 每次实际返回动作数 | 保存的 reward 总和 |
|---|---|---|---:|
| fixed | 0、5、10 | 5、5、5 | 0 |
| dynamic | 0 | 15 | 0 |

两臂首个完整候选和实际前 5 步动作相同；5 步 checkpoint RGB 哈希相同，10、15 步不同。checkpoint 只保存 RGB 哈希而非图像，不能从保存图像独立重算或判断视觉状态。reward 全零只描述记录的奖励，不是成功或失败判定；报告未保存可复算的末步目标几何误差或成功指标，不能由本报告推断任务表现或真实目标改善。

dynamic 在同一次规划中接受 5→10→15 两次扩展。对应保存的模型评分从 2.1697850→2.0384405→1.9789457，两个 disagreement 为 0.6332406、0.5415369；它们是模型内信号，不是物理收益。

## 模型入口事件与预算

独立按 `budget.events` 的 `kind` 逐事件统计调用次数和 samples，与每臂 `attempted` 六轴均完全一致；事件状态均为 `attempted`，且每个前缀累计均不超共同 cap：

| 预算轴 | fixed | dynamic | 每臂上限 |
|---|---:|---:|---:|
| predict_calls | 18 | 14 | 48 |
| predict_samples | 144 | 56 | 192 |
| encoder_calls | 18 | 6 | 48 |
| encoder_samples | 36 | 12 | 192 |
| action_encoder_calls | 27 | 17 | 64 |
| action_encoder_samples | 153 | 59 | 256 |

fixed 的 18 次 predict 都属 planning；dynamic 的 14 次包括 planning 6 次及 scheduling 8 次，后者不能再加到 14 次之外。事件是入口前计费记录，不是逐调用返回成功证明、GPU kernel 级资源测量或完整耗时/能耗计量。共同预设上限和相同物理时域成立于本次记录；**实际消耗不相等**，不能称等计算量对照。

报告记录模型运行前后摘要一致，均为 `3a9c7f7cf8ffa7c325072cb4756b94445439a94eec69b6da88d5f63a77784000`；这是 runner 自报的状态摘要，本次未重新加载模型独立复算。来源中 checkpoint weights 文件的当前 SHA 则与 protocol 所列一致。

## 解释边界

这是单个已知根、单次 fixed 与 dynamic 的开发性比较；无独立根或重复种子，不能推广，也不能把本次 15 步完成解释为 P1 通过或 P3 获准。两臂反馈频率、实际计算消耗不同，且动态扩展是模型评分驱动；没有保存可独立复算的真实末端任务指标，不能据此声称 dynamic 优于或劣于 fixed。若需更强结论，应预先固定真实任务指标、保存可复算状态与图像、保留双侧根快照，并在独立根/种子下复现。
