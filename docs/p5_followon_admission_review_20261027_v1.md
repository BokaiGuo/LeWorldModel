# P5 前瞻 C1 执行器与后续阶段准入复核

日期：2026-10-27。独占新增本文，不修改总账、源码、测试、旧报告或 sealed 字节。
本文是一次有边界的只读复核，不是新的阶段 admission consumer 或执行授权。
采用 documentation-and-adrs 的证据/决策边界记录方式：工程前提、实际运行与科学
准入分别陈述，避免把已完成的输入链工作再列为“尚未实现”，也不以此代替机制收益。

## 优先结论：executor v2 的 close 修复充分

复核文件：`le-wm-main/experiments/spectral_reachability/p5_native_c1_executor_20261027_v2.py`。
源码 SHA256：`6bdac8c89187c35e5bc789b9235ba4831c3182d283b3b3fdbe5ae2ca9fcf1e02`。
相对 v1 delta、完整关联控制流及新增测试已只读核对。对此前已报告的
close-budget 缺陷，**修复充分，未发现这项修复的新 ready-binding 阻断问题**。

1. `FinalJournal.append` 强制传递 `final=True`，因此 unchanged `base.operation`
   的 close intent 和 returned 都使用预留终结预算，不再受普通 payload/node 门阻断。
2. `close_receipt` 分开保存 call_attempted、call_completed、return_durable；只有
   close 调用返回且 returned journal 落盘，才记 completed。返回日志失败不能冒充完成。
3. finally 先保存 original_error；close 异常和 close_failed 日志写入异常分别留存。
   已有原始异常继续抛出，不被后续 close/失败日志错误替换。
4. close 状态为 failed、required 或 construction_attempted 时设置 global_error，
   后续 arms 保留 not_run，不再构造新环境。constructor 内保存 env 引用，亦修复
   “构造已返回但 returned journal 失败，外层拿不到 env”这一清理漏洞。
5. 成功路径仍需全部 11 个 prefix byte gates、16 帧和每 arm 原生调用 census。
   本次 delta 未改变 native_constructor、raw action、seed、goal、prefix 校验或指标。
   bridge runner_sources 仍要求 THIS v2 的精确路径/SHA，不接受 v1 收据替代。

本次独立实际运行：

```sh
cd /home/gbk/code/LeWorldModel/le-wm-main
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p5_native_c1_executor_20261027_v2.py
```

结果 **29 passed（1.82s）**。均为 fake-env、纯数值和 I/O 接口测试，没有神经前向、
AD 或真实 physics。回归覆盖普通预算耗尽仍 close、close intent/returned 日志失败、
原错保留和后续分支阻断。通过不等于实际物理执行或任务成功。

边界：预留预算不是无限磁盘或 I/O 成功保证；若 final 写入或进程本身失败，只能
保留已 durable 的日志/数组前缀，不能假定完整 terminal report。本文不宣称穷尽了
所有异常路径。nonce 不要求内部实现，按 main 外部唯一执行控制；不复用输出目录。

## 当前 C1 证据和必须重新绑定的执行前提

本次直接读到的旧 compiler 收据为
`le-wm-main/results/p5_prospective_c1_cuda_compiler_20261027_v1_cuda_main_v1/compiler_receipt.json`：
status=selected_packet_replayed，predictor13、JVP10、autograd_grad20、encoder1/
4frames、proprio1，attempted=completed；error=null、finalization_errors=[]、
source_check_errors=[]、model_before=model_after，selected_replay_exact=true，
partial_work_unknown=false，ridge 被选，P5/C2 均 false。
这是 main 的已有 HOST CUDA 结果，不是本次重新执行。

独立保存数组审计
`le-wm-main/results/p5_prospective_ridge_array_audit_main_20261027_v1/cuda_v1_audit.json`
为 saved_prospective_ridge_consistent。它支持旧版本 ridge/输入/动作/replay 的保存
数组算术一致性，不是 FD、真实物理控制或新 executor 绑定证据。

executor v2 source manifest 当前为 prepared_executor_source_no_branch_authority。
新独立 goal 已存在，不再列为缺失；旧 CUDA 成功也不应再写成“从未编译”。
但旧 ready protocol 绑定旧 executor，必须区分下列前提：

| 执行前提 | 本复核边界与 main 应核对的产物 |
| --- | --- |
| 新 authority 链 | 新 bridge prepare 明确绑定 v2 source SHA、原独立 goal、冻结 compiler；新 compiler prepare 保留其 canonical bridge digest。不得改旧 protocol 或追认旧 receipt |
| 新 CUDA 收据 | main 按新 prepare 独占运行并核验全部计数、八类数组、源/权重不变、raw↔normalized 映射和精确 selected replay；旧成功收据不替代此项 |
| 冻结先于执行 | bridge freeze 复现该次 prepare/receipt，落盘 selected_packet、all_branches、freeze_receipt；executor ready 再验证相同源/输入/动作与全部 artifact hashes |
| 三臂物理与评价 | main 唯一执行 compiled/nominal/factual_reference；每臂 fresh env、seed-before-reset、11 帧 byte gate、15 controls/151 substeps/16 observations/close1，失败与未运行槽全保留。物体 XY/角度/coverage 与 agent 诊断分开 |
| 新 chart 的数值诊断 | 同模型 FD/action-scale/replay 不能用旧 episode 的 v8 FD 跨输入追认。当前 native FD bound manifest 只读到 prepared_for_main_CUDA_FD，本文不把 prepare 算作 FD 完成，也不新造误差阈值 |

上表列验收条件，不声称后来运行不存在，也不由本文发起任何重跑。main 并行产物若
随后出现，须按其新绑定单独复核。即使三臂全部完成，仍只是固定已知根的前瞻
counterfactual C1 接线与执行证据，不自动证明完整 P0/P1/C2/P5。

## 后续 P 阶段仍不可省略的前置条件

按当前 closure audit 与 parallel progress 的原合同核对，不提高或降低门槛。
下列“缺”指科学准入证据尚未由所查合同建立，不是否认已有实现或校准结果。

| 阶段 | 已有证据必须保留 | 仍需满足，不能被当前 C1 替代 |
| --- | --- | --- |
| P0 / 新模型新任务前提 | 原生冻结 loader、输入/动作归一化、CUDA/JVP/ridge/replay 已有局部证据；新 goal 来源独立 | 各新任务/chart 的 source/weights/context/action scale、同模型 FD 和完整执行合同须各自绑定；严格前置与强基线资源单位不能沿用旧 passed 标志 |
| P1 / C2 重配置 | 既有 Reacher 与 PushT 实际负/混合结果、代码修复及能力探针均保留 | 预声明任务族中，以部署时可用信息选出的非贪心动作，在同一根上同时满足未来能力改变和真实任务收益；独立 calibration/outcome、固定原 margin/CI 门，不删反转根、不用 oracle 或物理未来选方法。当前仍未通过 |
| P2 多程序组合 | full20 CEM/random、MPPI-inspired、internal gradient MPC、无组合 SRA 和 paired composition 已有实际资产；paired 四根模型 Q 改善4/4、真实 Q 2/4、pose 1/4 属总账已有结果 | 上游 P1、稳定 root 身份与源码/动作/目标/预算 join、组合相对无组合 SRA 及强基线的独立真实增量。80predict=40完整H2只匹配该前向单位；JVP/VJP/SVD/backward、编码、搜索域、加载/反馈/墙钟不能免费或冒称等总预算 |
| P3 动态 horizon/计算 | v4 四对物理已有完整执行，pose dynamic−uniform 为0/+71.857102/+11.635172/0；已有核账与接线审计 | 几何驱动的预封存调度、同 root/反馈/有效 horizon 的强对照、噪声/成本/收益消融和真实质量—资源曲线；不能拿预测 Q 改善或少某类调用代替任务收益。goal-patch score-only 消融是待验证假说，不是完整新 compiler/物理成功 |
| P4 RGB 多轮闭环 | 已完成真实观察回灌与多轮工程对照，不再称“只有 runner”；现有结果不支持普遍任务收益 | 相应环境自己的 P1 与所采用 P2/P3 组件准入、可靠任务度量、独立根、冻结起终分布/预算、长程至终止、同 checkpoint/反馈强对照。PushT 接近/接触/推送/闭合分别评价，不能以混合 latent/state 距离代替物体任务指标 |
| P5 独立模型/架构与跨任务 | JEPA 原生独立模型加载、新鲜事实输入、v11/v12 实际 CUDA 与当前 prospective C1 已推进；不能再写独立架构资产不存在 | 预声明独立冻结架构/权重身份与合法资产、各模型 P0/任务前提、同一核心程序/机制与真实收益复制，及独立任务/根验证。三新 seed factual 面板物体全程静止、无 contact，仅 agent 移动，事实预测排序不能证明操纵/C2；不能后训练 adapter/teacher/readout 来补 training-free 主线 |

强基线已有内部实现不等于官方论文实现。MPPI-inspired shooting、internal gradient
MPC 应保留原命名；GRASP adapter 的机制子集、H1/H3 和资源限制按独立合同报告。
最新进度记 GRASP CUDA 已启动，并非“完全缺实现”，本文没有读取其未完成产物
或认定完成。paper-reported 数字不能进入本地同协议 paired 因果排序。

此外原三组核心实验没有因 C1 编译完成而消失：Executable Subgoal Gap 需要相同
底层控制器/资源下的真实实现偏差与闭环对照；Controllability-Changing Actions
需要能力变化与任务收益的同根合取及强对照；One Frozen Model, Many Tasks 需要
多目标/horizon/动作限制的独立 success/cost/时间结果。C3 调度科学要求亦未关闭。
这是“仍有明确前提”，不是理论否定，也不是无限追加新门槛。

## 来源快照、验证范围与不变性

阶段合同来源（本次读取时 SHA；它们是共享可追加总账，不要求后续永不变化）：

- `docs/D2P_PARALLEL_CUDA_PROGRESS_20261027.md`：`7d6fedf17f746e3d3c9c774e46477b1df453dc69315f8ea287fb5e01a2b208b7`，尤其“前瞻原生 C1”及“尚未完成的原始要求”。
- `docs/D2P_RESEARCH_CLOSURE_AUDIT_20261025.md`：`2c15f493c0fad66c346246de22b1dd8c5850edf79e0b7ea283a14f66ed5553ac`，原提案要求表与末尾新增闭环边界。

本次直接核对的固定文件：

- executor v2 test SHA：`1e08b9cd577e0d5bf90644c5451c6a62da432843faedc13e2aea8df47d92e96a`。
- `le-wm-main/results/p5_native_c1_executor_20261027_v2/manifest_source_frozen.json`：`52f52d9ac6cd9dcbfa2fd310f974e086c7cc3424b050989e0ea9879df9d320d1`。
- 旧 compiler `cuda_main_v1/compiler_receipt.json`：`2ee8a5185c978d01f0e5bbdaf086a1a87e6d1b4e332f6efb5ba76715ee018b12`。
- ridge `cuda_v1_audit.json`：`94dca6cb3fd6b0f159ad3b639badaac12c89a6b94987d0962ced18486b95a4d1`。

总账中的其他阶段数值在本文仅作有来源的状态引用，未逐一重新认证其全部 seal、
权重或原始数组，不冒充本次独立全仓库审计。本次只重跑 v2 的非神经 tests，
无 CUDA、神经/AD、真实 physics、训练、下载、删除或新根选择；总账与 sealed 输入未改。
