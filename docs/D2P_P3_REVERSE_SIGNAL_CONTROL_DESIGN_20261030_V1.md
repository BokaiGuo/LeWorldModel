# P3 在线反转信号负控：固定设计与 callback 接线

## 状态

本设计在已观察 RUN_MAIN01 之后形成，针对未来独立数据；**不是已完成的事前注册**。本次只新增 callback 接线和 RAM 测试，不运行 GPU、模型、仿真/物理或训练，不更改原科学门、封存合同和已有结果。负控不是已发现的科学效果，不能继承旧四槽 strict receipt。

现有 `p3_matched_order_feedback_seam_codex_20261029` 已实现 fixed/signal 两策略、观察反馈、endpoint bank 和逐轴 Ledger；`p3_matched_order_feedback_host_v2_codex_20261029` 及其 strict consumer 固定为两根四槽。`agent_p3_commitment_controls_20261029_v2` 的六臂为另一合同、实际预算不匹配。三者均未提供本次的反转信号三臂在线 producer/strict，不能直接重标原产物。

## 未来比较，固定六槽

按 root0、root1 顺序，每根依次运行 `fixed_order`、`signal_order`、`reverse_signal_order`。两个根是已知根而不是独立 fresh replication；全部六槽、错误、原生停止和未执行项保留，不按结果择槽、重试、补零或退款。固定顺序意味着潜在 warm/cache 和时间顺序影响，不能解释为随机化因果证据。根、模型、goal、normalizer、启动/目标身份与整段 initial warm10 必须在未来 prepare 中事前绑定，每臂独立初始化并核同根实际状态/像素/goal。

所有臂每个 parent 都实际付原 H1/H3 信号、原调度、四个 endpoint 候选的 whole/segmented replay、原 model-Q 稳定选择及独立 winner replay；路径包含一次 H1 和一次 H3，共 20 计划动作。每个完整 5 步 packet 后按实际返回 raw 与实际帧重建本臂上下文；H3 的后续 packet 是同一已选完整动作缓冲，并非在中途改变计划。

| 策略 | 第一块 | 第二块 |
|---|---|---|
| fixed | H1 | H3 |
| signal（原约束策略） | 原 decision 的 admissible H1/H3 | 剩余 horizon |
| reverse signal | 在同样的原 admissible decision 下，首块取另一 horizon（H1↔H3） | 剩余 horizon |

反转的是**原决策的首块 horizon**，不是数值 signal、梯度或最终动作取负；无需额外模型调用。原 `continue/progress/extension` 支持判断在 signal 和 reverse 首块完全一致，不支持则原样停止，不借 fixed fallback。第二块均不再用新 decision 改剩余额度；仍付信号及记录原 decision。各臂独立产生其自身实时观察和 signal；如果初始状态或原 decision 未被独立验证相同，不得声称两臂是同一个决策的精确反事实。即使首块互补，未来后续反馈不同也不是单独隔离初始方向的作用。

拟预先固定的主对照为同根计划 20 步 native7 distance 的 signal−reverse；辅助 signal−fixed 及 native success、joint XY/object XY/velocity，在 0/5/10/15/20 分时点报告。原生早停保留实际终点与缺失，不把终态复制到 20。该主/辅助设定仅为本方案设计，须在未来任何结果前经独立审读、可信不可变时间戳和独立授权后才可声称事前固定；后续代码、政策、输入或实现变动须重新封存，不可事后改主对照。

## 数值与 callback 计量

新实现 `le-wm-main/experiments/spectral_reachability/p3_feedback_reverse_control_20261030_v1.py` 复用原 `parent_task`、signal/spectra、`core.choose`、endpoint bank、candidate Q/whole+segmented/独立 replay、context join 和 Ledger。它没有修改旧模块、没有 monkeypatch 旧策略或伪造旧 schema。单臂完整计量向量的 caps/attempted/completed 都须是原 `design.expected()`：例如 predict 52、Jacobian 4、VJP directions 768、replay 18；其余轴完整保留，hook shape 多重集另比。不同轴有嵌套，不能加总为 FLOPs；不同时间的中途资源未保证相等。

packet 和 encode callback 各有 intent/return/failure 与实际调用计数；完整臂要求各 attempted=completed=4。`completed` 表示 callback 已返回，不代表其返回值通过 ACK/帧身份校验；错误 ACK 或帧哈希仍拒绝该槽。callback 本身不证明内部真实物理步数、encoder kernel、设备、模型、native IO 或总预算，`physical_steps`、`encoder_kernels`、FLOPs 保留 null。异常后 no-retry；已付 Ledger 和回调 attempted 不退款。发出 packet intent 后原生 stop 不追加动作、不编码缺帧。

`compare` 是**结构性对照**：核相同 root digest、原 decision/反转映射、H1+H3、完整 ledger 与 callback 数，分别输出已测数值轴和 hook shape 是否相同。它不独立验证 root、真实设备或物理 meter，`native_execution_certified=false`，更不是旧 strict consumer；不把 count 匹配说成等 FLOPs/wall/能源、正确预测或科学通过。

## 未完成的生产接线及门槛

当前入口只接注入的 session/backend/store，既不创建环境或模型，也没有持久化、来源封存、HOST 驱动、原生物理 meter、目标/帧身份严格消费、存储/显存门或独立 saved consumer。callback `kind='HOST_CUDA'` 只是接口标签，不是权威。`execute_roster` 在 session factory 抛错前没有获得 close callback；factory 仍须负责自身部分初始化的清理。store 写失败时也不能保证终态 durable。

未来真实实验必须先独立完成并审读六槽 runner、源/权重/准备绑定、原生资源和每步 task/goal/state/动作/RGB 的保存、失败收支与独立 strict consumer；验证每个 arm 的预定 caps、实际 attempted/completed、tensor shape、物理/编码计数，并分别保留无数据项。仅在可信事前封存后申请真实 GPU、仿真/显示和存储协调。旧四槽 strict 与本 RAM 测试均不能认证新六槽。没有更改 P1/P3/C3、freshness 或其他科学门。

## 测试

```sh
env CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_p3_feedback_reverse_control_20261030_v1.py \
  tests/test_p3_matched_order_feedback_seam_codex_20261029.py \
  tests/test_p3_matched_order_codex_20261029.py \
  tests/test_p3_feedback_budget_contract_20261030_v1.py
```

**36 passed**。覆盖固定/原 signal 与旧 seam 的 block、packet 和 ledger 等价；首块反转与原 gate、完整三臂账和 shape；不支持 decision 在付完信号后拒绝而非 fallback；原生提前停止、ACK/帧错误、encode/replay 失败不退款、不追加动作、固定六槽/cleanup、缺认证标志，以及账本/选择篡改拒绝。数据与后端均为 RAM fixture，不是生产、科学或硬件证明。
