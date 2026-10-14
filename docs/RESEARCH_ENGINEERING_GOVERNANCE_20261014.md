# LeWorldModel 科研工程治理规范

## 目的

本规范把 LeWorldModel 从“实验脚本集合”逐步整理为可审计的科研工程，同时保留
上游代码、两个历史总入口、历史协议、负结果和既有 artifact。它不改变任何科学
结论；它规定结论如何产生、复用和被拒绝。

核心原则是：

> 每项科学实验只定义一次；编排方式可以不同；执行、缓存、统计和证据认定必须
> 遵守同一份契约。

## 分层边界

```text
core       数据结构、输入 fingerprint、原子 IO、共用数学
studies    PointMaze、PushT、cross-environment、GDD-WM 科学实验
protocols  冻结输入、seed、horizon、阈值和 stage DAG
runtime    stage 执行、日志、状态、恢复、缓存和 artifact 校验
evidence   统计检验、证据记录和六目标判定
experiments 薄 CLI；历史入口继续保留并逐步接入 runtime
```

runtime 不判定科学主张，evidence 不负责 GPU 推理。实验完成和主张成立是两个
独立状态。

## 强制契约

### 1. 运行身份由真实输入决定

每次 run 必须记录 protocol version、代码版本/dirty 状态、配置、seed、数据集或
checkpoint 的 SHA-256，以及 split roots、action support 和设备信息。输入变化时
必须产生新的 fingerprint，旧缓存不得静默复用。

### 2. 缓存必须通过验证

`exists` 或非空文件不等于 completed。可复用 stage 必须同时满足：输入 fingerprint
匹配、协议版本匹配、依赖 artifact 验证通过、输出 JSON 可解析且 schema 满足要求。
无法验证时 fail-closed，重新运行或明确标记 blocked。

### 3. artifact 原子提交

JSON 和状态文件先写入同目录临时文件，再原子替换。stage 在启动、成功、失败和
中断时持久化状态；崩溃不能留下看似有效的半成品。

### 4. 证据显式引用

Claim Matrix 只消费本次 protocol manifest 显式声明的 evidence records。历史结果
可以使用，但必须以 `external_prior_evidence` 声明并记录 checksum；不能靠代码中的
固定历史路径自动补齐本次运行。

### 5. dry-run 无副作用

dry-run 只打印计划和输入检查，不创建正式结果目录、不写 manifest、不写日志。

## 状态模型

Execution status：`planned`、`running`、`completed`、`failed`、`blocked`、
`interrupted`。

Evidence status：`not_evaluated`、`insufficient`、`partial`、`negative`、
`supported`、`blocked`、`not_applicable`。

执行器不得把 `completed` 自动提升为 `supported`。例如 RGB successor stage 成功
只能说明程序完成；独立 held-out planning gate 仍需单独判定。GDD-WM 的 D0 schema
通过也不得提升为 D1 teacher validation 或 D4 planning success。

## 渐进迁移顺序

1. 为现有入口加入 manifest、input lock、stage status 和 schema validation；
2. 让 `run_full_spectral_transfer.sh` 与 `run_unified_research_protocol.py` 共享
   `experiments/research_runtime/`；
3. 迁移 geometry、latent export、empirical audit、claim matrix；
4. 逐步将 PointMaze、PushT、cross-environment 和 GDD-WM 的新代码放入 study-level
   protocol，而不是继续增加孤立总入口；
5. 最后再提取稳定的模型、几何和统计 API。

不进行一次性目录搬迁，不删除负结果脚本，不用新的汇总逻辑覆盖历史结果。

## 证据边界

当前 CUDA 实验和已有负结果保持原样。工程重构的通过只说明可复现性和边界保护
改善，不改变 alignment 普遍预测、稳定姿态闭合、普适 world model 或 commutator
通用 certificate 尚未成立的结论。

## 验收标准

- 输入文件或 checkpoint 改变后，旧 stage 不得被跳过；
- 损坏 JSON、schema 错误和缺失依赖不得被视为 completed；
- stage 中断留下可恢复的结构化状态；
- dry-run 不创建输出目录；
- claim matrix 不读取未声明的历史 evidence；
- 原有测试和历史 artifact 不因重构被删除或覆盖。

## 2026-10-14 首批实现状态

- `experiments/research_runtime/` 已提供 fingerprint、原子 JSON、artifact 校验和
  execution/evidence 双状态；
- `run_unified_research_protocol.py` 已在 manifest 中记录 source input lock、Git
  code identity、stage fingerprint 和显式 evidence records，并在执行前重新验证；
- `build_six_goal_evidence_matrix.py` 在 protocol manifest 存在时不会扫描历史
  visual-online 或 counterfactual 目录，且会拒绝 checksum 不匹配的 evidence record；
- unified runner 会持久化 `run_status.json`、每个 stage 的 `stage_status.json` 和
  失败时的 `execution_report.json`；
- `run_full_spectral_transfer.sh` 已实现 GPU fail-closed、run contract、损坏 JSON
  cache 拒绝和 dry-run 不创建输出目录；
- 相关协议测试与完整测试套件已通过：`632 passed, 3 skipped`。这只证明工程契约和
  回归行为，不改变任何 scientific evidence status。
