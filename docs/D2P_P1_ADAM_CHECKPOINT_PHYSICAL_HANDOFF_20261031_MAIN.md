# P1 Adam 5/10 更新：固定物理执行交接

5 和 10 次更新的 CUDA 候选分别按固定最低模型分数选出后，适配
到既有 24 行物理执行器。来源、全部四个候选动作哈希、赢家规则、
原始前缀与 cp5 RGB 均在 prepare 时复核。适配器、候选生成及原
物理代码相关测试共 10 项通过；测试不代表物理效果。

准备文件及 SHA-256：

- `results/p1_adam_checkpoint_physical_5_20261031_prepare_RUN01.json`：
  `eaec5459a4aa5755941ef2749fb0f6e4264600eb3c33f379b5266860a8b938af`。
- `results/p1_adam_checkpoint_physical_10_20261031_prepare_RUN01.json`：
  `5be37dd41e7edfb228390a8924ecfaf9688ea0e54cd93386a3d042aaf43b9202`。

两个 24 行物理作业进程已启动；最近检查时仍在执行前置核验，
输出目录尚未创建，不能认定物理步骤已经开始。预定输出目录分别是
`results/p1_adam_checkpoint_physical_5_20261031_run_RUN01` 和
`results/p1_adam_checkpoint_physical_10_20261031_run_RUN01`。
作业执行的是已由主机 CUDA 生成的动作；物理仿真本身不是神经推理。
完成后必须先核实完整 `run_report.json` 状态及实际哈希，再用同一
入口的 `consume` 子命令传入对应 `--step`、`--prepared`、
`--sha256`、`--directory`、实际 `--report-sha` 和独占的
`--output`；不得因进程退出或部分分片推断结果。

20 步的 24 条物理结果此前已完成，且该次新 CUDA 的 20 步赢家
与原动作逐行相同，可按身份绑定后比较。各预算点来自共享的
20 步优化轨迹，中间评估有额外成本；它们不是独立预算运行的
真实耗时曲线，前缀生成成本和统一总预算仍未认证。所有输入为
已知根及事后开发，禁止按物理结果选择预算或候选，不构成 P1
通过。提前成功和删失不得填充不存在的固定时域终点。
