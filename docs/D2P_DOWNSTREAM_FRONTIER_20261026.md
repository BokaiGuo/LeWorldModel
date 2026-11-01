# D2P downstream evidence frontier — Task A

2026-10-26。P1 未通过，P2–P5 均未准入。本页与新增只读审计器不改变任何封存源、准入代码或共享文档，不启动训练、adapter、神经推理或物理执行。

## 定义与依赖的核对

原附件 `861f6764-1b25-4227-bf15-f1a237ba7d8f/pasted-text-1.txt` 只明确命名 P0/P1；P1 是 kill experiment，成立后才值得实现多程序组合、动态 horizon、跨场景 harness。后续编号来自本地 closure audit，而非附件原始编号。

| 阶段 | 实际对象与前置条件 | 当前证据边界 |
|---|---|---|
| P2 | 多程序组合；先通过 P1；每边有动作见证和预测 postcondition，冻结 chart transition/search；真实收益超过无组合 SRA 和同预算 flat search | prefix-observe-suffix、dispatch/action-word 原型不等于组合搜索效能；未准入 |
| P3 | P1 及必要 P2 后，由动力学几何决定 horizon/算力；固定预算/时域、噪声与成本消融 | 旧 SRA horizon 源码不是 D2P 的验证；未准入 |
| P4 | 每环境自身 D2P 准入和适用的 P1/P2/P3；长时 RGB-only 闭环、根级配对 success/cost；PushT 分接近/接触/推送/闭合 | 可独立准备评价/回放合同；旧 SRA 与分支校准不能开放 Full D2P |
| P5 | 同模型多任务与独立冻结架构复制分别报告；各模型先验证动力学前提，预声明任务/资源合同 | 可只读查资产/许可/接口；repair/teacher/adapter 资产不能补 training-free 复制 |

来源：`D2P_RESEARCH_CLOSURE_AUDIT_20261025.md` 的 P0–P5 表；`D2P_CVPR_CRITICAL_PATH_20261026.md` 第 47–75 行的阶段矩阵与依赖链；`SPECTRAL_REACHABILITY_ATLAS_RESEARCH_PLAN_20261023.md` 第 406–413 行的旧阶段表。旧表的 P2 强调 Action-Order Primitives/charts，不能将非交换预测效应独自当作新版完整组合收益。

`le-wm-main/experiments/spectral_reachability/d2p_stage_admission.py` 实际只提供 P0→P1、P1→P2 标量门，没有 P3/P4/P5 专门准入器。`p2_admission` 精确要求：

- 非 calibration，声明最低 eligible roots 至少 2 且实际达到。
- mechanism_gate 必须属于三个 `P1_MECHANISM_KEYS` 之一；全部 `P1_INTEGRITY_KEYS`、所选机制门及全部额外 gate 必须严格为 True。
- required comparisons 包含 zero-prefix、greedy two-stage、full-horizon CEM，且全部预声明对照实际存在；检查所有 supplied comparisons，而不只 mandatory 三个。
- 每个 CI95 都必须两端有限、顺序正确、上界严格小于零；触零或平局不通过。

该标量函数本身不验证完整原始 root/source/protocol 绑定，不能把手工构造的合法参数当作 P1 成立。runner 的 calibration binding、来源隔离、封存和真实执行仍需单独验证。本审计器不增加替代准入通道。

## 本轮已执行、且不依赖 P1 的下游前置工作

1. 对显式 allowlist 的 admission、composition prototype、旧 SRA horizon、sampling/reported baseline 接口及测试源做 AST 静态核查，记录真实路径、函数行号与现有 gate 常量；未执行这些源码。测试文件存在只表示 implemented/test-source-present，不推断 tested 或 physically validated。
2. 对父线程新报告的保存标签逐根/候选核查重复和完整性，实际重计 4 根 ×16 分支 ×2 阶段：128 saved stages、64 terminals；官方与 object-only 终态成功均 0/64。此次仅重计已有 evaluator 输出，不冒称重新调用官方 evaluator 或新物理实验。
3. 已运行新增审计器测试及原阶段门回归：46 passed。覆盖缺失/损坏报告、超限读取、源码不执行、重复根/候选、缺标签、截断 bank、符号链接越界、肯定标志不放行。纯逻辑测试不充当物理效能。

实际产物：`le-wm-main/results/d2p_downstream_frontier_task_a_20261026/frontier.json`。其中 implemented、tested、calibration、physically_validated、admitted 分列；所有后续阶段的后两项均 false，不把上游物理校准误标为该阶段验证。

## 当前物理与任务合同证据

- `results/pusht_d2p_sealed_program_execution4_v1_20261026_cuda/report.json`：completed，5 项完整性门 true，64 已封存程序真实执行。这是已知四根校准，不是独立 outcome。既有 object-pose cost 下四根 selected 均差于 zero-prefix；不推广到全部指标。
- `results/pusht_d2p_sealed_task_contract4_v1_20261026/report.json`：completed，`official_goal_includes_agent_position=true`，`P1_admission=false`。父线程用实际安装的 official eval_state 与独立检查核验全部 128 保存阶段；本任务只重计报告标签。
- 官方成功约束 agent+object 位置与角度；object-only 不是同一任务。两套终态均 0/64，协议错位不挽救 P1，不允许按更有利结果更换任务定义。详见共享物理执行审计页，但本任务不修改该页。

所有输入仅作报告内容引用，不重新认证其 source seal。只读 frontier 不以 metadata 的 completed/True 推出机制成立；即使注入 P1_admission=true，也保持失败关闭，等待另行授权的完整验证器。

## 重跑与资源边界

在 `le-wm-main`：

```bash
.venv/bin/python -m pytest -q tests/test_d2p_downstream_frontier.py tests/test_d2p_stage_admission.py
.venv/bin/python -m experiments.spectral_reachability.d2p_downstream_frontier
```

CLI 只向 stdout 输出 JSON，不写输入或输出目录。本次 frontier JSON 由 apply_patch 新建；程序自身无写盘功能。每个显式输入最多读取 2,000,001 bytes，超出 2 MB 拒绝；无目录遍历、权重加载、大文件哈希或神经模块导入。报告是有限清单快照，不声称全仓库不存在其他证据。无新神经工作，因此不存在 CPU 神经降级；未来神经工作仍必须 HOST CUDA。无训练、提交、推送或删除。
