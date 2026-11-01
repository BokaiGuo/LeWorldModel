# Full D2P 与 SRA 物理配对：saved-only 前置

本审计只读固定资产，不 import 任何 producer、不加载模型、不调用 physics；新输出独占创建。
它不是可执行的 Full D2P 生产 runner，也不是执行授权。

## 已核实的接点

`p4_multiround_feedback_runtime_20261027_v1` 的 `arm_loop` 接受 `reset`、`backend`
和 `emit` 等参数，调用 `backend.encode/compile`，先持久化 selection 再执行五步。
但原实现固定 root1、固定原 prefix，实际/想象两臂在 5/10/15 步各编译一次
单 packet，并非可直接换输入的通用 Full D2P 接口。
`p4_feedback_root_runtime_20261027_v1.FeedbackAdapter` 暴露规划请求、选择验收和
执行回执，但依赖同根真实 capability/response source pack 和外部 driver。
`p2_no_composition_sra_20261027_v2.selected_actions` 导出旧封存的联合 H2 winner，
不是在线实时重编译；其 physical consumer 部署的是已有计划。

固定 report 中四根 SRA 与一根多轮记录有一个相同 root SHA 和 goal7；本审计没有
证明全部 reset、RGB、model、Q、normalizer 和 source closure 等价。SRA 原计划 10
步/一次 H2，而多轮为 20 步/固定 prefix 加三次反馈编译。禁止以相同 root 和目标
推断整体协议相同，也不把多轮 calibration 命名为 Full D2P。

## 下一个实现接缝和必要验收

1. 先冻结 Full D2P 算法组件和固定 root×arm×time roster；独立认证原始 task、
   root/state/history、goal/RGB、模型和归一化、reset/controller/renderer。已知 root
   只用于工程校准，不称 fresh。P1 等科学前置必须另有证据。
2. 新建独立 driver，把每臂自己的实际三帧和已执行两个 action packet 送给各自
   planner callback；planner 返回原目标绑定、候选、资源及可验证选择。不得把原
   `selected_actions` 当在线 SRA；要么为 SRA 实现并验证在线 adapter，要么事前明确
   对照的不同反馈机制，不能声称纯组件消融。两臂相同时间定义、动作 chart、guard、
   终止与评估；完整目标和 root 逐项绑定，不用物理标签选计划。
3. 物理前固定各资源轴上限和共同停止规则。predict API、样本、action/visual
   encoder、Jacobian、数学 VJP、SVD/solve、物理与 setup、I/O 和墙钟分别计
   attempted/completed；共享、继承成本不重复，未知 FLOPs 保留 null。检查
   88 次 SRA 单根和 80 次多轮双臂不是同一分母，不能据此宣称匹配。
4. 新 driver 的测试至少覆盖：真实来源闭包变动拒绝；跨根/目标/历史/时钟拒绝；
   选择落盘早于 step intent；每臂只用自己的历史；early stop/异常不补帧、
   不重试或替根，完整 roster 和缺失时间保留 null；预算用尽在 dispatch 前拒绝；
   固定来源及完整事件可重建。临时 fixture 不能当真实执行证据。
5. 需独立审查和未来显式 GPU/physics 授权后才可安排生产；此文件和审计结果
   不提供 release/dispatch 路径，不创建 authority，不修改旧封存。

## 实际执行

在 `le-wm-main` 中：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_codex_full_d2p_physical_preflight_20261030.py
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python -B experiments/spectral_reachability/codex_full_d2p_physical_preflight_20261030.py
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python -B experiments/spectral_reachability/codex_full_d2p_physical_preflight_20261030.py --verify
```

第一次生成后只能用 `--verify`；生成命令拒绝覆盖。该审计只检查固定文件哈希、
源码 AST 符号和列出的报告字段，不执行原 strict consumer，也不证明其他未列资产
不存在，更不能视为未来生产输入闭包或科学准入。
