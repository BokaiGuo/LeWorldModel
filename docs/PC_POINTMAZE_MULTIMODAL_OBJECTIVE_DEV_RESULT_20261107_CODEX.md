# PointMaze 多模态目标评分：开发结果

## 结论

在固定的 1884–1891 开发根、两个视角和三个五步候选动作上，补充目标 proprio 的官方权重评分**没有改变任何选中动作**，没有修复已观察的偏移视角排序错误。预定开发出口未达到；不应继续把这个固定权重评分当作该错误的已证实修复。

本次对照使用完整当前和目标 proprio；先前视觉评分只使用目标 RGB。因此这是更高目标输入权限的 **state-assisted** 诊断，不是同权限纯视觉比较，也不能用于 RGB-only 结论。这些任务、候选及历史结果已经暴露。

## 可复核结果

- 两臂均使用同一当前 RGB、当前 proprio、冻结模型、旧 centered 变换及候选。新臂计算视觉预测 MSE + `0.1 ×` 预测 proprio token 目标 MSE；`0.1` 取本地官方 PointMaze L2 配置。上下文为本地适配的单帧、时域为一个五步包，并非完整官方六步 CEM 复现。
- 在每一个候选上，视觉分数复现先前封存分数（`atol=1e-6`、`rtol=1e-5`）；组合代价经本地官方 `ReprTargetDistMPCObjective` 逐次核对。48 次模型预测产生 16 个视角的两臂选择，**0/16 发生变化**。
- 两臂四舍五入后的平均物理 regret 均为 **0.013466**；八根两视角平均 regret 中 **0/8** 改善。
- 1887 偏移视角的三个候选按 zero、+0.1、−0.1 排列。视觉分数分别为 `0.397016 / 0.396502 / 0.403472`，加权后为 `0.397091 / 0.396522 / 0.403753`；两臂都选 `+0.1`，物理 regret 均为 **0.143465**。
- 评分前只导出当前输入与目标输入；候选物理后继在独立评分进程之外。所选结果封存后才读取既有后继进行评估。目标 proprio 是允许的目标输入，而不是候选未来状态。流程隔离不是操作系统级强制沙箱。
- 配套测试 **20 passed**；包含评分路径不引用物理数据文件、评价按封存选择保留全部根的合成检查。测试不是独立任务或真实闭环效果。

该结果只排除本地兼容模型、单步预测、给定目标 proprio 和固定权重下的这一解释；不能排除其他目标函数、模型误差或输入兼容差异，也不能证明理论无效。未修改历史结果或任何 gate。

## 产物

- 协议：`docs/PC_POINTMAZE_MULTIMODAL_OBJECTIVE_DEV_PROTOCOL_20261107_CODEX.md`
- 脚本：`le-wm-main/experiments/pc_pointmaze_multimodal_objective_20261107_codex.py`
- 记录：`le-wm-main/results/pc_pointmaze_multimodal_objective_20261107_codex_selection.json`、同名前缀 `_audit.json`
- 测试：`le-wm-main/tests/test_pc_pointmaze_multimodal_objective_20261107_codex.py`
