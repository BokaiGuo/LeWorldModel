# PushT 周期评估的规划环境入口：开发验证

[新入口](../le-wm-main/experiments/pc_pusht_periodic_planning_env_v1.py)为单任务 PushT 显式构造周期评估器、原 TimeLimit、张量包装及可选 RGB 包装；复制原 PushT 分支的时钟和配置更新，不修改原注册表。上游规划评估器最终调用 `env.eval_state`；本入口生成的顶层环境在状态与 RGB 两种路径均将该调用转发给周期评估器。

真实环境对照检查了配置、观测空间、动作空间、相同种子和初始状态下的观测、一步动作及后继物理状态；角差 `2π+1` 时顶层调用的原规则与修复版判定不同。对照只是单步工程测试，不是训练模型、规划搜索或 RGB 闭环性能实验。

## 额外发现：状态模式 prepare 后的时钟

上游状态观测模式的 `TensorWrapper.prepare` 转发到底层 `PushTWrapper.prepare`，绕过 `TimeLimit.reset`；首次 `prepare` 后直接 `step` 会因计时器为 `None` 而报错。RGB 的 `PixelWrapper.prepare` 已显式重置计时器。新入口的 `PreparedTensorWrapper.prepare` 在成功准备之后将时钟归零，测试覆盖首次及重复 prepare；原实现未改。现有结果是否经历该调用路径尚未全面审计，不能假定其受影响，也不能把它当作此前负结果的原因。

验证命令（在 `le-wm-main` 下）：

```bash
PYTHONDONTWRITEBYTECODE=1 SDL_VIDEODRIVER=dummy MPLCONFIGDIR=/tmp/pc-mpl-cache PYTHONPATH=/home/gbk/code/LeWorldModel/le-wm-main/.jepa-physics-site:/home/gbk/code/LeWorldModel/le-wm-main/external/jepa-wms .venv/bin/pytest -q tests/test_pc_pusht_periodic_planning_env_v1.py tests/test_pc_pusht_periodic_eval_v1.py
```

结果：**12 passed**。新增入口必须被未来实验明确调用并封存；原 JEPA-WMS 默认入口和既有实验仍不自动使用它。当前没有运行真正的 `PlanEvaluator` 闭环或同核心 Planning Compiler，不能据此宣称 PC0–PC5 完成。
