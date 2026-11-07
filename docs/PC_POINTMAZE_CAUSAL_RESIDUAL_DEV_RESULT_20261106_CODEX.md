# PointMaze 因果历史残差修正：开发结果

在 `PC_POINTMAZE_CAUSAL_RESIDUAL_DEV_PROTOCOL_20261106_CODEX.md` 固定的 1884–1891 八个已暴露轨迹、两个实际渲染视角上，真实 GPU 评分使用上一观测、实际已执行的五步动作、当前观测及三个相同候选动作。修正量为上一观测预测的误差；没有使用当前候选的真实后继或物理结果选动作。先封存选择及预测特征，另一个进程再进行真实编码和物理审计。模型冻结并核对指纹。Python 进程分离不是 OS 安全隔离。

| 指标 | 原预测 | 历史残差修正 | 当前编码持久性 |
| --- | ---: | ---: | ---: |
| 任务平均候选后继编码 MSE | 0.179182 | 0.053363 | 0.068709 |
| 任务平均三动作物理 regret | 0.036687 | 0.034555 | 0.042493 |

持久性编码 MSE 是根据 `PC_POINTMAZE_CAUSAL_RESIDUAL_PERSISTENCE_AUDIT_PROTOCOL_20261106_CODEX.md` 补做的事后对照，不改变原预定出口。修正在 6/8 任务的平均编码 MSE 低于持久性；但持久性对所有动作给出同一个后继，不具备区分候选动作的能力。

按原先固定条件，本面板开发出口为真：修正平均 MSE 更低，平均物理 regret 不高于原预测，8/8 任务不退化、1/8 严格改善。**决策变化只有 1/16 个视角**：1889 的第二视角由 `+0.1` 切到 zero，其三动作真实终点距离分别为 0.143398、0.177518、0.145428。其余 15 个视角的选择均未改变；因此物理改善只由一个视角贡献，不能称为稳定控制收益。

当前方法只检验单个过去残差、单位系数和单个模型时钟，不构成因果机制发现或描述保留增量。需要额外任务、动作支持、真实连续反馈以及完整成本比较；1892–1895 已被历史实验暴露，不能称为独立确认。额外每次决策至少需要一次历史动作预测及相关编码，不能忽略其费用。

模型接收完整 proprio，属于 state-assisted；不能作为 RGB-only、PC4 或跨轨 gate 证据。任务来源与训练成员独立性也未建立。未改变 PC0–PC5。

原始选择与预测：`le-wm-main/results/pc_pointmaze_causal_residual_dev_20261106_codex_selection.json` 和对应 `_features.npz`；物理审计：`le-wm-main/results/pc_pointmaze_causal_residual_dev_20261106_codex_audit.json`；持久性参照：`le-wm-main/results/pc_pointmaze_causal_residual_persistence_audit_20261106_codex.json`。
