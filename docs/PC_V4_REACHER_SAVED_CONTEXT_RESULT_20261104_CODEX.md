# V4：Reacher 已保存 RGB 上下文接线结果

依据[预先协议](PC_V4_REACHER_SAVED_CONTEXT_PROTOCOL_20261104_CODEX.md)，未更改 v4 编译器在之前真实执行记录的第零决策 RGB 历史上完成一次离线规划；未派发动作。原冻结模型、动作归一化、二维目标 chart 和完整 latent MSE 与 v3 路径一致。此次调用 59 批、737 条轨迹样本、1920 个五步预测 token，记录了 231 条有限差分动作样本和 22 次 SVD；token 不是完整计算成本。

| 同一根的第零次规划 | 选中链模型内 latent MSE |
| --- | ---: |
| v4 response | 0.788459 |
| 已有 v3 response | 0.785329 |
| 已有 smooth CEM | 0.787526 |

三条结果来自已暴露单根，后两行是以前封存的记录，并非新增运行；已核对两臂第零上下文的 frames、root、goal 和 history 数组完全一致。v4 选中非零前缀，零前缀比较得分为 1.421463，因此未弃权。这些是冻结模型内的分数，不能推断真实终点、方法优势或统计显著性。

[独立参考路径审计](../le-wm-main/results/pc_v4_reacher_saved_context_20261104_codex_audit.json)重算了全部 59 批、737 个样本；投影最大绝对误差 `2.86e-6`、分数最大绝对误差 `4.77e-7`、动作映射最大误差 `1.79e-7`。同时核对请求到实际 float32 动作、合法边界、token、选中链实际被评分、精修前全链 incumbent 及弃权比较。参考路径与运行路径共享模型和同一数据，不是独立 checkpoint 或物理响应验证；声明的依赖封存也不保证完整环境闭包。

这证明 v4 在该现有视觉上下文上可接线并通过单次模型预测重算，**不证明 PC1 或 PC4 通过**：二维 chart 对任务响应是否充分、物理动作响应、v4 实际三轮 RGB 重编译、历史和 normalizer/训练来源均仍需单独处理。也不能把 v3 的物理轨迹嫁接给 v4。

排他产物：`le-wm-main/results/pc_v4_reacher_saved_context_20261104_codex_{manifest,result,audit}.json`。运行入口与独立参考审计分别为 `experiments.pc_v4_reacher_saved_context_20261104_codex` 和 `experiments.pc_v4_reacher_saved_context_audit_20261104_codex`，需使用对应 GPU UUID 与项目协作锁；已有输出不能覆盖。
