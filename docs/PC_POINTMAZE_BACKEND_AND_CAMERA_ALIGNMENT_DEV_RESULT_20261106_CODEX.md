# PointMaze 编码后端与训练参考相机：开发诊断

## 两条 DINO 兼容路径

按 `PC_POINTMAZE_DINO_BACKEND_PREDICTION_DEV_PROTOCOL_20261106_CODEX.md`，用相同 HF 权重严格映射至本地 DINOv2 源码快照，固定 checkpoint 的 predictor、当前 proprio、图像和动作，对 1884–1891 八个已暴露任务、各两个视角进行真实 GPU 配对评分。模型选择先封存，真实编码和物理轨迹后置评价。

| 指标 | HF shim | 源码快照 |
| --- | ---: | ---: |
| 平均物理 regret | 0.013466 | 0.026221 |
| 预测首选与真实编码首选一致 | 8/16 | 5/16 |
| 视角选择不一致任务 | 1/8 | 2/8 |

四个视角的动作选择变化。1887 偏移视角从 `+0.1` 变为 zero，物理 regret 从 0.143465 降到 0.069119，但两者仍未选择真实最优 `-0.1`。由于整体结果较差且未满足预定出口，不能将该后端替换称为修复。源码快照与映射权重也不是官方原始权重或端到端等价的独立认证。

## 目标图像参考中心

旧图像归一化将静态迷宫框中心对齐至 `(111.5,125.5)`，此前的归档相机匹配结果对应 `(111.5,111.5)`。新增以目标 RGB 估计、对当前与目标图像共同施加的训练参考对齐实现，保持旧版和历史结果不变；两方向偏移均限制在 ±20 像素。

按 `PC_POINTMAZE_TRAIN_CAMERA_ALIGNMENT_DEV_PROTOCOL_20261106_CODEX.md`，在上述同一面板比较三臂，选择先封存，真实编码及归档帧后置评估：

| 指标 | raw | 旧中心 | 训练参考中心 |
| --- | ---: | ---: | ---: |
| 初始图像与归档帧平均 MAE，像素范围 0–255 | 25.103 | 17.871 | 0.714 |
| 平均物理 regret | 0.019158 | 0.013466 | 0.013327 |
| 预测首选与真实编码首选一致 | 8/16 | 8/16 | 8/16 |

1887 偏移视角新参考的归档初态 MAE 从 18.080 降到 0.910，但物理 regret 仍为 0.143465，动作仍为 `+0.1`。因此物理排序问题没有修复，预定开发出口失败。新参考是基于此前开发任务识别出的坐标修正，不是官方相机参数来源认证；整数平移与最近边缘填充也不等同于新真实视角。新实现未接入正式规划器。

两项均为完整 proprio 的 state-assisted 开发，8 个任务此前暴露、两个视角共享任务；不能作为 RGB-only、独立确认或闭环收益。PC0–PC5 不变。原始产物：

- `le-wm-main/results/pc_pointmaze_dino_backend_dev_20261106_codex_selection.json`
- `le-wm-main/results/pc_pointmaze_dino_backend_dev_20261106_codex_audit.json`
- `le-wm-main/results/pc_pointmaze_training_camera_alignment_20261106_codex_selection.json`
- `le-wm-main/results/pc_pointmaze_training_camera_alignment_20261106_codex_audit.json`
