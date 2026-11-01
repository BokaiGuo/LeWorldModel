# P5 多步误差的代码合同初步审计

对主机 CUDA MAIN02 的已开放、已知根物理面板比较，block-y 第五步逐枝平均绝对误差 off/on 分别约 2724.72/2705.79；这是异常大的数值，不应在审计之前归因于理论。

本次主线程检查了 `p5_rgb_cuda_runner_20261030_codex_v1.py`、`p5_rgb_known_physical_compare_20261030_codex_v1.py`、`p5_rgb_causal_panel_bridge_20261030_codex_v1.py` 和原训练／评估源码 `run_causal_visual_pusht_world_model.py`：

- 图像路径均为 RGB、OpenCV `INTER_AREA` 缩放 64×64、CHW 和 `/255`；桥接实现用 NumPy，训练实现用 PyTorch，尚未证明各设备的每个浮点位都一致。
- 该 checkpoint ABI 固定 `action_repeat=1`；训练按原始 `[-1,1]` 动作执行一步，runner 向每次 `predict_next` 输入保存的相同 raw 动作。消费者验证 10 步原始动作哈希，并将后五步与每条记录的动作哈希及物理 step 6–10 对齐。
- 训练以 `(state-mean)/std` 为状态监督，原评估用 `state*std+mean` 还原；消费者相同地还原并对照 `[agent_x,agent_y,block_x,block_y,sin(angle),cos(angle)]`。这些源码合同中尚未发现足以解释异常的简单二次归一化或明显时间错位。
- 当前 runner 从 step 5 的两张真实 RGB 编码一次，之后连续 5 次递归反馈模型状态。原训练数据生成每行一条一步转移；本批没有逐步刷新真实图像。逐步复算显示 off 组 block-y 的平均绝对误差在预测 horizon 1–5 依次约 `15.78, 146.82, 429.84, 1106.34, 2724.72`。这与开放递归误差增长一致，但不能据此唯一归因于模型不稳定；预处理、部署分布、历史帧、动作与接触仍需进一步逐项验证。

这些是对代码和已保存产物的有界核查，没有本轮重新执行 P5 神经网络，也没有证明完整训练来源或所有运行时张量等价。应补充 CUDA 上真实观测逐步刷新和同一动作的一步预测对照，保存实际模块输入张量与预测；禁止用 CPU 神经推理、训练新模型、事后去除不利样本或将该已知面板误称为独立泛化。

数值来源：[已开放面板对照](../le-wm-main/results/p5_rgb_known_physical_compare_20261030_codex_v1_MAIN02.json)，SHA256 `a16ff8b11eaf7f18cd27b0182f3e9b9d9f81224c8ce309fb75fd3794a8733211`。
