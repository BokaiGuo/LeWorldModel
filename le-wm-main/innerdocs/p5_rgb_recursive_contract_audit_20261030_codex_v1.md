# P5 开环递归合同审计

范围：仅接受并重新校验的已公开 MAIN02 开环和 MAIN01 真实观测刷新日志，固定 164 枝、每条件 82；无神经执行、CUDA、物理、训练或未开放 holdout。模型是原本地物理状态标签监督的视觉模型，不是 JEPA。

## 代码逐层对照

1. `run_causal_visual_pusht_world_model.py` 的 `device_batch` 将 state 和 next_state 各自 `(raw-mean)/std`；`model_loss` 对 `model(...)` 返回的 decoded 与 normalized state、predicted 与 normalized next_state 分别计算损失，并用 `predicted-decoded` 对齐真实 normalized effect。其 `forward` 使用 previous/current RGB 生成 state，再调用 `predict_next`。
2. 同文件的 `predict_next` 计算 `delta=self.dynamics(features)`，但**返回** `state+delta`，并非只返回 delta；`physical_metrics` 解码时做一次 `predicted*std+mean`。目标图像由独立 goal head 处理，state predictor 不输出 goal 相对量。
3. 原 planning `visual_plan_action` 在多步情况下将 `states=model.predict_next(states, action, ...)` 递归，末端才 `raw_from_normalized`。所选 checkpoint 的 temporal dynamics、previous-action dynamics、contact-frame 等开关均为 false，故 forward 的额外 temporal/previous-action 参数对该 checkpoint 的 dynamics 不生效；只在 initial encoder 中利用两帧。原 planning 是否实际被该模型报告采用、其准确性和训练历史包是否完全同版，不能从此证明。
4. MAIN02 runner 先 encode 两帧，每步直接将前一步返回的 normalized state 和相同 raw action 送入 `predict_next`，不额外加 state、不对 action 或 state 做再次归一化。物理对照用相同 mean/std 仅在记录预测后解码。刷新 runner 每一步重新 encode 同枝真实两帧，然后预测一步；H1 两方案输入相同。

上述静态逐层证据未发现可确认的状态归一化、delta/绝对值或本 checkpoint 的递归接线错误，因此不提出未经证实的“修正”。训练主要是一跳有监督目标，代码的多步调用不保证稳定，也不证明误差来源。

## saved 诊断

对 h=1…5，每枝比较开环本步输入（h1 为原 encode，之后为前一步开环输出）与同一步刷新真实观测 encode；再比较两种一步输出。两者都由相同历史 physical 动作驱动。逐轴和每枝保存 normalized 差，off/on 分开计算完整 82 枝平均绝对差及六维 L2 的逐枝均值。差异反映不同输入和其对应输出的联合变化，不是局部 Jacobian、因果分解或误差放大率。

动作 L2 > 0.85 只作为当前源码默认训练动作最大范数的参考计数；历史训练报告未证明该字段及旧环境源码的完整身份，不能因此判定训练分布外。当前训练源码使用 raw action，物理既有合同也记录 raw action 原样送环境；没有证据据此再缩放 100。off/on renderer 与原训练外观差异仍未排除。

后续判别：主线程先检查完整按轴误差、递归输入分歧及同动作支持；若另获授权，可固定相同单步接口对比同一输入调用 `forward` 与 `encode_state`+`predict_next` 的 CUDA 输出、以及逐次记录预测状态并在相同输入上做局部扰动；必须先登记干预、固定分母与权限、适配现有硬超时和验收。不得将 teacher-forcing 信息优势当公平闭环收益，不从两个非独立已知 root 宣称机制或科学通过。
