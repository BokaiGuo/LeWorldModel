# P1 V10：V8 index16 的 cp5 真实 RGB 后缀重规划最小方案

## 范围

V10 只绑定已封存的 V8 model-only 选择结果：`selected_index=16`，
候选 `rgb_delayed_x23_neg04`，五步前缀动作 SHA256 为
`0db61c3546f87f17bdf69efa4b55e6a8e9678ae7b4627c1c3cebd731d58bdb36`。
本方案不读取或修改 V9 文件，不使用 cp5 之后的物理结果，不运行 CUDA 或物理，
也不打开 P1 admission。

## 最小闭环

1. 从同一 reset 精确执行 V8 index16 五步前缀，保存实际 cp5 PNG、五个实际动作及
   文件/像素/动作哈希。
2. 仅从实际 cp5 RGB 分割推杆、物体和目标，计算任务法向
   `n=(goal_centroid-object_centroid)/||.||`，并取物体沿 `n` 的后侧一像素边界为
   支撑面。
3. 从相同原始 CEM 随机张量构造两臂：
   - `v10_contact_normal`：通过 V8 已冻结的 RGB 推杆响应模型，把后缀前三步投影到
     图像位移空间；每步至少沿 `n` 前进 0.25 px，切向位移不超过 0.75 px。
   - `matched_unconstrained`：不投影，使用相同原始样本。
4. 推杆响应递推必须用实际第五个前缀动作初始化上一动作。把上一动作清零会漏掉
   cp5 后第一步的速度记忆，属于接线错误。
5. 对每个候选的 nominal、双轴正负扰动及 checkerboard 正负扰动，共五个变体：
   - 十步后缀全部经冻结模型 rollout；
   - 五个变体的前三步都必须满足支撑面距离不超过 6.5 px、接触向量与任务法向
     夹角余弦不低于 0.8；
   - 任一变体不满足即记为不可行；
   - 可行候选按十步终态 latent 代价的 `mean + range` 排序。

## 同预算合同

两臂均固定为 5 次 CEM 迭代、每次 64 个候选、8 个 elite、5 个稳健变体、
2 个五步 packet。每臂逻辑预算为：

- RGB encode：1；
- action encoder：10 次；
- predictor：10 次；
- predictor samples：3,200；
- 完整后缀长度：10 步；
- wall cap：1,800 秒。

两臂共享随机数种子和标准正态噪声序列，但分别根据各自得分更新 CEM 分布。
几何投影只使用 CPU 数组运算，单独报告耗时；不得用虚假神经调用补齐预算。

## Fail-closed 规则

- cp5 RGB、前缀动作或 V8 来源哈希不一致：两臂都中止；
- RGB 分割失败：中止，不允许模拟器状态回退；
- 无候选在五个变体上保持前三步接触：V10 abstain，并按失败计；
- 任一臂调用计数或完整十步 rollout 不合格：该配对为工程无效；
- 保存验证前不得物理执行后缀；已知开发根结果不得称为独立确认。

## 后续实现文件

最小生产实现应继续只新增：

- `le-wm-main/experiments/spectral_reachability/p1_v10_cp5_rgb_suffix_geometry_20261101_parallel.py`
- `le-wm-main/experiments/spectral_reachability/p1_v10_cp5_rgb_suffix_runner_20261101_parallel.py`
- `le-wm-main/tests/test_p1_v10_cp5_rgb_suffix_runner_20261101_parallel.py`

当前设计侧车及测试：

- `le-wm-main/experiments/spectral_reachability/p1_v10_cp5_rgb_suffix_design_20261101_parallel.py`
- `le-wm-main/tests/test_p1_v10_cp5_rgb_suffix_design_20261101_parallel.py`

这份交付只证明 V10 合同和纯函数草案可复核，不是 CUDA 计划、物理结果或 P1 机制证据。
