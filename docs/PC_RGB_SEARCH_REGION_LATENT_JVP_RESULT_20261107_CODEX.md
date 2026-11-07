# RGB 搜索区域表示导数开发结果

按 [协议](PC_RGB_SEARCH_REGION_LATENT_JVP_PROTOCOL_20261107_CODEX.md)，在 4 个已曝光 TwoRoom 任务、5／15／25 步、每时域两个物理动作方向上完成 GPU 检查；共 24 个方向，不是 24 个独立任务。

默认高效注意力后端首次 JVP 直接报错：`aten::_scaled_dot_product_efficient_attention_backward` 不支持所需二阶导数。首次运行未生成结果。随后在任何完整样本结果产生前修订协议，在局部强制 math SDPA；默认与 math 后端的名义动作前向结果最大表示元素差为 **6.83e-6**，最大目标分数差为 **3.17e-4**，均低于本次预定容差。这不认证其他设备或所有候选。

| 检查 | 结果 |
| --- | --- |
| JVP 与有限差分，步长 0.001／0.002／0.01 | 各 **24/24** 达到本次向量 L2 判据 |
| 三个步长的最大向量 L2 误差 | 分别 **0.003075／0.001141／0.000764** |
| JVP/VJP 伴随关系 | **23/24** 达到本次判据；最大绝对差 **0.00019253** |

唯一未通过项为任务 9000、25 步、方向 0。按[事后重放协议](PC_RGB_LATENT_JVP_ADJOINT_REPLAY_PROTOCOL_20261107_CODEX.md)连续重算 10 次，差异均为 0.000192523，未出现可见运行间波动。该项投影 JVP 为 0.237565，VJP 为 0.237372；三个尺度的投影有限差分分别为 0.237599、0.237529、0.237584。有限差分接近 JVP，但这不足以确定误差根因或宣布 VJP 实现错误；原始失败按原判据保留。

运行器与解析函数测试共 **5 项通过**。本次只说明抽查内点的模型表示求导行为，既不证明边界处正确，也不证明表示具有真实物理相关性、谱搜索收益或 gate 通过。源动作为裁剪后的 CEM 链，图像为数据集初始帧而非执行时重渲染；真实状态未输入规划器。

[多时域原始结果](../le-wm-main/results/pc_tworoom_search_region_latent_jvp_20261107_codex.json) · [固定样本重放](../le-wm-main/results/pc_tworoom_latent_jvp_adjoint_replay_20261107_codex.json) · [多时域运行器](../le-wm-main/experiments/pc_tworoom_search_region_latent_jvp_20261107_codex.py) · [重放运行器](../le-wm-main/experiments/pc_tworoom_latent_jvp_adjoint_replay_20261107_codex.py)
