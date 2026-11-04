# v7 Reacher 实际探针：模型与物理 RGB 响应

按[开发协议](PC_V7_REACHER_ACTUAL_FD_PHYSICAL_DEV_PROTOCOL_20261104_CODEX.md)，重放 v7 第三轮真实使用的两组 nominal 和 `±0.05` 探针，共 **42 个五步动作包**。每个分支从同一 reset、warmup 和前两轮真实派发动作重建，根 RGB 以及各分支起始 qpos、qvel、目标一致；保存物理终点 RGB、状态、原动作和冻结编码器 latent。

| nominal | 模型响应范数 | 物理 RGB 编码响应范数 | 矩阵绝对误差 | 相对物理范数误差 | 矩阵方向余弦 | 目标 latent MSE 两两排序一致率 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4.9433 | 6.4293 | 1.8704 | 0.2909 | 0.9797 | 178/210（84.76%） |
| 1 | 7.9796 | 10.2272 | 2.9245 | 0.2860 | 0.9785 | 177/210（84.29%） |

这两组结果表明，在所测任务 chart 和尺度上，模型响应与实际后继 RGB 编码的整体方向接近，但幅度误差约 29%，目标 latent MSE 的两两排序也并非完全一致。210 对来自各自同一组 21 个候选，**不是 210 个独立样本**；排序未给近似平局设额外阈值。chart 是 latent 空间投影，不是物理关节坐标，目标 latent MSE 也不是实际成功。

[原始结果](../le-wm-main/results/pc_v7_reacher_actual_fd_physical_dev_20261104_codex_result.json)绑定[动作、图像与编码数组](../le-wm-main/results/pc_v7_reacher_actual_fd_physical_dev_20261104_codex_arrays.npz)；[独立统计重算](../le-wm-main/results/pc_v7_reacher_actual_fd_physical_dev_20261104_codex_audit.json)验证了实际请求批次、两个矩阵与排序计数，未再次运行物理环境。模型内部数值一致、模型与真实 RGB 响应不完全一致是不同判断。

本结果仅覆盖一个已暴露根、最终 5 步和一个有限差分尺度；没有验证所有 v7 名义状态、完整物理轨迹、任务物理距离排序或其他模型。不能据此宣告 PC1 通过或推断所有负结果源自模型误差。
