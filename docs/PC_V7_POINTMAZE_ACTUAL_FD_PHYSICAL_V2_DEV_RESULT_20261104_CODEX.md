# v7 PointMaze：实际有限差分探针的物理响应诊断

按[原协议](PC_V7_POINTMAZE_ACTUAL_FD_PHYSICAL_DEV_PROTOCOL_20261104_CODEX.md)
及[精度修正协议](PC_V7_POINTMAZE_ACTUAL_FD_PHYSICAL_V2_DEV_PROTOCOL_20261104_CODEX.md)，
重放已暴露根 0 第三轮实际使用的两个 nominal、各 21 条五步探针；
没有利用物理结果重新选择动作。每次分支的 shadow RGB 边界均等于封存的
第 25 帧；native 重放的前 26 个状态均与此前 native 重放匹配。

| nominal | 模型对 native 响应绝对误差 | 相对 native 范数 | 响应余弦 | 模型评分对 native 距离的非并列成对排序一致率 |
|---:|---:|---:|---:|---:|
| 0 | 0.19265 | 0.39610 | 0.92469 | 182/210 = 86.67% |
| 1 | 0.21879 | 0.51319 | 0.89776 | 172/210 = 81.90% |

同一 shadow 物理后继的 RGB 位置估计与 native 响应相比，相对误差分别为
0.29947、0.38131，余弦为 0.95596、0.97684。它们并不精确等同于
native 状态：像素量化在 ±0.05 小扰动上可能显著影响差分。
模型对 RGB 响应相对误差为 0.48943、0.46806，余弦为 0.87501、0.88381；
模型评分与 RGB 终距非并列排序一致率分别为 172/207、157/204。

**解释边界：**该点上整体响应方向较接近，但幅度和部分动作排序有实质误差；
不能仅凭这些数将原因归于模型，因果图像状态估计、模型投影、接触和
跨后端差异都可能相关。native 并非原 shadow 运行逐状态记录，尽管两者使用
同 XML、相同封存动作且 RGB 边界吻合。模型分数与物理距离是不同评分；
成对比较并不表示模型估计的绝对坐标与物理坐标一致。单根、单边界、
单尺度的 42 个探针不构成 PC1 通过或独立泛化。

首版诊断因将历史 warmup 从原始精度误转为 float32，在严格 native 前缀
核对中失败，最大状态差约 `1.59e-7`，未保存物理探针结果；保留其
[部分 manifest](../le-wm-main/results/pc_v7_pointmaze_actual_fd_physical_dev_20261104_codex_manifest.json)
及[原脚本](../le-wm-main/experiments/pc_v7_pointmaze_actual_fd_physical_dev_20261104_codex.py)。
v2 仅修正重放精度，不是规划器修复。

[运行脚本](../le-wm-main/experiments/pc_v7_pointmaze_actual_fd_physical_v2_dev_20261104_codex.py)
封存源、动作和依赖；[原始结果](../le-wm-main/results/pc_v7_pointmaze_actual_fd_physical_v2_dev_20261104_codex_result.json)
包含完整 native 状态和指标，
[数组](../le-wm-main/results/pc_v7_pointmaze_actual_fd_physical_v2_dev_20261104_codex_arrays.npz)
包含 210 张分支 RGB、动作和状态。
[复算审计](../le-wm-main/results/pc_v7_pointmaze_actual_fd_physical_v2_dev_20261104_codex_audit.json)
检查哈希、原始探针、图像因果估计、状态、矩阵和统计；它复用运行脚本的
误差及排序计算函数，并未独立重渲染或重跑 native 后端。
