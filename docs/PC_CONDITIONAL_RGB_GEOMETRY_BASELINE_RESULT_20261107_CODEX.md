# 条件尾部面板：RGB 几何强基线

依据[开发协议](PC_CONDITIONAL_RGB_GEOMETRY_BASELINE_PROTOCOL_20261107_CODEX.md)，
先对已有 roots 20–49 用当前 RGB、目标 RGB 和既有 TwoRoom 几何规则生成
并封存固定 25 步动作；之后以物理状态只用于重置和评价的方式执行。
两个旧 CEM 臂直接使用已有物理记录，不是本轮重新运行。

| 开放环方法 | 25 步内曾成功 | 最后一步成功 | 相对几何的配对差异 |
| --- | ---: | ---: | --- |
| 固定 RGB 几何 | 30/30 | 30/30 | — |
| 条件末帧 CEM | 29/30 | 未在本表重算 | 几何独胜 1，独负 0 |
| 条件时序 CEM | 28/30 | 未在本表重算 | 几何独胜 2，独负 0 |

相对时序 CEM 的独胜是 root 38、39；相对末帧 CEM 的独胜是 root 39。
30 个输入均可解析，没有弃权。配对数少、任务已曝光，不作统计显著性
或独立泛化声明。此规则的门几何和动作尺度是 TwoRoom 专用人工先验，
其表现不证明模型评分、行动后果描述或程序编译有增量。

动作先由只接收图像的生成函数产生、封存，再读取物理真值重置与评价；
整个实验未做进程级权限隔离。独立脚本重新生成 30 条动作并重放全部
750 步，逐步距离和终止标志均一致；最大初始渲染与输入像素平均绝对
差约 `7.31e-5`，这不是隐藏状态认证。重放使用同一生成实现和模拟器，
并非独立实现或独立任务。两项测试通过，环境仍有 Gym 观测类型及动作
转换警告。本轮没有运行 GPU：几何基线本身不需要模型查询。

这个结果进一步说明，不能将当前条件尾部方法在该开发面板上的表现
当作强规划优势；后续若比较视觉世界模型方法，必须把适用的几何
基线纳入，同时在不依赖 TwoRoom 手工先验的任务上验证。它不能推断
已有模型后继失配是否由代码 bug 引起，也不能据此改判 PC0–PC5。

[实现](../le-wm-main/experiments/pc_conditional_rgb_geometry_baseline_20261107_codex.py) ·
[完整结果](../le-wm-main/results/pc_conditional_rgb_geometry_baseline_20261107_codex_result.json) ·
[重放审计](../le-wm-main/results/pc_conditional_rgb_geometry_baseline_20261107_codex_audit.json) ·
[动作封存](../le-wm-main/results/pc_conditional_rgb_geometry_baseline_20261107_codex_actions.npz)
