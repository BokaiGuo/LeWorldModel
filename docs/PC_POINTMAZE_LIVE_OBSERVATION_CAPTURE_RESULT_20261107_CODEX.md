# PointMaze 现场反馈观测封存：开发结果

按 [协议](PC_POINTMAZE_LIVE_OBSERVATION_CAPTURE_PROTOCOL_20261107_CODEX.md)，
在八个已曝光根上用冻结 PointMaze JEPA、完整 proprio 和持续真实物理执行，
复现原有反馈过程；原运行器和历史结果未修改。此次实际重规划前的 32 张 RGB
和对应 proprio，以及 8 张执行结束后的图像，均以无损数组保存并记录哈希。

- 32/32 次选中索引与历史一致；全部 1536 个候选评分最大绝对差为 0。
- 8/8 条完整执行动作及物理末距与历史一致；现场 proprio 与规划记录匹配。
- 本次 32 张现场重规划图像与此前事后重放生成的图像逐像素一致，
  proprio 最大绝对差为 0。历史运行没有保存现场图像，不能直接证明
  历史现场图像与它们相同。
- 新增封存与原反馈候选测试共 3 项通过；实际 GPU 运行完成，既有模型指纹检查通过。

新运行、数组和审计分别为
`le-wm-main/results/pc_pointmaze_live_observation_capture_20261107_codex_run.json`、
`..._observations.npz` 和 `..._audit.json`。哈希及动作结果可用审计文件核对。

本次使用 `experiment-design` 限定事先条件和历史对照。这是同一算法在已曝光
任务上的工程复现，不是新增方法、等成本对照或独立确认；属于 state-assisted，
不能用于证明 RGB-only 闭环或变更 PC0–PC5 的判定。现场封存只适用于本次运行，
不补造历史观测证据。
