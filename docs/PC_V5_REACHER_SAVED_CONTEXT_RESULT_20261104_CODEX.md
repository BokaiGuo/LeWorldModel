# V5 Reacher 已保存 RGB 上下文结果

未修改的 v5 核心在已暴露的 Reacher 真实 RGB 首轮上下文完成单次 15 步规划，使用 1920 token、737 条预测、59 批，选中链的模型 latent MSE 为 0.786575，未弃权。相同上下文已有 v4 分数 0.788459；这只是单根模型内分数，不证明物理效果或规划优势。

[参考路径审计](../le-wm-main/results/pc_v5_reacher_saved_context_v2_20261104_codex_audit.json)逐批重算全部样本，最大投影差 `2.87e-6`、分数差 `4.77e-7`，核对 token、选中完整链和弃权。它共享同一冻结模型和 RGB 上下文，不是独立 checkpoint、物理响应或闭环证明。

[v2 结果](../le-wm-main/results/pc_v5_reacher_saved_context_v2_20261104_codex_result.json)及其 manifest 绑定 v5 和适配依赖。首轮在沙箱中缺少 GPU 设备，模型分配前退出，未派发动作；其 manifest 保留，按[重试协议](PC_V5_REACHER_SAVED_CONTEXT_V2_PROTOCOL_20261104_CODEX.md)使用新前缀。控制台文案沿用 v4，但 manifest 的方法及实际调用为 v5。环境依赖闭包和数据来源仍未完全认证。
