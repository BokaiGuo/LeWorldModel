# E3 已观测 cp5 目标：CUDA 编码结果

按 `D2P_E3_CP5_TARGET_CUDA_PROTOCOL_20261031.md`，两张既有
control5 图像使用冻结 PushT 模型在主机 `cuda:0` 重新编码。
两根的重编码与此前保存的 192 维 float32 latent 均逐字节一致，
最大绝对差及 MSE 均为零；模型参数与 buffer 前后哈希一致。

- 结果：`le-wm-main/results/e3_cp5_target_cuda_20261031_RUN01.json`
- SHA-256：`ff788213aff290ebd636fce22e55de64d3fddae128d8492bfdc4aa995c586518`
- 两次 encoder 调用；GPU：NVIDIA GeForce RTX 5060 Ti。

这只是目标图像与保存 latent 的接线验证。目标来自已知轨迹，
并附有历史动作见证；未进行新规划或物理闭环，不能视为独立
多任务泛化证据，也不改变原 G1 合同或 P1 阶段准入。
