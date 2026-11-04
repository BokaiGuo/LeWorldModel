# JEPA 历史保持 rollout：单根开发结果

按照[协议](PC_JEPA_PUSHT_HISTORY_AWARE_ADAPTER_PROTOCOL_20261104_CODEX.md)，在已暴露 episode 1、冻结 JEPA PushT checkpoint、相同 RGB 和位置／速度上下文上，比较原包装器的历史零填充与[显式历史适配器](../le-wm-main/experiments/spectral_reachability/pc_jepa_history_rollout.py)。两臂使用同一条真实动作后缀，预测完成后才读取未来图像作为评价标签。

| 预测时域（每 packet 5 步） | 原包装器目标 patch MSE | 历史保持目标 patch MSE | 两者预测 patch RMSE |
|---:|---:|---:|---:|
| 1 | 0.05608356 | 0.06643459 | 0.05913549 |
| 2 | 0.05558545 | 0.06110700 | 0.06286010 |
| 3 | 0.06216161 | 0.06275669 | 0.03950789 |

[原始输出](../le-wm-main/results/pc_jepa_history_rollout_dev_20261104_codex.json) SHA-256 为 `3d0147056586ce8936c77fd2a9a6b21dc314c16da8768b46d9398d8ee428ff3f`；模型预测共六次，两次图像编码、一次本体编码。首步两臂均重现[此前四条件诊断](PC_JEPA_PUSHT_HISTORY_PACKET_DIAG_RESULT_20261104_CODEX.md)对应结果，误差小于 `1e-6`。历史保持适配器的四项合成测试覆盖动作滑窗、batch、因果性、输入不变性和非法历史拒绝：`4 passed`，不代表物理或科学结果。

训练配置和加载器提供具体对齐依据：帧以五步间隔采样，每帧对应从该帧开始的连续五个动作；训练 `shift=1` 以该帧预测下一帧。三帧上下文首步的动作槽因此为 `[0:5, 5:10, 10:15]`。这是对已检查源码和绑定配置的解释，不单独证明 checkpoint 训练过程全部来源。原包装器首步未接收真实历史，在其槽位填归一化零；不能不加说明地把两条预测当作相同接口。

首次运行因新实验桥接遗漏 predictor 的 patch 到空间轴 reshape 而在第一个预测后失败，没有读取未来标签或写出结果。随后按 `VideoWM.forward_pred` 的 AdaLN 分支恢复形状，并完成上述运行；这是本次新桥接的代码错误，不说明历史实验的失败由此造成。实验执行的是原 `EncPredWM.unroll` 方法，但以受控 `NativeCore` 桥接代替完整 `VideoWM` 构造；桥接只复制该 AdaLN 路径的 action identity 和输出 reshape，未认证完整预处理或其他模型分支。原包装器源码 SHA-256 `105a6768050e4b34c95de8d7d0cc0c7ba07b69d364acf0649ba2625bacce907b`，参考 `VideoWM` 源码 SHA-256 `657bd3e9eb2ec35442cc06e208b4632a53503a35a83a96abbbf82608d2fe9154`；这些在运行后另行计算，不是完整运行时依赖封存。

**边界：** 本根上原包装器三个时域的目标 latent MSE 均较低，但这是已暴露数据的单轨迹、单指标比较；不能据此选择正式方法、宣称物理收益或否定训练对齐。模型使用 agent 位置和速度，不是 RGB-only；未运行 v7 搜索或真实闭环，也不支持 PC5 通过。下一步应先对实际合法观测来源、训练归一化和历史动作链做完整输入认证，再在事前固定的开发协议下测试 v7 接线与同权限基线。
