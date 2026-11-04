# v3 Reacher 冻结模型接口预检结果

[v2 预检协议](PC_RESPONSE_TEMPORAL_V3_REACHER_ADAPTER_PREFLIGHT_V2_PROTOCOL_20261104_CODEX.md)
在已暴露根 `20263160`、原冻结 Reacher checkpoint、既有动作统计与
真实热身历史上完成。根和目标 RGB 哈希与既有收据一致。每个时域
测试三组原始动作；时间顺序 packet 归一化可逆，缓存预测与既有
root-anchored 参考预测的最大绝对差分别为：

| 原始动作时域 | 最大绝对差 |
| ---: | ---: |
| 5 | `4.77e-7` |
| 10 | `5.96e-7` |
| 15 | `9.54e-7` |

均在预先规定的 `2e-5` 绝对及相对容差内；二维图像 latent 投影、
完整 latent 评分均有限。测试固定动作且未派发新策略动作，不验证
局部响应与物理相关性，也不构成 E1 闭环、PC0 或 PC1 通过。

## 首次失败与修正

第一次宿主机预检的最大差约 0.482。源码核对发现参考
`_terminal_batch` 对输入执行 `tanh` 有界映射，第一次预检错误地
将已归一化的实际动作直接当作映射前参数，导致二次压缩。v2 对
严格处于动作域内部的探针应用精确逆映射并检查往返；未更改容差，
也未修改先前部分产物。该问题定位在**预检参考接线**，不能据此
声称 v3 编译器或冻结模型本身有 bug。相关适配器及方法测试共
13 项通过；测试与模型内数值一致都不代替真实控制证据。

最初沙箱也因未暴露 NVIDIA 设备而停止，未启动模型推理；后续使用
GPU 可见环境及独立输出。预检依赖清单未声称完整环境闭包。

- [原协议](PC_RESPONSE_TEMPORAL_V3_REACHER_ADAPTER_PREFLIGHT_PROTOCOL_20261104_CODEX.md)
- [结果](../le-wm-main/results/pc_response_temporal_v3_reacher_adapter_preflight_v2_20261104_codex_result.json)，SHA-256 `19eceac763bdd2e102acf95450c4913168bba7673605ae38efc2c4ade4dce1fd`
- [RGB 与历史](../le-wm-main/results/pc_response_temporal_v3_reacher_adapter_preflight_v2_20261104_codex_context.npz)，SHA-256 `92cf35069d1c2acec6cacc94e43789b7d47348a2590f960934e23108b4882b73`
