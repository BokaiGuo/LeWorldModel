# P1 固定候选：模型原生 rollout 对照结果

依据 `D2P_P1_OFFICIAL_ROLLOUT_PARITY_PROTOCOL_20261031.md`，
在主机 RTX 5060 Ti / `cuda:0` 比较全部 96 条已执行候选：

- 五步和十步各 96/96 条自定义预测，与当前 checkpoint 模型的
  原生 rollout 预测 latent 逐元素相同，最大绝对差均为 0。
- 两个时间点的自定义目标 MSE 与此前封存记录也逐项相同，
  最大绝对差均为 0。
- 冻结模型参数和 buffer 前后哈希一致；本次代码路径执行
  96 次真实 cp5 编码、384 次 predictor、384 次 action encoder。

原生 rollout 的编码入口被提供相同的已编码 context，以隔离
动作包和时间窗口接线；因此两条路径共享图像编码、模型、
输入历史和 normalizer。对照排除这批输入下两套 rollout 的
实现差异，不能排除共享接线错误、checkpoint 训练数据或
normalizer 来源不明、真实物理预测失真，也不构成 P1 通过。
本次没有物理新执行或候选选择，结果仍来自两个已知根。

结果：`le-wm-main/results/p1_official_rollout_parity_20261031_RUN01.json`
SHA-256 `1fc797dfd20ae4bda767f3807b3108dcb124c9a435d57004ecb94eeeacbe3c41`。
