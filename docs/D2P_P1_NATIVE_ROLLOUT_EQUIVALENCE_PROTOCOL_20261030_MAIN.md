# P1 自写 rollout 与 LeWM 原生 rollout：CUDA 审计合同

已知结果后提出的工程诊断，不是独立确认。固定全部 24 条已封存
cp5 RGB、根历史、前缀与 Adam 选出的两包动作；不得根据结果选择行。
使用相同冻结 checkpoint、图像预处理、动作 normalizer 和 CUDA 精度，
比较自写两包 rollout 与安装的 `LeWM.rollout` 的末端 latent 和
相同目标 MSE 对两包 raw 动作的梯度。原生 rollout 收到同一状态历史
和前两包历史动作，以及两包候选动作；分别记录完整预测和梯度误差、
动作身份、设备和模型散列。比较数值差异不等于证明数据来源正确。

预测与梯度差异若显著，需先检查动作时间对齐、原生 API 输入语义和
浮点批处理差异；不能立即归因物理模型。若一致，只排除这一具体
两包实现与当前安装原生接口的差异，不证明训练期 normalizer、真实
执行、编码目标或模型预测正确。该审计不执行物理、不改变正式 gate。

## 实际结果

主机 RTX 5060 Ti CUDA 完成全部 24 条，冻结参数与 buffer 散列前后一致。
自写与当前安装 LeWM 原生 rollout 的末端 latent 最大绝对差为 **0**；
目标 MSE 对全部 20 个 raw 动作分量的梯度最大绝对差为
**4.470348358154297e-8**。对应每行预测、梯度和评分均已保存；消费
测试独立从保存数组复算差异，并检查动作哈希、24 行分母及原 Adam
选中评分在 `1e-5` 内一致。

这排除了被测固定输入上的明显两包 rollout／梯度接线差异，但两条
路径共用当前模型、预处理、历史和 normalizer，不排除这些共同输入
有误，也不证明训练来源、物理预测、P1 机制或跨根泛化。检查不是
新的物理执行或公平性能比较。

- [CUDA 结果](../le-wm-main/results/codex_p1_native_rollout_equivalence_20261030_main_RUN01.json)，
  SHA-256 `7152974b335328940b0630313d6d049874a5f72a5ea9e20621f47bf304991855`。
- [审计代码](../le-wm-main/experiments/spectral_reachability/codex_p1_native_rollout_equivalence_20261030_main.py)；
  [保存数组测试](../le-wm-main/tests/test_codex_p1_native_rollout_equivalence_20261030_main.py)
  1 项通过，测试不重新运行神经模型。
