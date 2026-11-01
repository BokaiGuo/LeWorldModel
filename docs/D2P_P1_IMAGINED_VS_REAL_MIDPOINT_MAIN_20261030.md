# P1：相同优化器下的真实与想象中点

固定 24 个既有前缀，首阶段均为主机 CUDA 的 10 次更新，执行五步
后以相同初值、相同优化器再做 10 次更新；唯一规划输入变化是
中点 latent：真实 RGB 编码或模型对已执行五步的预测。想象组也
采集并编码实际 RGB，但不向规划器提供该真实编码。所有 24 对的
前五步动作、状态和完整物理快照逐步一致，两组均执行满十步，
原生成功均为 0/24。

真实中点组相对想象中点组，物体位置误差有 6/24 条更低、物体
角度误差 10/24 条更低、推杆误差 6/24 条更低、原生联合位置
误差 6/24 条更低。两个已知根和重复前缀不能视为 24 个独立样本。
结果说明当前开发面板上，替换中点并没有产生一致的物理改善；
不说明反馈理论无效，也不能用它证明 P1 成立。

24 条真实与想象中点编码的均方差平均为 `0.0356500`。独立的
主机 CUDA 重放从固定 cp5 图像、同一已执行动作重算两个中点：
保存向量最大绝对误差均为 0。这里复用了同一 checkpoint 和模型
实现，不是独立 ABI 或训练 normalizer 来源证明。真实历史及动作
接线在该对照范围内没有发现错误，并不能排除其他代码或协议问题。

两组阶段优化账本均为各阶段 10 backward、22 action encoder 和
22 predict；每阶段都编码一次。想象组另外执行一次 action encoder
和一次 predict 来产生中点，该两项是显式代码路径计数，不是
独立 hook 计量；两组并非严格等总成本或完整成本认证。源码和
保存消费未证明完整依赖闭包或 OS 隔离。本实验为看到既往结果后
制定的已知根探索，不是预注册独立确认。

固定协议
`le-wm-main/results/codex_p1_two_stage_launcher_20261030_v1_prepare_pred10_MAIN01.json`
SHA `91703a1069185f6e6bcdd98d7a61a07c9d70e887aaa7a88c216634e75594891c`；
真实轨迹保存消费
`le-wm-main/results/codex_p1_predicted_midpoint_10x2_20261030_main_saved_MAIN01.json`
SHA `d8f74f0c71aa8130ab774574a7de2befffd313c45d3c0d1895f382245e5f3bc3`；
逐对结果
`le-wm-main/results/codex_p1_predicted_vs_real_10x2_20261030_main_RUN01.json`
SHA `6a6e5a045a5905d11e388289cd085de2ab97b31ab3b12a0d0a314527d0685d41`；
CUDA 数值重放
`le-wm-main/results/codex_p1_predicted_midpoint_replay_20261030_main_RUN01.json`
SHA `427a6c815567b1142a3b8d3525b870e56dc8a37789de2c8cc3ec6fcac38c8316`。
协议和配对篡改测试共 2 项通过；P1 仍关闭。
