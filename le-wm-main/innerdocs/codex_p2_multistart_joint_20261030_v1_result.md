# P2 multi-start joint 对照：探索性结果

## 合同与完成情况

模型协议在本次模型运行前固定，见 `codex_p2_multistart_joint_kernel_20261030_v1.md`；物理协议在本次物理运行前固定，见 `codex_p2_multistart_joint_physical_20261030_v1.md`。设计时已有历史结果，因此不称预注册或独立确认。固定分母为四个已知 base × 两臂共八槽，所有槽均实际运行并经保存消费；base 2 和 base 3 是重复根，不是两个独立样本。

无组合 `joint_multistart` 和 `prefix_chain` 使用相同四个首包锚点、完整 20-logit 球、每臂 16 候选、5 Jacobian、5 SVD、17 score；上限为 48 packet/nested model calls 和每臂 450 秒。前者在四锚点做全 10 步、20 维 Jacobian 并生成完整动作；后者分别在四个前缀上下文计算局部单包 Jacobian。候选不是逐点相同，搜索几何、雅可比维度、计算工作及实际耗时不同，不能将差异直接归因于程序组合，也不能称等总算力。

主机 CUDA 八槽完成；固定合作锁仅约束参与者。物理回放每槽使用字面保存的 15 步共同前缀和 CUDA 赢家的 10 步动作，绑定 reset/goal，cp15 RGB 与状态守卫，遇原生终止或截断即停止，不补步。全部八槽物理达到 25 步，均无终止、截断；保存消费和四作业合并完成。消费者为独立进程，但复用部分生产代码，不是独立科学复现。

## 固定分母结果

Q 为模型赢家保存的预测 cost；位置、角度为最终物理误差，角度为弧度；native success 为环境原生指标。object_success 在所有槽均为 null，不应替换成失败或成功。

| base | 策略 | Q | 位置误差 | 角度误差 | native success |
|---|---|---:|---:|---:|---|
| 0 | joint_multistart | 0.870782 | 79.061736 | 0.901563 | false |
| 0 | prefix_chain | 0.788026 | 13.234272 | 0.080992 | false |
| 1 | joint_multistart | 0.672910 | 67.127038 | 0.452136 | false |
| 1 | prefix_chain | 0.614643 | 26.175187 | 0.102859 | false |
| 2 | joint_multistart | 0.523988 | 80.873255 | 1.562795 | false |
| 2 | prefix_chain | 0.508096 | 61.697382 | 0.431733 | false |
| 3 | joint_multistart | 0.523988 | 80.873255 | 1.562795 | false |
| 3 | prefix_chain | 0.508096 | 61.697382 | 0.431733 | false |

本批描述上，prefix_chain 各列槽的预测 cost、位置及角度误差均较低，但八槽原生成功均为 false；重复根和不等实际计算量限制解释，不能据此声明公平优势或机制成立。

## 实际资源

每槽 joint：packet 44 次、Jacobian 5 次（均 192×20）、SVD 5 次、score 17 次；chain：packet 28 次、Jacobian 5 次（一次 192×20、四次 192×10）、SVD 5 次、score 17 次。各轴 attempted 和 returned 相同。action_encoder、predict、predictor、pred_proj 各计数与 packet 相同但为嵌套调用，不能相加当作独立总调用。joint 分支墙钟 0.707–0.925 秒，chain 0.446–0.458 秒；模型加载 4.234 秒，CUDA 子进程总墙钟 11.668 秒，不应归属为每臂独占成本。

物理四作业墙钟之和 36.625732 秒；八槽累计 constructor 8、close 8、reset 16、goal_bind 8、goal_setter 24、goal_render 16、set_state 32、explicit_render 888、public_step 280、physics_step 2832，以上 attempted 和 completed 均一致。这些是不同层次的调用，不能简单相加。FLOPs、能耗、引擎内部 backward 调用数、完整动态源码闭包为 unknown；模型加载开销与物理开销分别报告，不伪造等成本。

## 可核验产物

- CUDA prepare `results/codex_p2_multistart_joint_host_20261030_v1_PREP_01.json` SHA256 `d83faeb229070f3ce86addb07347123499555f7056cfe295c4426f8d239cddb8`。
- CUDA 运行 `results/codex_p2_multistart_joint_host_20261030_v1_RUN_01`；saved receipt `results/codex_p2_multistart_joint_saved_20261030_v1_RECEIPT_01.json` SHA256 `dddd5d3021c9263231cd0ade7e2ae14fd91490ad210bc33d3cdb9578c7dea8b9`。
- 物理四份 prepare、四个运行目录、四份 saved receipt 均以 `results/codex_p2_multistart_joint_physical_20261030_v1_B{0,1,2,3}_` 为前缀；整批 `results/codex_p2_multistart_joint_physical_20261030_v1_JOIN_01.json` SHA256 `f87186e10e18e977503dc25a44ab6b0f2405e142dbaa729fd2c4227d97414b78`。
- 离线内核、host 和 supervisor 测试 9 passed；物理生命周期、来源、停止和篡改测试 12 passed。离线测试本身不是实验。

本结果为已知根探索性物理回放；P1/P2 科学 gate 不变，既不证明泛化，也不证明 P2 通过。
