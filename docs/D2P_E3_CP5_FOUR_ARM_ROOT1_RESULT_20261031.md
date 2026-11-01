# E3 已知根 cp5 目标：四臂真实反馈结果

按 `D2P_E3_CP5_FOUR_ARM_ROOT1_PROTOCOL_20261031.md`，固定第二个
已知根、历史 cp5 RGB 和对应原生状态；control0 时原生成功为 false。
四臂使用同一冻结模型及其各自的真实 RGB / 动作历史，主机 CUDA
规划、真实物理执行。原生成功或截断后停止，不补后续步数。

| 臂 | 实际控制步数 | 原生成功 | 终点物体位置误差 | 终点物体角度误差 |
| --- | ---: | --- | ---: | ---: |
| structured | 2 | true | 13.9500 | 0.13431 |
| native-style | 3 | true | 13.9088 | 0.13392 |
| random | 20 | false | 0.5845 | 0.00181 |
| immediate-progress | 5 | true | 5.8632 | 0.03040 |

random 的物体位置和角度误差更低却未成功；物体误差不能替代包含
推杆的原生联合成功。三条成功记录在其各自停止时评价，不能把
缺失的第 20 步补齐，也不能将四臂当作四个独立任务。

保存态消费器复核目标 RGB 哈希、目标 latent、物理目标状态、终点
原生成功、模型／journal 哈希、7 个实际到达节点的账本与
autograd API 计数、选中动作与实际执行前缀；未到达节点保持
`not_started`。消费器没有独立复演完整物理事件链。模型参数和
buffer 前后哈希一致；目标及根来自历史轨迹，前缀与完整流程
成本尚未得到公平资源认证，native-style 不是官方强基线。

这是一个新的真实反馈开发任务，不是独立多任务泛化、E3 通过、
P1 机制通过或方法优势证明。尤其三条成功都发生在第一包，只有
random 到达了之后的真实反馈节点，不能把此任务称作四臂均完成
多步重规划验证。固定矩阵原有 G1 目标和结果未修改。

- 任务接线代码：`le-wm-main/experiments/spectral_reachability/e3_cp5_four_arm_root1_20261031.py`
- 目标及错误根绑定测试：2 项通过。
- 准备 SHA-256：`c010862c9bd586920522b769786bab24f923a25d4eac4cb7806cfbefc25541ff`
- 报告：`le-wm-main/results/e3_cp5_four_arm_root1_20261031_RUN01/report.json`，
  SHA-256 `40cf4312c7195cd3ed5f142f99503749dc5eee0408d6ec60ba20c5f967fc4947`
- 保存检查：`le-wm-main/results/e3_cp5_four_arm_root1_saved_20261031_RUN01.json`，
  SHA-256 `2f6d034438ce1d4c2ea3a317477ed39c17a55ace02f3908d2a6f718ee0eade91`
