# 第二已知根：四臂赢家追加预算结果

按 `D2P_FOUR_ARM_WINNER_BUDGET_ROOT1_PROTOCOL_20261031.md`，
主机 CUDA 和真实物理执行完成。四臂均完成 20 步，均未原生
成功；各臂以自己的真实 RGB 和动作历史规划。16 个反馈节点
均封存并通过独立保存检查。

| 臂 | 物体位置误差 | 物体角度误差 | 四节点同步墙钟（秒） | 每节点 predict |
| --- | ---: | ---: | ---: | ---: |
| structured | 56.3030 | 0.10359 | 6.902 | 75 |
| native-style | 77.2534 | 0.54424 | 2.358 | 81 |
| random | 35.2834 | 1.29081 | 2.681 | 81 |
| immediate-progress | 30.3113 | 0.98511 | 3.611 | 88 |

位置与角度排序不同；以上值均不能代替原生联合成功判据。
同一已知根的先前未追加预算四臂中，structured 与
immediate-progress 不受该优化器改动，报告的物体位置误差
分别为 56.3030 和 30.3113；两种追加优化臂的动作可变。不能
以此建立独立统计或全流程等预算优势。

16/16 节点保存检查复核 report/journal 哈希、CUDA/模型身份、
每节点 forward hook 与账本、赢家追加的分数及动作、最终实际
执行动作。检查未独立复演完整物理事件链；反向调用未独立 hook，
墙钟包含审计开销，不代表公平端到端成本。该根及任务先前已
观察，native-style 不是认证的官方强基线；本轮不证明非即时
进展、真实能力变化、P1 或独立泛化。

- prepare SHA-256 `c3dc32c5c55372af3c469aa1768d377443cef264e61cfc35f223a57f61450847`
- report：`le-wm-main/results/p1_four_arm_winner_root1_20261031_RUN01/report.json`，
  SHA-256 `5e58003df2d3d70eb87843789b2bb3b0994750ce5dcad2ee4825325a64f14838`
- saved check：`le-wm-main/results/p1_four_arm_winner_root1_saved_20261031_RUN01.json`，
  SHA-256 `f2da283612b6ffc38c2b4358cad86b49f82e4b2c3868f48fb19ae04796c3b29f`
