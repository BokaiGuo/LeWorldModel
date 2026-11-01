# E3 根 0 cp5 目标：固定四臂及中止后续跑

在根 1 的开发结果后，对事先已有的另一根及该根历史 control5
目标运行同一四臂主机 CUDA／真实反馈开发合同。目标初始未原生
成功；真实目标图像、latent、原生目标状态、起始根和历史经绑定。
根 0 接线及错误根拒绝测试 2 项通过。

| 臂 | 控制步数 | 状态 | 终点物体位置误差 | 终点角度误差 |
| --- | ---: | --- | ---: | ---: |
| structured | 7 | 原生成功 | 6.6579 | 0.04391 |
| native-style | 2 | 画布 guard 拒绝，删失 | 不补终点 | 不补终点 |
| random | 4 | 原生成功 | 8.6283 | 0.07619 |
| immediate-progress | 6 | 原生成功 | 6.6579 | 0.04391 |

首次运行执行 structured 后，native-style 第 2 步的真实推杆
坐标 y=514.295，触发既有 `fresh RGB/canvas guard`；动作和失败
观察均已保留。该 guard 反映超出所要求的画布范围，不能未经
证据宣称为 CUDA 或理论失败。报告中 native-style 的 `final_metrics`
对应先前被接受的状态，不能当成拒绝后的终点。执行器按既有
fail-stop 流程中止未启动的两臂。

只对原清单中未启动的 random 和 immediate-progress 另起固定续跑，
不重试失败臂、不覆盖原记录。续跑的 3 个已到达节点通过保存态
目标、动作、账本和 Python autograd API 检查；未独立复演完整物理
事件链。structured 的结果来自首次运行报告，未使用完整成功报告
专用消费者对整个失败批次认证。两批必须分开保存，native-style
保持删失，不能汇报为四臂均成功。

这些目标来自已知轨迹；两根及多个臂不是独立测试任务。成本组成
仍不等价，也没有官方强基线认证。结果不证明独立多任务泛化、
E3 或 P1 通过。

- 首批报告 SHA-256
  `97c249e57ae85ce7a676dc63d0f9880b5521d1c4a524cfefe7cbf13dab92a6cb`
- 续跑报告：`le-wm-main/results/e3_cp5_four_arm_root0_remaining_20261031_RUN01/report.json`，
  SHA-256 `e24aedab551d2f22c2b70ced736c31ed96f76cd06effc48af95689db5685b7d1`
- 续跑保存检查：`le-wm-main/results/e3_cp5_four_arm_root0_remaining_saved_20261031_RUN01.json`，
  SHA-256 `4ff251c24751bb839e497986592bf94f6a86ac8f4bf8ec5931965a0353a9859b`
