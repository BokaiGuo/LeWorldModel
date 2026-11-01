# E3 根 0：完成首次中止前未执行的两臂

原合同四臂固定；首次运行的 structured 已在第 7 步原生成功，
native-style 在第 2 步观察到推杆 y=514.295 超出画布而由既有
fresh RGB/canvas guard 拒绝，random 和 immediate-progress 均为
`not_started`。保留所有已执行和失败结果，不重试前两臂，也不
把该失败归类为模型理论负结果。

本合同仅执行原四臂清单中未启动的 random、immediate-progress，
其目标、动作初始化、优化预算、真实反馈时点、物理参数和停止
规则保持原合同。单独输出和成本记录，不覆盖原有产物。它不是
重新选根或按成功结果筛选；两次运行的四臂需保留实际分批与
native-style 删失，不能称四臂均完整成功或全流程公平。

原准备 SHA-256 `f486478d34a1d3d2018ae5895bf7a16b1c6ab713fdcffa643006a8e7f3254032`；
失败报告 SHA-256
`97c249e57ae85ce7a676dc63d0f9880b5521d1c4a524cfefe7cbf13dab92a6cb`。
