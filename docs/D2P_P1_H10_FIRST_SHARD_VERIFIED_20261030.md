# P1 十步三前缀干预：首锚点独立保存检查

运行 `codex_p1_three_prefix_h10_runner_20261030_run_MAIN01` 的首个锚点已经结束；绑定的 `anchor_00/report.json` SHA-256 为 `a7a0d63978c6b7cc54c9e91dcdbef16e532e0665e6b70ded728f9551dadc0d86`。使用固定准备文件重建并调用保存消费者 `check_shard`，该锚点 42/42 后缀均为 `probe_completed`，无删失、原生早停或工程错误。

该检查覆盖这个锚点的动作、绑定轨迹、逐帧文件、原始物理状态和资源日志；这是 selected 前缀单臂，其他前缀和锚点仍在运行，无法由此计算三前缀比较或 P1 机制结论。完整 1008 分支必须等全运行结束后经整体保存消费者验收。

随后同一固定准备文件下的 `anchor_01`（zero）也经 `check_shard`
独立保存检查，42/42 为 `probe_completed`，报告 SHA-256 为
`466ecfda8bad61d880ace2e897e33d84dbfc486ed4a0a4fca8f9cb6034bbfc39`。
该检查不代表整次运行完成；greedy 和其余锚点仍需完成及验收。

首组 `anchor_02`（greedy）随后也通过同一 `check_shard` 检查，
42/42 为 `probe_completed`，报告 SHA-256 为
`c6399c49caf84b653bc9df751bb2fffa9fc65d49736fe08fe8a8648767202691`。
这完成了首组三个前缀的分片验收，但不是完整固定面板，也不是独立确认。

第二组的 selected 锚点 `anchor_03` 随后亦通过 `check_shard`，
42/42 为 `probe_completed`；报告 SHA-256 为
`9aeccc89c620a86167e36744e0835f450746ad306fe2aaf3f3404368d56e2b36`。
完整运行仍在进行，不能以该分片代替完整面板分析。

第二组的 zero 锚点 `anchor_04` 亦通过 `check_shard`，42/42 为
`probe_completed`；报告 SHA-256 为
`7ef25bb3703b0375080a7c22fcd2f1246f820f02fd86d439bf49665f8f82fa79`。

第二组 greedy 的 `anchor_05` 同样通过 `check_shard`，42/42 为
`probe_completed`；报告 SHA-256 为
`e35f819f5678f579be771e0628452fee14f46831c51fbba670bfc3ea03f5a750`。
这仍只是分片验收，完整面板与机制分析必须等待全部 24 个锚点。

第三组 selected 的 `anchor_06` 通过 `check_shard`，其中 41 条为
`probe_completed`、1 条为 `native_stopped`；报告 SHA-256 为
`058bd84877fcfbdaf0d0b3440defd1f964048dcc7452f7b4400e1b347ce17989`。
原生早停按实际事件保留，不能补造十步终点，也不能将它计为完整后缀。
