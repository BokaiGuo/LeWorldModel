# P5 固定真实观测刷新：主机 CUDA 执行

2026-10-30，按固定的 164 枝、每枝五个时间点，在已开放的历史物理面板上执行真实观测刷新的一步预测；未训练模型、未读取未开放 holdout。此运行与先前开放五步滚动使用不同的观测信息，不能直接视为公平闭环规划对照。

- 刷新监督源码 SHA256 `d5ab673a87a15460a3875d80266cad75819d37d092070f9605667ac7e832a289`；固定输入 SHA256 `46bfde766c6ce6d99c95cba0c37909c4437dcfbdb9e5e049da02ca0d83403409`。相关离线测试 31 项通过。
- 主机静态前置 [报告](../le-wm-main/results/p5_rgb_host_preflight_20261030_codex_v1_REFRESH_MAIN01.json) SHA256 `f64fa54c600a9205d43a0b68f22f9e49caaa000dd337e4d26f7e268e7a02815b`；驱动三项版本一致，设备节点可读写，磁盘可用约 30142 MiB。此报告本身刻意不授权启动。
- 机器级 systemd 首次请求因需要交互认证而未启动任何作业；随后在主机用户级服务 `lewm-p5-refresh-20261030-main01.service` 执行。服务的控制组、30 分钟上限、10 秒停止宽限、`KillMode=control-group` 和 `SendSIGKILL=yes` 已检查。GPU UUID 为 `GPU-4e8973ff-0278-53f3-2bae-2615d461e441`，总显存 16311 MiB；启动阈值为空闲 8192 MiB、总量 15000 MiB，子任务硬限时 1500 秒，runner 调用间限时 1200 秒。GPU 空闲快照并非资源预留。
- 服务正常退出，监督器报告 `complete_self_report_only`。监督日志 SHA256 `9825d3e46d084a777b68a5b700db9250031bee87094a7c015be2d005af42d6a6`；事件日志 SHA256 `3737d1c786c571dee55113779e4521a404ce0275d0babbbde75268be41f5fb26`。
- 保存验收 [结果](../le-wm-main/results/p5_rgb_refresh_supervision_20261030_codex_v1_accept_MAIN01.json) SHA256 `a1be8db350ad913fbabc4d87ebbbe5d47e14ea57121dc9aa4710f42294b0f799`，状态为 `self_report_consistency_only_not_scientific`，P5 准入保持关闭。验收路径与生产监督器属于同一源码，不能声称独立硬件证明。

固定输入为 164 × 5 个时刻，每个时刻依次编码真实观测并预测一步。运行和调用验收本身不证明刷新降低误差或解决模型失真；下方另行报告保存数据比较。

## 保存数据对照

后续已完成同动作、同时间的 [物理比较](../le-wm-main/results/p5_rgb_refresh_openloop_physical_compare_20261030_codex_v1_MAIN01.json)，SHA256 `6685430d507ddc808b500356e094622816fbc180bd2f3bca73c12cd3d56b51c8`；来源与指标合同见 [说明](../le-wm-main/innerdocs/p5_rgb_refresh_openloop_physical_compare_20261030_codex_v1.md)。固定 164 枝、off/on 各 82 枝，每枝同 plan 6–10 真实状态比较，所有六轴各自保留。新增消费者测试 13 项通过。

在第五时刻，block-y 平均绝对误差从开放滚动的 off/on `2724.72/2705.79` 降为逐步真实观测刷新的 `17.53/10.75`；第一时刻两路径逐枝预测相同。但这不是各轴普遍改善：第二时刻 block-x 的 off/on 误差由 `67.05/70.52` 增至 `85.69/85.41`。第五时刻是五步开放预测对一步刷新预测，信息与预测时域不同；这是 teacher-forcing 诊断，不能证明刷新控制器的公平闭环收益、唯一因果机制或 P5 通过。82 枝均来自共享的已知根，不是独立任务样本。

后续 [递归合同审计](../le-wm-main/innerdocs/p5_rgb_recursive_contract_audit_20261030_codex_v1.md) 将训练损失、原规划递归和本地 runner 的源码逐层对照，未发现可确认的二次归一化或误将 delta 当作下一状态的错误。固定面板中，从第 2 到第 5 时刻，开环递归输入与真实帧重编码输入的六维归一化 L2 差异均值由 off/on `0.456/0.458` 增至 `9.228/9.180`。这只是不同输入间的描述性差异，不能直接推定局部 Jacobian、唯一根因或训练分布外；同输入 CUDA 接口对照及扰动实验仍待执行。
