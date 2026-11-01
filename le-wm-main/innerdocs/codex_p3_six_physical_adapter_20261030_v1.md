# P3 六槽 raw10 物理适配进度

固定 root-major、uniform/signal/reverse 六槽；每槽选择已保存赢家的完整 H2 raw10，warmup 原 raw10，无子重规划、缩放、clip 或补步。原生 terminated/truncated 即停并保留实际返回前缀；失败不重试。公共物理资源上限来自既有 moving10 contract：六槽 construct/close 各6、reset12、public step120（含 warm60 + 最多 plan60）、space.step1224、set_state24、goal_render12、goal_bind6、explicit_render132。构造器返回句柄前抛错的内部资源仍 unknown；调用轴嵌套，不可合计为 FLOPs。

独占 `codex_p3_six_physical_adapter_20261030_v1.py` 调用原 moving physical branch，该 branch 内含原 reset/placement、warm10、原历史 RGB/state/snapshot、goal bind、native stop guards。新 wrapper 在 env.step 实际返回后立即写入 sent/actual float32 bytes 和停止标志，再检查动作 ACK、最后恢复 step；ACK 不同即拒绝继续。若 step 已返回但 ACK 验证或日志失败，外层旧 Ops 不一定记录 completed；必须以实际已返回的 ACK event 与外层 attempt 分别报告，不能将未完成 outer 操作当作零物理成本。

`codex_p3_six_physical_saved_20261030_v1.py` 可离线检查单槽实际 ACK 前缀、位级动作、停止与长度，但不认证来源、reset、预算或整批 journal；非完整独立 producer consumer。共7项离线测试通过，包括原分支委托、step恢复、ACK篡改、早停不补步、六槽顺序与动作 SHA 篡改。测试未构造真实模拟器，不证明 runtime 复现。

尚缺六槽独占 host runner、绑定之前完整来源/实际源组件与 prepare、六槽 durable roster/report/失败和 hook/cleanup census、完整独立 saved consumer 的真实 frame/goal/reset/metric 重算；因此没有启动物理或 GPU，不能把 ACK helper 当执行授权。原根是已知根，每策略每根搜索预算不同，历史共享信号与全成本不证明相同；物理固定10只约束请求长度，不改科学 gate。
