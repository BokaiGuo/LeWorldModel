# E3 cell12：目标与起点重合导致首步原生停止

执行固定格 `r1.g1.h1.dDxy` 的已知根探索：已有 control5 来源
`agent_e3_cell12_feedback_prepare_20261029_v1_MAIN01.json` 的 SHA-256
为 `913b0bf7affc1552fbc442fd40ac3dd47bdec001c08a1a6838598ab65b4cdde1`。
启动闭包与 authority 已生成并通过 precheck；precheck 只加载
checkpoint、检查依赖，没有神经前向或物理执行。正式单次执行
使用主机 CUDA，事件记录 `cuda:0`、RTX 5060 Ti 和实际分配检查。

结果目录 `le-wm-main/results/e3_cell12_cooperative_native_20261031_main_RUN01`：
report SHA-256 `f2fbdbd27a2a5d3ee1594386dc7fb1d9a8212a4a0b3e1eaaf84639b8e4b69429`，
completion SHA-256 `66af0b4ea2d3b1f7ffcaff73d38fb0f5533af1534e55ee6a425f67b018c07056`。
原生子进程返回零后，调用既有 chunked identity 与下层保存消费者：
状态 `native_stopped_prefix`，10 个 warmup 步、1 个控制步、13 帧，
该步 `terminated=true, truncated=false`；终止行与所有帧匹配，
但无第 10/15 步反馈边界。未物理重放，严格完整封存和科学准入
均为 false，不能把这一运行计为反馈闭环、任务优势或 E3 通过。

保存事件显示 plan step 0 的七维 goal 与 state 完全相同。源码
`agent_e3_frozen_model_task_resolver_20261029_v1.py` 的 G1 明确
把原 control0 状态设为目标，因此这是该固定任务设计的已实现
行为，不是本次运行中已证实的目标接线 bug。第 1 控制步终止
符合保存消费者按该目标核验的原生规则，reward 是七维距离，
成功则由位置与角度条件定义，二者不可混为同一指标。
该结果暴露的是任务对在线反馈评价的局限：若初始目标即满足，
无法到达后续反馈边界。不能修改已封存结果或把它事后换为新目标；
后续应在另一个事前固定、非初始成功的任务上验证反馈。

使用单机合作锁的入口
`experiments/spectral_reachability/e3_cell12_cooperative_run_20261031_main.py`
保留一次性 claim 与父日志；锁不是跨入口或跨主机全局排他。
入口测试 1 项通过。该运行没有改变 P1 或 P3 的判定。

**覆盖更正**：该格在此次执行前已有
`agent_e3_chunked_identity_host_20261029_v1_CELL12_MAIN01`；
跨任务记录也已将 cell12 标为前缀停止。本次检索遗漏了大写
目录名与合并表述，因此这是对已知槽的重复执行，而非新增槽。
保留两次产物，不把它们计为独立任务；目标等于起点的源码与本次
实际终止证据仍成立。
