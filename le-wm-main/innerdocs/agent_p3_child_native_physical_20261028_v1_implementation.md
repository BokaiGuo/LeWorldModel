# P3 observed child 原生物理桥实现（待 MAIN 审核执行）

独占源码及测试为 `agent_p3_child_native_physical_20261028_v1`；本实现文档另名
`_implementation.md`，不覆盖已经冻结的同 stem 协议文档。不修改 producer、阈值、配额、根或旧结果。

## 输入与严格闭包

固定真实 compiler report SHA `885f69f6de5ce9b78b8f3acdc84f14ef8891ba3ecd79912fc9af4e1920fc7ffb`，
完整 strict SHA `d6322717231d72bb18e26eab1acbf6afe2dbc65eaa601af1fface21083a7e91c`。
先检查所有外部 authority SHA，再完整调用原 compiler.consume 并逐字规范化比较完整 strict 返回。
继续完整调用 P4.consume(output,report_sha,prepared,prepare_sha)，保留其预算/操作账/full arrays/seal 检查。
拒收 partial producer 返回，不以 rolling supplement 替代原 producer。所有 prepare source、源码、报告、PNG
SHA 入新 prepare 的 sources 闭包；verify 完整重建，而不是只信 report 字段。

补查原 P4 journal 的全 warmup0..10、plan0..boundary PNG、state 与全部 body snapshot，包括
position/velocity/unwrapped angle/angular velocity/force/torque/sleep flag/float64 bits。
native obs angle 是 unwrapped angle %2pi，其余六维 exact；不能把 snapshot angle 加2pi蒙混通过。
snapshot 不含 solver warm-start cache，不能声称从 snapshot setter 完整恢复；实际必须原 seed/placement + 连续重放。
核对原 sourcepack blind 的 warmup RGB/state/dynamics、真实 raw10 非零历史、源 control_return、task export、goal RGB、
child context 与 boundary observed event ordinal。所有 observed 节点包括预算拒收节点先核元数据，不按结果筛选。

## 真实执行协议与有限预算

run 默认拒绝；仅 MAIN 显式授权后调用原 instance-local metered native_factory 与纯物理 P4.setup。
不调用其神经 run/encoder。对41条分别新建实例，恢复原 seed/placement、原 warmup10，然后逐位执行该节点真实
executed_prefix。每一 recovery PNG/snapshot/goal 都在写入日志后、下一动作前与原 frame exact 核验。
边界再封完整 context/raw10/selection，执行 selected raw5，H3 其余动作不释放。不共享实例或借用 root J。
5 adaptive 原整节点预算拒收与2 not_reached 保持 unknown，完整24child/48method不删。

41分支 boundary 合计385：warmup410 + executedprefix385 =795恢复 public 上限，selected205，总1000。
constructor41、close41、reset123、set_state246、goal_render123、goal_setter164、goal_bind41、
explicit_render1082、native internal physics10246（含 reset246，public 内10000）。
这是嵌套资源，不相加成“总计算”或FLOPs。所有 attempted/completed/未返回 pending保留，不退费、不重试。
native earlystop 不 padding；故障停止后续 executable，仍保全48行。cleanup即使工程预算触发仍尝试，单独记账。

输出总cap512MiB，normal384MiB/finalreserve128MiB；journal80MiB、每PNG256KiB、diskfloor200MiB。
1082*256KiB +80MiB journal+8MiB其他normal余量低于384MiB；每次实际写与dispatch仍 failclosed，
这不是每张图必能压到256KiB的承诺。wall600s作用域为verify后执行入口（含source读取/cleanup），
终态report写入不含在600s内，不能声称whole-job600s。prepare/verify/consume只做saved算术，0新神经。

native success 原生 joint XY（agent+object四维）与原 angle term；native7 reward为七维距离负值，含velocity，
不能当物理任务成功。另列object position、wrapped angle、velocity误差。不声明object-only成功阈值。
P1/P3/C3/compute advantage/independent confirmation/scientific admission全部false。
这些是同两根observed子节点的工程校准，不是41独立根，不等实际资源，也未实现完整树。

## MAIN 精确入口（所有文件参数绝对路径）

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -m pytest -q /home/gbk/code/LeWorldModel/le-wm-main/tests/agent_p3_child_native_physical_20261028_v1_test.py
/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -m experiments.spectral_reachability.agent_p3_child_native_physical_20261028_v1 prepare --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_child_native_physical_20261028_v1_prepare_MAIN01.json
```

prepare终态打印实际SHA，以下 `ACTUAL_PREPARE_SHA` 必须替换成该值，不预造未来SHA。

```bash
/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -m experiments.spectral_reachability.agent_p3_child_native_physical_20261028_v1 verify --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_child_native_physical_20261028_v1_prepare_MAIN01.json --prepare-sha256 ACTUAL_PREPARE_SHA
/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -m experiments.spectral_reachability.agent_p3_child_native_physical_20261028_v1 run --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_child_native_physical_20261028_v1_prepare_MAIN01.json --prepare-sha256 ACTUAL_PREPARE_SHA --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_child_native_physical_20261028_v1_MAIN01 --authorization MAIN_P3_CHILD_NATIVE_PHYSICS_ONLY
/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -m experiments.spectral_reachability.agent_p3_child_native_physical_20261028_v1 consume --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_child_native_physical_20261028_v1_MAIN01 --report-sha256 ACTUAL_REPORT_SHA --receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_child_native_physical_20261028_v1_consume_MAIN01.json
```

run只打印compact summary，consume持久化完整rows，NEW文件/目录排他。`ACTUAL_REPORT_SHA`来自实际run终态。
本worker未运行prepare/physics/神经/实际consume，也不写MAIN产物。RAM/saved回归不是科学证据；
真正接线/原生逐位重现、实际资源、早停与工程失败须由MAIN审读后执行验证。
