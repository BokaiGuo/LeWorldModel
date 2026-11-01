# P1 非赢家物理对照启动记录

按 `D2P_P1_ALL_NONWINNER_PHYSICAL_PROTOCOL_20261031_MAIN.md`，
固定三个非赢家 rank，每个 rank 24 条；无结果筛选。准备文件
和 SHA-256：

- rank 0: `p1_uphill_nonwinner_physical_20261031_prepare_r0_RUN01.json`
  `5ce3e38a1c86945383a92eb28d735aa95a0f043dda6709b875b9d244fa38aad2`
- rank 1: `p1_uphill_nonwinner_physical_20261031_prepare_r1_RUN01.json`
  `45aea30301f95f79ca99527c8233e83f84b70fd02eb2f8c5df281b77445c4de1`
- rank 2: `p1_uphill_nonwinner_physical_20261031_prepare_r2_RUN01.json`
  `e02c138e6cfe40230ddecce9cf052259d67085ffad5a1f9f06ee783ead175d84`

主机启动进程 PID 619149、619176、619202，对应输出目录
`le-wm-main/results/p1_uphill_nonwinner_physical_20261031_run_r{0,1,2}_RUN01`。
启动后核查进程存活，但尚未看到输出目录，处于执行前重建阶段；
不能据此认为任何物理轨迹已完成。PID 和状态必须重新检查。

全部 `run_report.json` 出现且状态为 `recorded_not_consumed` 后，
使用实际报告 SHA 分别调用同模块的 `consume --rank R --directory
ABS_RUN_DIR --report-sha ACTUAL --prepared ABS_PREP --sha256 PREP_SHA
--output ABS_CONSUME`；输出名须以
`p1_uphill_nonwinner_physical_20261031_consume_rR_` 开头。先验收
所有 72 条再比较已有 24 条赢家，保留早停，不按结果选样本。
已有合成测试覆盖全部非赢家索引与动作篡改拒绝，不能替代物理
验收。这是事后已知根探索，不是独立确认或 P1 科学通过。
