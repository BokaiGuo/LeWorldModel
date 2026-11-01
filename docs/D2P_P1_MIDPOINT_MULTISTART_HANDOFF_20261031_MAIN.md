# P1 真实中点多起点反馈运行交接

固定 24 条真实中点上，主机 CUDA 完成四起点 Adam 后续规划；
模型与缓冲区哈希前后一致。规划结果 SHA-256
`2d3c69d56e50a7a051e5316a56a1b90854446e3aecbb5e59ff44fdf6bed627b8`。
每条按代码路径记录 2 次编码、85 次动作编码和预测、80 次
backward；不等于完整成本认证。随机起点测试 1 项通过。

物理准备文件
`le-wm-main/results/p1_uphill_midpoint_multistart_physical_20261031_prepare_RUN01.json`
SHA-256 `9d56daea430e6e0090ae35daaac0d0ddcfe734981063c7d45e71a954bc40b232`；
已核查所有赢家规则、候选动作范围和原第一包身份。物理运行目录
`le-wm-main/results/p1_uphill_midpoint_multistart_physical_20261031_run_RUN01`
已启动；启动不等于完成，须重新核查进程与完整报告。

完整报告 `recorded_not_consumed` 后，按实际报告 SHA、上面的
准备文件和 SHA，调用同模块 `consume`，输出绝对路径须以
`p1_uphill_midpoint_multistart_physical_20261031_consume_` 开头。
还需与原轨迹逐条核对第一包的状态、快照和停止标记，再比较
开放、单起点反馈和多起点反馈。早停不补固定时域终点；
共享根不是独立样本。实验属于已知根开发，不是等预算对照、
独立确认或 P1 通过。
