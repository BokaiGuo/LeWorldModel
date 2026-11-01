# 联合优化前缀 H10 运行交接

准备文件 `le-wm-main/results/p1_joint_h10_20261030_prepare_RUN01.json`，
SHA-256 `71579e186113a71d5571006de8519633950b05c32e887bb86563db1bba946b34`。
执行目录 `le-wm-main/results/p1_joint_h10_20261030_run_RUN01`；固定
24 锚点、1008 分支。启动时已认证新 selected 的实际前缀轨迹，
以及 old zero/greedy 的同根、目标、条件、动作身份。单元测试
`tests/test_p1_joint_h10_20261030.py` 通过。

作业启动后必须先查询进程与执行目录，不因本交接记录推断进程
仍存活，也不在作业存活时重新启动或覆盖。只在完成全部 24 个
shard 且 `run_report.json` 的状态为 `recorded_not_consumed` 后，
独立固定该文件的 SHA-256，再运行
`experiments.spectral_reachability.p1_joint_h10_consume_20261030`。
消费器逐 shard 核查动作字节、帧、终止/删失和资源回执后输出
逐行物理指标，缺失终点保留 null。任何部分结果、失败、异常退出
都先审计代码及输入，保留原始目录，不当成科学负结果。

这是已知根的事后开发干预，不训练模型、没有神经推理；结果不能
替代各前缀独立后缀规划、统一完整资源预算、真实反馈闭环或新根
确认，不能打开 P1。消费器和正式后续结果尚未运行。
