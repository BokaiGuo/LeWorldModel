# 重选动作物理执行交接

固定来源
`le-wm-main/results/p1_uphill_physical_20261030_source_RUN01.json`
SHA-256 `2f8c75e502794248c399385e01db5cc991fb69b1eab7f34e4676becd1fe04e81`；
准备文件
`le-wm-main/results/p1_uphill_physical_20261030_prepare_RUN01.json`
SHA-256 `d9ac91f50bafeb012b99c414d2ded081bffc79dd4ccfbcf9f606f806dbc80a74`。
执行器
`le-wm-main/experiments/spectral_reachability/p1_uphill_physical_20261030.py`
复用既有物理合同，按
`D2P_P1_UPHILL_PHYSICAL_PROTOCOL_20261030_MAIN.md` 逐项绑定
CUDA 重放结果、重选输出、预热动作和 native 物理源。

作业已启动，独占输出目录
`le-wm-main/results/p1_uphill_physical_20261030_RUN01`。
后续已完成：report SHA-256
`b1fbe4b4811cc8d110aca0d4f90edef13382321ecd21dbac66f1f893ddbda825`；
独立保存态验收及逐对分析通过，见
`D2P_P1_UPHILL_PHYSICAL_RESULT_20261030_MAIN.md`。
先检查进程和 report，不以本交接文件推断运行或完成状态；不要
重启存活进程，不覆盖、删除既有结果。完成后独立计算 report
哈希，以 `consume --join-sha <上述来源哈希> --report-sha <报告哈希>
--output <上述运行目录>` 复核全部来源、执行及保存态。
如果出现工程错误或部分运行，先排查代码和输入，不作为科学负结果。

该合同是已知根、固定 15 步动作探索；原联合 winner 与重选动作
的前后缀都可能不同，不能把终点差归因前缀机制或称公平强基线。
此物理作业没有神经推理，不改变冻结模型或 P1 gate。
