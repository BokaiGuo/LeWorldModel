# 真实 cp5 暖启动实验交接

固定规则见 `D2P_P1_UPHILL_CP5_WARMSTART_PROTOCOL_20261030_MAIN.md`。
主机 CUDA 完成全部 24 个前缀各自的十步后缀规划；报告 SHA-256
`0334d3485c6631b357b7aada1855767b221800d684955d0b932cd7d4c7c00688`，
manifest SHA-256
`b9fd1ba1b6233132b2ad3ba9b8013fa5f88faf598ee81dba53b20d9ece0f6bad`。
GPU 设备为 RTX 5060 Ti，冻结参数和 buffer 的运行前后哈希一致。
物理准备过程已逐项消费保存规划账本，准备文件 SHA-256
`5a8eed8cc3bbca6e496121ed6a6bca2fc99b130e4a3173fa6c9ca9dd319264b5`。

物理运行目录为
`le-wm-main/results/p1_uphill_cp5_warm_physical20_20261030_run_RUN01`。
运行先重建来源，目录可能延后出现。查询进程和目录后再决定
后续操作；不覆盖、不重复运行或将部分分母当结果。完整
`run_report.json` 状态为 `recorded_not_consumed` 后独立计算
其 SHA-256，从 `le-wm-main` 运行：

```bash
.venv/bin/python -B -m experiments.spectral_reachability.p1_uphill_cp5_warm_analysis_20261030 \
  --directory /home/gbk/code/LeWorldModel/le-wm-main/results/p1_uphill_cp5_warm_physical20_20261030_run_RUN01 \
  --report-sha256 ACTUAL_REPORT_SHA256 \
  --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/p1_uphill_cp5_warm_physical20_20261030_prepare_RUN01.json \
  --prepare-sha256 5a8eed8cc3bbca6e496121ed6a6bca2fc99b130e4a3173fa6c9ca9dd319264b5 \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/p1_uphill_cp5_analysis_warm_20261030_RUN01.json
```

该分析验收完整物理保存态，按相同前缀和任务列出独立后缀的
实际物理指标及停止/删失，不能将共同初始化说成公平强基线；
规划账本的分类值目前是上界而非经过覆盖认证的全流程实际成本。
