# 重选前缀的独立规划物理执行

CUDA 规划已完成，物理准备程序复核了全部 24 条规划保存账本，
并绑定每个前缀、cp5 RGB 和十步动作；准备文件 SHA-256
`7984bcd309a008ed63a4d0752be5726361b30e5042eb129cc42d72f7be1e42b9`。
物理输出目录为
`le-wm-main/results/p1_uphill_cp5_physical20_20261030_run_RUN01`。

启动后先确认进程和目录的实际状态，不能仅凭本文件判断存活；
执行进程需要先重建和核查来源，目录可能延后出现。严禁覆盖或
重复启动；若失败或中断，保留失败现场并检查代码和身份绑定。
完整 `run_report.json` 的状态为 `recorded_not_consumed` 时，
独立计算其 SHA-256，再使用以下命令验收保存态：

```bash
.venv/bin/python -B -m experiments.spectral_reachability.p1_uphill_cp5_physical20_20261030 consume \
  --directory /home/gbk/code/LeWorldModel/le-wm-main/results/p1_uphill_cp5_physical20_20261030_run_RUN01 \
  --report-sha ACTUAL_REPORT_SHA256 \
  --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/p1_uphill_cp5_physical20_20261030_prepare_RUN01.json \
  --sha256 7984bcd309a008ed63a4d0752be5726361b30e5042eb129cc42d72f7be1e42b9 \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/p1_uphill_cp5_physical20_20261030_consume_RUN01.json
```

此作业只有物理执行，没有神经推理。先前的神经规划在主机 CUDA
完成。该批为已知根开发探索，不是 P1、共同后缀机制或等预算
强基线证明；提前停止的预定终点不得插补。
