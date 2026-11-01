# P1 新前缀共同多起点规划与物理执行

24 条真实 cp5 RGB 规划已在主机 CUDA 完成，模型前后哈希一致。
规划文件
`le-wm-main/results/p1_uphill_multistart_adam_20261030_main_RUN01.json`
SHA-256 `018dffcb53e5e4dd40e6323fc3665ca53321cb027023c0cc4c1edb5daa06229f`。
固定四起点中选中 0/1/2/3 的记录数分别为 11/3/5/5；这些是
重复臂及共享根，不是独立任务，也不是物理成功。按代码路径记账，
每条 168 次 action encoder、168 次预测、80 次 backward；这些
不是覆盖全流程的独立 hook 账，也不含前缀生成成本。

物理准备
`le-wm-main/results/p1_uphill_multistart_physical_20261030_main_prepare_RUN01.json`
SHA-256 `c58e71cf4a5f42d4ab5668993e95869d64718a3b3d5ed37172816dc980e82ff8`。
绑定了 24 条 RGB、原始前缀、起点哈希、赢家规则和后缀字节；
五项测试覆盖来源绑定和篡改拒绝。物理运行目录为
`le-wm-main/results/p1_uphill_multistart_physical_20261030_main_run_RUN01`。
启动后需核验实际进程和报告，不能仅凭本文判断完成。此阶段
没有神经推理；不得使用模型分数代替真实执行结果。

完整 `run_report.json` 状态为 `recorded_not_consumed` 后，计算
实际文件 SHA-256 并在 `le-wm-main` 执行：

```bash
.venv/bin/python -B -m experiments.spectral_reachability.p1_uphill_multistart_physical_20261030_main consume \
  --directory /home/gbk/code/LeWorldModel/le-wm-main/results/p1_uphill_multistart_physical_20261030_main_run_RUN01 \
  --report-sha ACTUAL_REPORT_SHA256 \
  --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/p1_uphill_multistart_physical_20261030_main_prepare_RUN01.json \
  --sha256 c58e71cf4a5f42d4ab5668993e95869d64718a3b3d5ed37172816dc980e82ff8 \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/p1_uphill_multistart_physical_20261030_main_consume_RUN01.json
```

仅在全量验收后按根、条件和角色比较实际停止、成功与终态；
早停不得补固定时域终点。这是已知根、事后开发对照，尚无
全流程等预算认证，不是 P1 通过或共同后缀能力干预。
