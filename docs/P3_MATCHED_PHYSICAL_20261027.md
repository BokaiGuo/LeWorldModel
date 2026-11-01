# P3 保存路径物理执行 v1

仅工程校准，P1 未通过、P3 不准入。不修复当前 Q 的 agent/object 失配，不训练、
不拟合 readout、不重选根、不调阈值、不重复规划。仅新增 driver、测试与本文件。

## 输入与端点

输入原 P3 CUDA report 与其原 protocol、主线程 main_v2 horizon correction、
原 compilation 及官方 policy。重算原 correction，逐 root identity 验证全部 8 条
路径；绑定 protocol hash、旧源码 SHA、correction archive seal、compilation、
checkpoint/config、原物理 root bank/upstream、精确动作 bank、目标和 Q hash。
正确预测 context 保留 emb3 + incoming[1:] 两个 between-state 动作。

注意：corrected search horizon 与 selected best path horizon 不是同一字段。
实际执行 5 条 H2（10 raw steps）、3 条 H3（15 raw steps）；不把固定组称为固定
H 评估，不补零，不比较假想同一时间终点。每条路径两次独立 reset/warmup 后物理
重放，16 个分支全部保留，比较 state/command/guard/终态及真实不同 endpoints。
每次派发前用现有 RoyalBlue RGB guard，丢失观测即停，无盲继续。

## 落盘与失败语义

物理执行独立于模型：每次 env.step 前写 attempted journal，返回后写 step_returned，
每个 RGB 与观测保存 PNG/hash/state；fsync 后继续。异常 env.step 不退款。
重放的 reset/encoder/evaluator/guard/KeyboardInterrupt 失败都保留分支；物理 setup
失败产生所有未执行槽的 blocked receipt。不成功分支也在固定分母内。
每分支 physical.json、全体 physical_report.json 在可选 CUDA 加载/编码之前落盘。
编码只读取已持久化且 hash 匹配的 terminal PNG；编码失败另写 encoding.json，
继续保留其他分支，最终 report fail closed。无法写盘/磁盘耗尽不能承诺报告保存。
编码失败绝不回退 CPU。新 predict calls=0；无规划、无反向。

计数分开记录 selected-path attempted/returned env.step、completed primitive steps、
planned H 和已纠正 search max H。两次 replay 不当作一次物理预算；reset/warmup
成本为独立共享设置，不包含在 selected-path step counters，非全流水线等预算。
官方 evaluator/pose 仅后验报告，不进入 dispatch 或选根。guard 中止不是任务成功。

## 主线程命令（此任务不执行物理/CUDA）

在 `/home/gbk/code/LeWorldModel/le-wm-main`，先预检（不分配 CUDA、不创建 env）：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m experiments.spectral_reachability.p3_matched_physical_20261027_v1 \
  --p3-report results/P3_ADAPTIVE_RUNTIME_20261027_cuda_v1/P3_ADAPTIVE_RUNTIME_report_20261027.json \
  --correction results/P3_ADAPTIVE_RUNTIME_ACCOUNTING_AUDIT_20261027_main_v2/report.json \
  --compilation results/pusht_d2p_metric_program_preflight4_v1_20261026_cuda/report.json \
  --policy /home/gbk/.stable_worldmodel/checkpoints/pusht/lewm \
  --output results/P3_MATCHED_PHYSICAL_20261027_v1_cuda \
  --encode-terminals --preflight-only
```

主线程 HOST CUDA 执行：同一完整命令删除 `--preflight-only`。不带
`--encode-terminals` 则纯物理执行，无神经计算。输出目录 exclusive，不可覆盖。
最多 190 selected-path env.step（含独立 replay，不含 warmup）、206 PNG，
16 次 batch1 CUDA 终态编码、0 predict。保守预留 100 MB 磁盘、4 GiB 显存、
10 分钟窗口，均为估计不是实测。2.4 GB 空间不需复制权重或大数据。

测试包含真实保存路径校验及纯非神经环境 fixture 的 H2/H3、初始/中途不可见、
终止、失败/中断成本、物理先落盘、编码失败、重放差异。不是物理或 CUDA 效益证据。
尚缺主线程真实运行验证；即使完整性门通过也不能推导 P1/P3、compute advantage、
物体控制修复或独立根泛化。
