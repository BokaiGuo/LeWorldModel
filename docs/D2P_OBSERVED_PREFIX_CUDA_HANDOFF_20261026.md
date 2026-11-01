# 真实前缀反馈固定后缀预测：已完成 CUDA 校准

最新状态（2026-10-26 22:46 后）：主机审批恢复，实际进程退出 0。产物 `le-wm-main/results/pusht_observed_prefix_prediction4_v1_20261026_cuda/report.json` 为 completed，6 项 gates 全部 true：四根、64 prediction samples、参数不变、缓冲区不变、eval、输入不变。全部 64 次前缀重放的 RGB 哈希与物理状态精确一致，无后缀重优化、选择改变或 CPU 回退。下文审批超时为先前历史，不能继续描述为当前阻塞。

| 根 | imagined-prefix 终态向量误差 MSE 中位数 | observed-prefix 终态向量误差 MSE 中位数 | 误差下降候选数 |
|---|---:|---:|---:|
| 0 | 0.05172 | 0.00856 | 16/16 |
| 1 | 0.21070 | 0.05551 | 15/16 |
| 2 | 0.22327 | 0.01615 | 16/16 |
| 3 | 0.50838 | 0.41358 | 14/16 |

按根哈希与候选 ID 连接原编译、原物理执行及新预测，使用原封存投影计算真实终态编码预测误差。61/64 候选的预测误差下降；后三列为校准描述统计，不是独立根显著性。第四根即便真实反馈后仍有较大局部预测误差；不能声称 feedback 已修复全部动力学问题。

这一对照支持真实观测更新能减少本批开环误差传播，但原程序和真实终态未改变，不能宣称任务收益或 P1 通过。下一项效力证据仍需冻结反馈后重编译策略与等反馈预算强基线，在独立根上真实执行。

目的：区分 imagined-prefix 误差传播与真实前缀条件下的局部后缀预测误差。不重新优化后缀、不改变程序选择、不打开 P1。

新 runner：`le-wm-main/experiments/spectral_reachability/run_pusht_observed_prefix_prediction.py`。

读取已封存 64 程序及真实执行报告；逐根重放原前缀，要求真实 RGB 哈希及物理状态与旧分支逐元素一致；用 F2,F3,F4 及 A1,A2,A3 作为观测上下文，固定原封存后缀 A4 预测终态。计划 64 次 CUDA prediction，另有观测编码。参数、缓冲区、eval、来源绑定及运行前封存都需核验。

2026-10-26 两次主机执行请求均在自动权限审批阶段超时，均未创建进程，不是 CUDA 运行失败或实验负结果。未切换 CPU。不要将本页或源码存在当作作业存活证据。

在 `le-wm-main`，主机审批通过后执行：

```bash
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/lewm-matplotlib .venv/bin/python -u -m experiments.spectral_reachability.run_pusht_observed_prefix_prediction --execution results/pusht_d2p_sealed_program_execution4_v1_20261026_cuda/report.json --policy /home/gbk/.stable_worldmodel/checkpoints/pusht/lewm --output results/pusht_observed_prefix_prediction4_v1_20261026_cuda
```

只有最终 report、全部 gates 与真实进程退出码核验后才能记录完成；若存在失败目录，不覆盖，使用新版本路径并保留失败产物。上下文移位与残差分解相关测试 9 项通过，仅为数组合同测试，不是预测有效性证据。
