# Event-gated recurrent memory audit

本轮审计针对接触丰富任务中的短期历史提取失败，使用同一个 JEPA-WM PushT checkpoint、同一 validation episodes、episode-disjoint split 和 fit-only normalization。原有 126-row export 仅提供每个 episode 6 个 stride-5 windows；本轮重新生成 420-row enriched export，每个 episode 20 个 stride-1 windows。

## 已修复的协议/实现问题

1. recurrent 分支此前直接使用 384 维 raw latent，而 Fourier 对照使用 fit-only POD rank 8，导致参数接口和表示预算不匹配。现在 recurrent 与线性对照共同使用 fit-only POD，并在 matched-rank 复核中固定 rank 8。
2. `event_gated_gru` 此前只是把 event scalar 拼接到 GRU 输入，并没有事件触发的状态更新。现在使用 candidate `GRUCell` 与 input-dependent gate 的显式状态混合。
3. temporal evaluator 在写 enriched export 前未创建输出目录；已修复并通过项目 `.venv` 的完整依赖运行。
4. recurrent audit 增加 `--horizon`，可在同一 export 上审计一步和四步 task effect。

## 预先固定的审计结果

五个 episode-disjoint split，matched POD rank 8、hidden size 8、120 epochs：

| horizon | Fourier RMSE | explicit event-gated RMSE | event-gated 胜 Fourier |
| --- | ---: | ---: | ---: |
| 1 | 32.75 ± 4.65 | 33.45 ± 3.55 | 2/5 |
| 4 | 64.35 ± 3.75 | 75.37 ± 6.11 | 1/5 |

普通 GRU 在两个 horizon 上均未在全部 split 超过 Fourier。显式 gate 相对普通 GRU 在 horizon 1 为 3/5、horizon 4 为 2/5 胜出，但这不是跨 split 的稳定优势。

## 结论边界

前两个问题确实是工程/协议问题，修复后 recurrent 误差下降；但在时间支持扩大、表示预算匹配、显式 event gate 和 multi-step horizon 都修正后，Fourier 仍未被稳定超过。因此当前证据不支持“从 learned latent 稳定提取短期记忆并获得稳定任务收益”。这条结果应解释为：latent 中的历史信息可被利用，但当前 event-gated recurrent 表示没有把它稳定转换成超过低阶 Fourier 的 held-out task-effect 预测。

本审计没有 action-ranking、contact-mode successor label、姿态闭合或 online-planning 结果，因此不升级为任务利用或控制成功结论。

## Artifact

- runner: `le-wm-main/experiments/empirical_spectral_audit/run_event_gated_recurrent_memory_audit.py`
- evaluator fix: `le-wm-main/experiments/theory_spectral_transfer/run_jepa_wm_temporal_eval.py`
- summary runner: `le-wm-main/experiments/empirical_spectral_audit/summarize_event_gated_recurrent_memory_audit.py`
- enriched export: `le-wm-main/results/recurrent_memory_export_20261013/jepa_pusht_windows20_metrics_enriched_export.json`
- horizon-1 summary: `le-wm-main/results/recurrent_memory_audit_20261013/explicit_gate_h1_summary.json`
- horizon-4 summary: `le-wm-main/results/recurrent_memory_audit_20261013/explicit_gate_h4_summary.json`
