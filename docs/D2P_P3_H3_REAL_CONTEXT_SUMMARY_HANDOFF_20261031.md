# P3 全量物理配对：保存消费准备

`experiments/spectral_reachability/p3_h3_real_context_physical_summary_20261031.py`
在全部 18 批完成后，重建已封存清单、对每批重新消费事件链与
PNG，并检查第 15 步 guard 在第 16 步派发前、状态／快照／图像
与原轨迹一致。只在全部批次通过后生成配对表；两个填充对标识
并剔除。提前停止不补固定二十步终点，最后状态和原生成功另存。
按原／新动作及根任务身份去重，若同输入的物理结果不一致则
拒绝静默合并；结果仍只是两个已知根，不是独立任务数。

实际第 0 批测试覆盖正常事件、guard 顺序与快照篡改拒绝；
另外测试固定分母、填充排除及重复结果不一致拒绝。共 3 项通过。
截至本记录，第 0–3 批已经完成保存验收，剩余批次仍在运行；
未基于部分批次汇总效果。

完整运行结束后：

```bash
.venv/bin/python -m experiments.spectral_reachability.p3_h3_real_context_physical_summary_20261031 \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/p3_h3_real_context_physical_summary_20261031_RUN01.json
```

上述命令应在 `le-wm-main` 中执行；它只消费保存结果，不运行
神经模型。任何批次缺失或验收失败都不能视为科学负结果。
