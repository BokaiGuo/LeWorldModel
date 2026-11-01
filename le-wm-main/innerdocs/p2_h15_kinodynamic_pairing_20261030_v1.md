# H15 四方与 H2 kinodynamic beam：已保存物理记录配对

CPU saved-only 重建两根五方证据。各自完整 consumer 校验 physical journal、来源和保存的图像，beam 另重建其已保存模型搜索。跨导出不假定 task ID 相同：核对原 7 维 root/goal、root SHA、goal pixel、goal-attached 完整物理 snapshot，以及 10 次 warmup 返回状态和停止标志。两套 task ID 和导出 SHA 不同，保留原值；这证明列出的物理绑定，不声称任务文件逐字等同。五方均有真实返回的共同控制步 1–10；beam H2 只执行10步，11–15为 null，不延续末值。

第10步依次 H15 / CEM / GN / gradient / beam：

| 根 | native distance | object position error | wrapped angle error (rad) | native success |
|---|---|---|---|---|
| 0 | 170.562 / 292.569 / 264.390 / 131.444 / 186.609 | 40.119 / 32.151 / 44.739 / 43.677 / 24.636 | 0.266927 / 0.152419 / 0.294210 / 0.267256 / 0.060641 | 均 false |
| 1 | 187.554 / 414.724 / 365.926 / 204.072 / 135.082 | 86.560 / 69.587 / 143.868 / 106.346 / 94.351 | 0.221863 / 0.514983 / 0.482462 / 0.317610 / 0.325283 | 均 false |

这只是已知两根不同次物理执行的相同返回控制步；不是相同实时状态、独立泛化或随机因果比较。native distance 为完整7维距离，包含速度；不同指标不可合并成单一排名。有限 bank beam 并非 RRT 或所有 kinodynamic search，H2 每 packet 的球约束与 H15 联合球及三段规划尺度不同。

每根 beam 144 predictor samples、144 predict API、零 VJP；H15/CEM/GN/gradient 分别 132/132/117/117 predictor samples，VJP 为 192/0/768/768。API 与其内部 predictor 调用重叠，不相加。beam 实际每根物理20 public steps 含 warm10、204 physics steps；四方每根25 public steps 含 warm10、254 physics steps。FLOPs、总 CPU 开销及完整墙钟成本不齐，不宣称等算力、等总预算或等搜索覆盖。

报告 SHA256 `bc12e44fb004ad9924c7ff98b8a4b45dc2646f69ecbd6dac9d3291ba5530a268`。保存重建不等于独立物理重演；本轮没有模型推理、GPU或物理调用。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_p2_h15_kinodynamic_pairing_20261030_v1.py
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.p2_h15_kinodynamic_pairing_20261030_v1 verify --output /home/gbk/code/LeWorldModel/le-wm-main/results/p2_h15_kinodynamic_pairing_20261030_v1_CODEX01.json --sha256 bc12e44fb004ad9924c7ff98b8a4b45dc2646f69ecbd6dac9d3291ba5530a268
```
