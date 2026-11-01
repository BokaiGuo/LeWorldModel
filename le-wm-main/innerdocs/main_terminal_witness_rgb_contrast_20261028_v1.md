# MAIN 独立旧/新 terminal RGB 配对

源码与测试为同stem，结果 `results/main_terminal_witness_rgb_contrast_20261028_v1_MAIN01.json`，SHA `b740fe3e086d8e49c6bb4b0c7dbd3016d1f5761194060bba8991384bb022c67d`。

完整来源、主机CUDA strict及四条件数字见项目文档 `docs/D2P_SELECTOR_CAUSAL_FACTORS_MAIN_20261028_v1.md` 最新追加章节。此工具不运行神经或物理，不重新拟合Q，不筛根；只按外部固定report/receipt SHA读取原192维NPY并独立复算。

比较的任务、root、condition、goal、Q、context与时刻10相同，动作各自绑定，只有index0动作完全相同。相同动作的state/预测/实际latent逐位一致；其它三条预测和实际Q都下降，但jointXY两条损害、angle三条损害。不能将此结果外推为旧80对没有反转，也不能把实际编码Q当成功标准。全4条件、2known roots、fresh0；P1和science仍false。

差分恒等式保留共同误差、动作响应误差及交叉项，负项不删除，不能据此拟合部署修正或认定唯一根因。12项算术回归不算科学结果，完整上游验收由两个各自producer完成，本独立工具明确`full_upstream_rebuilt_here=false`。

```bash
env OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q tests/test_main_terminal_witness_rgb_contrast_20261028_v1.py
```
