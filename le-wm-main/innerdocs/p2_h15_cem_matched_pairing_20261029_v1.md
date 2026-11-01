# 同根 H15 与共同域 CEM：真实物理 saved 配对

两根均为已使用 off 根。两份物理 consumer 和其上游 HOST saved consumer 被重新执行；逐根对齐 task SHA、完整原物理 root、warm trace、goal、goal pose、goal pixel 与条件；按真实 returned control step 1–15 配对。两条轨迹及动作并不相同。时间点是 warm10 后第 n 个控制返回，不是同一个实时物理状态。两根两方法均完成第15步，没有本次真实早停；额外 fixture 验证一方第10步停止后第11–15步均为 null，无末值延续。

第15步（H15 / CEM，较小是该指标数值更小；不是综合排名）：

| root | native distance | object position error | wrapped angle error radians | native success |
|---|---:|---:|---:|---|
| 0 | 164.290 / 222.067 | 40.119 / 32.151 | 0.266927 / 0.152419 | false / false |
| 1 | 163.371 / 190.588 | 46.381 / 44.204 | 0.000246 / 0.174686 | false / false |

逐步完整数值与 H15 减 CEM 差值在结果 JSON。每根每方法实际 132 predictor samples、132 predictor API、396 predictor context tokens；H15 另有192 VJP directions / 192 grad API，CEM 为0。每根两方法各25 public steps（含 warm10）、254 physics steps、81 renders。逐项 attempted/completed/caps 和完整物理 axes 均在 JSON；FLOPs、CUDA 时间及 backward engine calls 为 null，不以零替代。HOST wall 还含不同的模型、来源检查、保存审计和 OS 条件，不能解释为纯模型速度；CPU候选构造也未完整计量。共同动作域和相等预测样本不等于总预算或总 FLOPs 相等。

已知根、回顾性、非独立泛化；固定 CEM 协议在该 CEM 运行前确定，但 H15 结果已知，不能称盲态预注册。saved 消费可检查保存闭包，不能证明不存在未记录调用或独立重演物理。

报告 SHA `b20f02286623e5c2f390bed70051d58abe136a139d825ddd078472773de0eb05`；复核：

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_p2_h15_cem_matched_pairing_20261029_v1.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.p2_h15_cem_matched_pairing_20261029_v1 verify --output /home/gbk/code/LeWorldModel/le-wm-main/results/p2_h15_cem_matched_pairing_20261029_v1_CODEX01.json --sha256 b20f02286623e5c2f390bed70051d58abe136a139d825ddd078472773de0eb05
```

如需新建一份报告，使用 `build --output` 和一个此前不存在的、以 `p2_h15_cem_matched_pairing_20261029_v1_` 开头的绝对路径；不得覆盖现有产物。
