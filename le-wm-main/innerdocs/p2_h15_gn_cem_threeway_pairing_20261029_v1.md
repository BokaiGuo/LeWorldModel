# H15、CEM、GN：两根同目标同返回步物理配对

三套 physical saved consumer 和其上游 HOST consumer 已重跑；逐根核对 task SHA、物理根、warm trace、目标、目标姿态与目标像素。三方法两根均完成 warm10 后返回步 1–15，无缺失或真实早停。这里的“同时间”是同一个返回控制步，不是同一实时物理状态；动作与轨迹不同。每步原值与三组差值见完整 JSON。早停后的 null 行为另以 fixture 验证，不把末值延续。

第15步，顺序均为 H15 / CEM / GN；原生距离、位置误差单位沿用 native，角度为弧度：

| 根 | native distance | object position error | wrapped angle error | success |
|---|---|---|---|---|
| 0 | 164.290 / 222.067 / 203.958 | 40.119 / 32.151 / 26.728 | 0.266927 / 0.152419 / 0.155198 | false / false / false |
| 1 | 163.371 / 190.588 / 513.902 | 46.381 / 44.204 / 143.868 | 0.000246 / 0.174686 / 0.482462 | false / false / false |

每根神经资源 attempted/completed/caps 一致，顺序 H15 / CEM / GN：

- predict API、samples；action encoder calls、samples；predictor calls、samples；pred_proj calls：各 `132 / 132 / 117`。
- predict、action encoder、predictor context tokens 及 pred_proj rows：各 `396 / 396 / 351`。
- Jacobian API：`1 / 0 / 4`；VJP directions、grad API、raw backward callbacks：各 `192 / 0 / 768`。
- 其余账本轴 `encoder_calls/samples`、`projector_calls/samples`、`model_SVD_calls`、`array_SVD_calls`、`solve_calls`、`signal_compile_calls`、`program_compile_calls`、`replay_calls`、`metric_calls`、`schedule_decisions` 均为账本记录的零。**零仅表示该轴未被这些接线计数，不代表相关 CPU 工作或重放不存在**：H15 谱库内部 SVD/ridge 和 GN 四次 numpy 求解没有完整记入对应轴，多个 replay/metric 也在外层执行而未通过该轴计量。

每根每方法物理 attempted/completed 均为：constructor 1、close 1、reset 2、set_state 4、goal_render 2、goal_setter 3、goal_bind 1、public_step 25、physics_step 254、explicit_render 81。两根共享上限各为上述两倍，完整逐根全部轴、caps、源 SHA 和操作耗时均在 JSON；账本轴之外的活动不能推定为零。

FLOPs、CUDA 时间与 backward-engine 调用数均未知/null。HOST 全流程墙钟 H15/CEM/GN 分别约 70.57/65.53/75.83 秒，含不同装载、记录、审计及 OS 条件，非纯计算时间；物理墙钟约 14.86/15.93/16.19 秒。预测样本和 VJP 不同，共同动作域不等于等总预算，不能由此断言算力或速度优势。

两根都是已知根，独立物理时刻的回顾性描述，不是 fresh-root 泛化或随机因果比较；多指标也不构成单一综合排名。消费者是 saved 证据重验，不是重跑物理、也无法证明不存在未记录调用。

报告 SHA256 `7f033a9323ab3a4fa4346f49f2d29cd7a0e725ce18bf66e5fb9e80f662658e88`；复核命令：

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.p2_h15_gn_cem_threeway_pairing_20261029_v1 verify --output /home/gbk/code/LeWorldModel/le-wm-main/results/p2_h15_gn_cem_threeway_pairing_20261029_v1_CODEX01.json --sha256 7f033a9323ab3a4fa4346f49f2d29cd7a0e725ce18bf66e5fb9e80f662658e88
```
