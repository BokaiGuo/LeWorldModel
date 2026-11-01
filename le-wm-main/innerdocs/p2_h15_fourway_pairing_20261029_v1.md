# H15、CEM、GN、projected-gradient：两根逐步配对

四套 physical saved consumer 和上游 HOST consumer 已重新验证。逐根核对 task SHA、完整原物理根、warm trace、目标、目标姿态及像素；四种方法两根均完成 warm10 后返回步1–15，无真实缺失。这里的同时间是相同返回控制步，并非同一实时状态：四次执行的动作和物理轨迹不同。完整 JSON 保存每步四方法原值及六组方向固定的成对差值。fixture 验证早停后为 null，不延续末值；若两方法仍有当步则保留其配对。

第15步，顺序为 H15 / CEM / GN / gradient：

| 根 | native distance | object position error | wrapped angle error (rad) | native success |
|---|---|---|---|---|
| 0 | 164.290 / 222.067 / 203.958 / 136.335 | 40.119 / 32.151 / 26.728 / 43.677 | 0.266927 / 0.152419 / 0.155198 / 0.267256 | 均 false |
| 1 | 163.371 / 190.588 / 513.902 / 173.257 | 46.381 / 44.204 / 143.868 / 59.445 | 0.000246 / 0.174686 / 0.482462 / 0.059996 | 均 false |

每根实际神经账本 attempted/completed/caps 均一致，按上列顺序：

- predict API、samples；action encoder calls、samples；predictor calls、samples；pred_proj calls 各为 `132 / 132 / 117 / 117`；对应 context tokens 和 pred_proj rows 各 `396 / 396 / 351 / 351`。
- Jacobian API 为 `1 / 0 / 4 / 4`；VJP directions、grad API、raw backward callbacks 各为 `192 / 0 / 768 / 768`。
- 其余账本轴 encoder/projector calls 和 samples、model/array SVD、solve、signal/program compile、replay、metric、schedule 为记录的零。它们不是这些操作真实总数：部分 CPU SVD/solve、评分和外层 replay 没有由该轴完整计量。各轴 attempted、completed、cap 及嵌套计时见 JSON。

每根每方法物理 attempted/completed 均为 constructor 1、close 1、reset 2、set_state 4、goal_render 2、goal_setter 3、goal_bind 1、public_step 25（含 warm10）、physics_step 254、explicit_render 81；两根共享 caps 为各两倍。FLOPs、CUDA 时间、backward engine calls 均未知/null。HOST 全流程墙钟 H15/CEM/GN/gradient 为约 70.57/65.53/75.83/71.32 秒，包含装载、记录、审计和 OS 条件，不是纯计算或速度对比。

共同动作域、部分计数相同不代表等总算力或等总预算；两根已知且结果已观察，不是 fresh-root 泛化或随机因果比较，多指标不支持单一综合排名。saved audit 不是独立物理重演，不能证明未记录调用。本次不执行神经模型、GPU 或物理。

报告 SHA256 `8ac4255e86fe112f9a7c0f7e0d8303df67cce4323ea493ba30a0d72e2d6b4dc0`：

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.p2_h15_fourway_pairing_20261029_v1 verify --output /home/gbk/code/LeWorldModel/le-wm-main/results/p2_h15_fourway_pairing_20261029_v1_CODEX01.json --sha256 8ac4255e86fe112f9a7c0f7e0d8303df67cce4323ea493ba30a0d72e2d6b4dc0
```
