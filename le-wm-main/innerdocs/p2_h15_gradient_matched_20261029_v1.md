# H15 同域 projected-gradient MPC：固定协议与交接

既有 P2 projected-gradient 是 H2、10 raw steps，不能当 H15；P3 闭环20亦不是本两根共同域的等价对照。本候选是内部 open-loop H15 projected-gradient shooting，不是官方 MPC 复现，也不是 GN：在固定零 logit 初始化上进行四轮，每轮对当前 full 192×30 Jacobian 计算 `2 Jᵀ(terminal-goal)/192`，以固定步长 `1,4,16,64` 沿负梯度移动并投影到 30 维球，四个候选均评分，只有严格改善才替换 incumbent。raw 使用与 H15 相同的 numpy float64 tanh 转 float32；目标是原 identity192 terminal MSE。已知根及其他方法结果已被观察，该协议先于本方法运行固定，但不盲、不独立，不能事后调整或删根。

每根预算：17 次候选/初始评分、16 次双方向差分探针、2 次 whole/segmented 最终重放、4 次当前点完整 Jacobian；实际 ledger 预固定 117 predictor samples、4 Jacobian、768 VJP、768 grad 和 raw backward callbacks。差分探针不参与选择，失败整根 fail closed。用完整 Jacobian 计算标量目标梯度，并不是一次标量 VJP 的省算实现；与 GN 的采样/AD 轴相同，但不保证等 FLOPs、CPU 成本或墙钟。梯度及目标评分的 numpy 运算未由 ledger 精确计量。H15/CEM/GN 与本方法均不构成等总预算。

新增 HOST/slot/solver 和物理 adapter，复用原 model/precision/来源守卫与 saved audit；物理仅接受完整两根 HOST consumer 和所给 report SHA 后各根最终 raw15，不能依物理数据或评分重新挑选。两根 physical 需要独立授权，不在此运行。CPU 替身测试 13 项通过，包含两根 slot、AD/预算/差分、早停与篡改拒绝、梯度与 GN 不同；不能证明真实 CUDA ABI、数值容差或物理可执行性。沿用的限额是 cooperative guard，不能中断单次卡死 GPU 调用。未执行 GPU、物理或训练。

HOST prepare CODEX02 SHA `667ca9e8cd85f0971c453b656a9b5d54dfb06d282388fc3d132140411cadb1e1`；CODEX01 因增加来源绑定已失效，保留勿用。运行前协调 GPU，输出必须不存在。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.p2_h15_gradient_matched_host_20261029_v1 verify --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/p2_h15_gradient_matched_host_20261029_v1_prepare_CODEX02.json --sha256 667ca9e8cd85f0971c453b656a9b5d54dfb06d282388fc3d132140411cadb1e1
# 协调并明确允许 GPU 后：
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.p2_h15_gradient_matched_host_20261029_v1 run --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/p2_h15_gradient_matched_host_20261029_v1_prepare_CODEX02.json --sha256 667ca9e8cd85f0971c453b656a9b5d54dfb06d282388fc3d132140411cadb1e1 --output /home/gbk/code/LeWorldModel/le-wm-main/results/p2_h15_gradient_matched_host_20261029_v1_cuda_MAIN01 --authorization MAIN_APPROVED_P2_H15_MATCHED_GRADIENT_HOST_CUDA
# REPORT_SHA 使用真实 run 输出，完整成功后：
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.p2_h15_gradient_matched_host_20261029_v1 consume --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/p2_h15_gradient_matched_host_20261029_v1_prepare_CODEX02.json --sha256 667ca9e8cd85f0971c453b656a9b5d54dfb06d282388fc3d132140411cadb1e1 --output /home/gbk/code/LeWorldModel/le-wm-main/results/p2_h15_gradient_matched_host_20261029_v1_cuda_MAIN01 --report-sha256 "$REPORT_SHA"
# 仅完整 consumer 后，以真实 HOST report SHA 创建固定两根物理 prepare：
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.p2_h15_gradient_matched_physical_20261029_v1 prepare --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/p2_h15_gradient_matched_physical_20261029_v1_prepare_CODEX01.json --host-report-sha256 "$REPORT_SHA"
# 再由物理 prepare 实际输出的 PREPARE_SHA 及独立物理授权执行：
SDL_VIDEODRIVER=dummy PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.p2_h15_gradient_matched_physical_20261029_v1 run --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/p2_h15_gradient_matched_physical_20261029_v1_prepare_CODEX01.json --sha256 "$PREPARE_SHA" --output /home/gbk/code/LeWorldModel/le-wm-main/results/p2_h15_gradient_matched_physical_20261029_v1_physics_CODEX01 --authorization MAIN_P2_H15_MATCHED_GRADIENT_TWO_ROOT_PHYSICS_ONLY
# PHYSICAL_SHA 必须取实际物理 report SHA：
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.p2_h15_gradient_matched_physical_20261029_v1 consume --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/p2_h15_gradient_matched_physical_20261029_v1_prepare_CODEX01.json --sha256 "$PREPARE_SHA" --output /home/gbk/code/LeWorldModel/le-wm-main/results/p2_h15_gradient_matched_physical_20261029_v1_physics_CODEX01 --report-sha256 "$PHYSICAL_SHA"
```
