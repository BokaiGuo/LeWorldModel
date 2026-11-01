# H15 GN 固定两根物理接线：等待 HOST 终态

本接线预先固定 root0、root1 的 off 物理根和 warm10，唯一动作来源为对应 root 已完整审计的 GN HOST `result.solution.actions`，原始 15×2 float32；不从候选或分数重新选、不根据物理结果筛根、不 clip/round/tanh。固定两根顺序与 H15/CEM 相同，物理结果是相同初始根在另一次执行，不是同一实时状态，也不是独立 fresh-root 泛化。若上游任一根失败、不完整、无法通过 saved consumer 或结果 SHA 未经用户提供，拒绝生成物理 prepare。

资源上限沿用既有两根物理接线：50 public steps（包括 warm10）、508 physics steps、162 renders 和两个新环境实例；原生早停如实记录缺失，不补帧或重复末值。独立 saved consumer 检查上游 HOST consumer、两根任务/目标/动作字节、逐操作账本、完整日志链、PNG、原生状态与终止；它不是独立物理重演，也不能证明日志之外的行为。prepare、run 和 consume 不加载模型或使用 GPU；物理执行须另行授权。测试仅使用替身环境，不证明真实物理表现。

默认上游路径固定为先前交接命令的 `results/p2_h15_gn_matched_host_20261029_v1_cuda_MAIN01`，其固定 prepare 是 CODEX02；若 session 73899 使用了其他输出路径，须在获知终态后明确确认并修改/重新测试接线，不能猜测或扫结果。当前没有读取该目录或启动物理。以下命令仅在用户通知 HOST 完整成功、提供实际 report SHA 后使用；物理 run 还需另行明确授权。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
# HOST_SHA 是用户通知并核实的完整 report SHA，不预填。
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.p2_h15_gn_matched_physical_20261029_v1 prepare --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/p2_h15_gn_matched_physical_20261029_v1_prepare_CODEX01.json --host-report-sha256 "$HOST_SHA"
# PREPARE_SHA 是上一步实际返回值。
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.p2_h15_gn_matched_physical_20261029_v1 verify --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/p2_h15_gn_matched_physical_20261029_v1_prepare_CODEX01.json --sha256 "$PREPARE_SHA"
# 另获物理执行授权后：
SDL_VIDEODRIVER=dummy PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.p2_h15_gn_matched_physical_20261029_v1 run --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/p2_h15_gn_matched_physical_20261029_v1_prepare_CODEX01.json --sha256 "$PREPARE_SHA" --output /home/gbk/code/LeWorldModel/le-wm-main/results/p2_h15_gn_matched_physical_20261029_v1_physics_CODEX01 --authorization MAIN_P2_H15_MATCHED_GN_TWO_ROOT_PHYSICS_ONLY
# REPORT_SHA 是 run 实际输出，不预填。
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.p2_h15_gn_matched_physical_20261029_v1 consume --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/p2_h15_gn_matched_physical_20261029_v1_prepare_CODEX01.json --sha256 "$PREPARE_SHA" --output /home/gbk/code/LeWorldModel/le-wm-main/results/p2_h15_gn_matched_physical_20261029_v1_physics_CODEX01 --report-sha256 "$REPORT_SHA"
```
