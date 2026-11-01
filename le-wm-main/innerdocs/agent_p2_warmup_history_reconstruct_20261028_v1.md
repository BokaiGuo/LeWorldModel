# 已知四根缺失 warmup 历史的最小 collector

本版本只重建固定缺口 warmup10，不重选计划、不执行选中动作、不运行模型。MAIN 授权后才可调用 `run`。不改既有 inventory、producer、物理记录或共享文档；P1/P2/freshness/scientific_admission 均 false。RAM 测试不构成物理重建证据。

## 固定操作及输出

先调用原 `collect_nonterminal_boundary_contact_roots(count=4, seed=202753010, oversample_factor=4)`，按原顺序核全部四个 state+goal fingerprint。保留原预定 nonterminal collector recipe，不增加 outcome 筛选。仅对固定缺口 slots 1/2/3 调用原 `_reset_and_warmup(history_len=3)`；其底层 root-anchored helper 执行原 reset、三包各五次 zero step，返回真实边界5/10/15。通过 env proxy 记录，不 monkeypatch 原文件或 globals，不自行设置 goal_pose。

三根合计保存12张实际 PNG：每根 goal、warmup5/10/15。journal 记录 reset seed/state/goal、每次 step 后原生 state7、terminal flags、零动作见证、边界 state 和 distinct observation event ID。重复 render 需同 state/同 pixels，不能拿同帧改名充当时间历史。三帧堆叠和 goal pixel SHA 必须与原 sealed context 完全一致，warmup15 state7/goal/raw-history 也必须精确相等；满足后同一根的两臂共享该真实历史。root0 使用已有证据，明确 `already_saved`，不伪造本轮返回；四根/八臂分母始终保留。

原 actual step5 PNG/动作/goal 已由 inventory 闭包，本次不重做选中前缀、不 render step5、不编码、不预测。新报告不是原8臂物理报告的改写，也不是 stageB 的16predict adapter。

## 预算与范围

计划 attempts：factory1、collector reset32、缺口 root reset3、warmup step45；selected action0、encoder/projector/predict/AD0。prepare 的 planned_returns 为 null，只有真实 ledger return 才记入实际返回。每次 public call 先 fsync attempt，返回后立即 fsync return，后处理失败不抹掉已发生的返回。异常/native stop 立即停止，保留失败和未运行行，不补步、不重试、不换根。计数不做等成本优势或科学推断。

计量仅 public factory/reset/step/render/close；reset 内部 setter、goal bind、render、space.step 和 constructor 内部工作均 unknown/null，不能把 public32+3 resets 当内部全账，也不声称 factory 前安装 native hooks。调用原 helper 时原生 stop 包括 stop 在第15步也保持失败，不拿失败当理论反例。

事前工程限制：新独占64MiB输出，200MiB保留空间，900秒 call-boundary/final wall（非抢占 watchdog）；源码、依赖、原资产、模型文件哈希前后核对。预留1MiB终态空间；close 清理仍记 attempt/return，允许在 dispatch cap 后执行，最终超限记失败而非阻止清理。所有写入 exclusive，无覆盖封存资产。依赖原 helper 的模块导入可能导入 torch，但本 collector 不加载模型、不执行神经算子，无 CPU fallback。本 worker 仅测试 RAM。

## MAIN 命令

在 `/home/gbk/code/LeWorldModel/le-wm-main`，以下 SHA 必须用真实已封存文件的实际值，不填写未来返回或未来报告 SHA。环境变量为任务专用。输出路径必须尚不存在。

```bash
P2_WARMUP_INVENTORY_SHA=$(sha256sum results/agent_p2_midpoint_triplet_inventory_20261028_v1_MAIN01.json | cut -d' ' -f1)
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_warmup_history_reconstruct_20261028_v1 prepare --inventory results/agent_p2_midpoint_triplet_inventory_20261028_v1_MAIN01.json --inventory-sha "$P2_WARMUP_INVENTORY_SHA" --output results/agent_p2_warmup_history_reconstruct_20261028_v1_prepare_MAIN01.json
P2_WARMUP_PREPARE_SHA=$(sha256sum results/agent_p2_warmup_history_reconstruct_20261028_v1_prepare_MAIN01.json | cut -d' ' -f1)
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_warmup_history_reconstruct_20261028_v1 verify --prepared results/agent_p2_warmup_history_reconstruct_20261028_v1_prepare_MAIN01.json --prepare-sha "$P2_WARMUP_PREPARE_SHA"
# 仅 MAIN 审读并明确授权实际物理后：
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_warmup_history_reconstruct_20261028_v1 run --prepared results/agent_p2_warmup_history_reconstruct_20261028_v1_prepare_MAIN01.json --prepare-sha "$P2_WARMUP_PREPARE_SHA" --output results/agent_p2_warmup_history_reconstruct_20261028_v1_physical_MAIN01 --authority MAIN_APPROVED_KNOWN4_WARMUP_RECONSTRUCTION_ONLY
P2_WARMUP_REPORT_SHA=$(sha256sum results/agent_p2_warmup_history_reconstruct_20261028_v1_physical_MAIN01/report.json | cut -d' ' -f1)
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_warmup_history_reconstruct_20261028_v1 consume --prepared results/agent_p2_warmup_history_reconstruct_20261028_v1_prepare_MAIN01.json --prepare-sha "$P2_WARMUP_PREPARE_SHA" --report results/agent_p2_warmup_history_reconstruct_20261028_v1_physical_MAIN01/report.json --report-sha "$P2_WARMUP_REPORT_SHA" --output results/agent_p2_warmup_history_reconstruct_20261028_v1_consume_MAIN01.json
```

## 测试与停点

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p2_warmup_history_reconstruct_20261028_v1.py tests/test_agent_p2_midpoint_triplet_inventory_20261028_v1.py
```

RAM env 使用真实 packet-contract 的五次逐步执行和边界纯度检查，但它不是 PushT。测试覆盖32+3+45、完整分母、四根身份、精确 history/goal/state、native stop/异常不补步、独占工件、attempt/return 区分、saved 变异拒绝及冷导入拦截。未运行 physics、模型或网络。

实际联合回归：本测试 + frozen inventory/history_lineage/teacher_suffix/teacher/native_gap，237 passed in 43.08s。实际 saved `build()` 仅在 RAM 构建 prepare，四个原根身份均匹配、3640个 source/input bindings 校验通过；没有落地 prepare 或执行 physics。首次本文件 RAM 回归31 passed，补完 reset/cleanup/严格消费回归后本文件35项；不把合成成功当真实 collector returns。

依赖 MAIN 先验证 frozen inventory/原闭包可用并审读本实现；本 worker 不声称已获得新 warmup10 returns。真实重建可能因漂移/终止/依赖哈希/工程限制失败，届时保全失败，不改门槛。后续仍需将新历史与原 warmup15/actualstep5 全8身份连接，并评估 public-rollout/manual 固定16predict 对照需要哪些额外真实编码；本版本不擅自复用不对应的 latent，不宣称 C3/P2/P1 通过。
