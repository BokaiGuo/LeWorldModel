# 全8真实 triplet：full3 编码前置与 public/manual ABI 比较

独占本 stem source/tests/innerdoc；不改 frozen collector、midpoint、suffix、lineage 或 MAIN 文件。仅 MAIN 审读后调用 HOST CUDA；worker 只做 RAM 和 saved 检查，不运行神经、物理、下载或后代代理，不读取 reserved fresh roots。采用 experiment-code 的冻结输入、可复现 journal 和独立资源轴，不采用训练、调参、强求正结果或图表要求。

## 精确输入和必须补齐的编码

绑定原4 known roots、全8已选计划，原始 raw10/suffix5、goal0/Q、normalizer、权重均不变。warmup重建 prepare `376adbeebb4a9f749269448121ed477d16705392fc3bb9fe212b57f3b21a9a8c`、report `89d545d538a75f8ae7778d3d4761651ff5b7866ad2041712b26915b82e303c1f`、strict `151e9ec6a36a0bd189476298f32381efd32d21c98781c0bedac06ac660579330`。prepare 重跑原 frozen consumer 并与 MAIN strict 内容精确相等，不仅验 schema/status。root0 复用原 inventory 认证的实际 warmup10；其余固定3根复用新实际重建10。warmup15 和 actualstep5 均与原8臂帧/action/goal 身份闭合。三种时间角色固定、路径和 observation ID 必须不同；不复制帧、不重新 render、不用相近 pixels 代替同一资产。

旧 teacher full3 的 older2 是原始观测编码，绝非预测历史；但所绑定旧资产中无这批 `[warmup10,warmup15,actualstep5]` 同批 full3 原生 encoder/projector 回执。已有 singleton z5 不冒充该回执。目录名候选检查也没有发现 triplet/full3 专用产物；这不是全盘资产不存在的证明。故当前**不能直接声称0encoder准备就绪**。

本实现交两段可调用真实入口：

- `cache` 前置：全8逐臂原 public `encode`，每臂真实3帧、同批 full3，8 encode API +8 encoder +8原projector，各 encoder/projector 24 samples；0predict/AD/physics。保留八臂，不按相同 pixels 去重；不把原旧2 latent 填进新 full3 以省编码。比较新编码与旧 teacher older2/z5 的 exact/maxabs/MSE，只诊断，差异不触发换根/模型/normalizer/门槛。
- `compare`：要求上述完整cache及其 externally hashed strict，由本版本 consumer 重算相等；全8各 public 与 manual 两路线，共16 predict +16 AE +16 predictor +16 pred_proj，8 rollout wrapper，0encoder/projector/AD/physics。编码前置是本设计的真实成本，不能在最终成本论证中省略。

## 原生 ABI

224 RGB、patch14 从绑定 checkpoint config 推导257 encoder tokens，不硬编码197。原 transform/image mean/std，不改 action normalizer。MATH SDPA；matmulTF32/cudnnTF32/autocast均禁用，逐次 entry 和 before/after precision 校验。模型加载 `.eval().requires_grad_(False)`，前后 state hash 与原 midpoint model state 一致；原 source/weights/input 也前后 SHA 校验。

cache 输入 `[1,3,3,224,224]`，encoder flatten `[3,3,224,224]` → `[3,257,192]`，CLS→原 projector `[3,192]`，缓存 `[1,3,192]`。每个原生输入/完整输出立即独占 NPY 持久，非只存CLS或终态。

public 输入真实六维 pixels `[1,1,3,3,224,224]`、相应 cached emb `[1,1,3,192]`、三 outgoing actions `[1,1,3,10]`（A2/A3/A4）。`rollout(history_size=3)` 的 T=H=3，n_steps=0，恰好一次 predict；不是0predict，也不是3future/额外bootstrap。输出 `[1,1,4,192]` 必须保留前三个真实cache latents，最后为已收费 predict full3 的最后输出。manual 独立 `action_encoder([1,3,10])→predict([1,3,192],AE)`。不调用 criterion/get_cost、不重新编码goal，仍按原goal/Q与既有真实terminal分栏计算。

严格消费验证完整原生 AE/API/predictor/pred_proj 的输入、full3输出、public wrapper 输入输出、原始RGB/归一化/动作buffer与 journalchain；先确认相同输入，再逐buffer报告 exact/maxabs/MSE和首个不等buffer。真实路线差异保留为结果，不强行报失败或调容差。ABI/身份/计量缺失则 failclosed，不把工程失败算理论反例。两路线相同也不能证明共享历史协议正确，更不能证明P2组合机制或物理收益。

资源账分别给 API attempts/entered/returns/saved、actual axis0 samples、temporal rows、inclusive wall。pred_proj16调用48flat rows；nested墙时不能相加。完整4根/8臂/16路线；错误后保留全部分母与未执行行，缺数组不补零，无重试。128MiB输出、200MiB磁盘floor、900s边界/final wall（非抢占watchdog），新prepare有独占launch claim，禁止静默重跑。

继承的重建账已真实发生32 collector resets +3 root resets +45 warmup steps（另有61 public renders及close），由 `inherited_resources.reconstruction` 完整保留。原 strict 的 `new_physics_steps=0` 是 saved consumer 自身新增成本，不代表历史从未做过物理；本stageB的0physics也仅限本版本新增操作。

## MAIN 精确命令

在 `/home/gbk/code/LeWorldModel/le-wm-main`。未来cache/prepare/report SHA 一律取真实生成值，未预造。两阶段都需 MAIN 审读授权，不由worker执行。

```bash
P2_STAGEB_MODULE=experiments.spectral_reachability.agent_p2_stageb_triplet_cuda_20261028_v1
P2_STAGEB_CACHE_PREP=results/agent_p2_stageb_triplet_cuda_20261028_v1_cache_prepare_MAIN01.json
P2_STAGEB_CACHE_RUN=results/agent_p2_stageb_triplet_cuda_20261028_v1_cache_cuda_MAIN01
P2_STAGEB_CACHE_STRICT=results/agent_p2_stageb_triplet_cuda_20261028_v1_cache_consume_MAIN01.json
.venv/bin/python -B -m "$P2_STAGEB_MODULE" prepare --stage cache --output "$P2_STAGEB_CACHE_PREP"
P2_STAGEB_CACHE_PREP_SHA=$(sha256sum "$P2_STAGEB_CACHE_PREP" | cut -d' ' -f1)
.venv/bin/python -B -m "$P2_STAGEB_MODULE" verify --prepared "$P2_STAGEB_CACHE_PREP" --prepare-sha "$P2_STAGEB_CACHE_PREP_SHA"
.venv/bin/python -B -m "$P2_STAGEB_MODULE" run-cuda --prepared "$P2_STAGEB_CACHE_PREP" --prepare-sha "$P2_STAGEB_CACHE_PREP_SHA" --output "$P2_STAGEB_CACHE_RUN" --authority MAIN_APPROVED_KNOWN8_STAGEB_HOST_CUDA_ONLY
P2_STAGEB_CACHE_REPORT_SHA=$(sha256sum "$P2_STAGEB_CACHE_RUN/report.json" | cut -d' ' -f1)
.venv/bin/python -B -m "$P2_STAGEB_MODULE" consume --report "$P2_STAGEB_CACHE_RUN/report.json" --report-sha "$P2_STAGEB_CACHE_REPORT_SHA" --output "$P2_STAGEB_CACHE_STRICT"
P2_STAGEB_CACHE_STRICT_SHA=$(sha256sum "$P2_STAGEB_CACHE_STRICT" | cut -d' ' -f1)
P2_STAGEB_PREP=results/agent_p2_stageb_triplet_cuda_20261028_v1_compare_prepare_MAIN01.json
P2_STAGEB_RUN=results/agent_p2_stageb_triplet_cuda_20261028_v1_compare_cuda_MAIN01
.venv/bin/python -B -m "$P2_STAGEB_MODULE" prepare --stage compare --cache-report "$P2_STAGEB_CACHE_RUN/report.json" --cache-sha "$P2_STAGEB_CACHE_REPORT_SHA" --cache-strict "$P2_STAGEB_CACHE_STRICT" --cache-strict-sha "$P2_STAGEB_CACHE_STRICT_SHA" --output "$P2_STAGEB_PREP"
P2_STAGEB_PREP_SHA=$(sha256sum "$P2_STAGEB_PREP" | cut -d' ' -f1)
.venv/bin/python -B -m "$P2_STAGEB_MODULE" verify --prepared "$P2_STAGEB_PREP" --prepare-sha "$P2_STAGEB_PREP_SHA"
.venv/bin/python -B -m "$P2_STAGEB_MODULE" run-cuda --prepared "$P2_STAGEB_PREP" --prepare-sha "$P2_STAGEB_PREP_SHA" --output "$P2_STAGEB_RUN" --authority MAIN_APPROVED_KNOWN8_STAGEB_HOST_CUDA_ONLY
P2_STAGEB_REPORT_SHA=$(sha256sum "$P2_STAGEB_RUN/report.json" | cut -d' ' -f1)
.venv/bin/python -B -m "$P2_STAGEB_MODULE" consume --report "$P2_STAGEB_RUN/report.json" --report-sha "$P2_STAGEB_REPORT_SHA" --output results/agent_p2_stageb_triplet_cuda_20261028_v1_compare_consume_MAIN01.json
```

RAM regression：`OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p2_stageb_triplet_cuda_20261028_v1.py`。

实际联合回归：本文件42项 + frozen reconstruction/inventory/history_lineage/teacher_suffix/teacher/native_gap，共279 passed in71.25s。包含真实saved全8 triplet/重建strict重消费相等、actual saved cache prepare/verify、relative/absolute cache路径一致性；原生执行仅 NumPy RAM stand-ins。首轮37pass/1fail是fixture声明非零prefix但paid A3置零，已修正fixture并增加stale-prefix拒绝，保留该失败记录；没有修改旧normalizer/源码/科学阈值。命令行 `--help` 已实际运行。

未决：MAIN尚未执行本版本真实cache编码，故16predict准备目前受此精确前置阻塞；无本版本真实预测回执。所有模型/Q与原生目标不同义的限制仍保留，P1/P2/freshness/scientific_admission均false。测试成功只证明RAM合同与saved接线，不是科学结论。
