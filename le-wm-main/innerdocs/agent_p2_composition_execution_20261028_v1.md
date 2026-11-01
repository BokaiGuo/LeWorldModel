# 已封存 P2 真组合动作的最小执行/消费接线

本 stem 只新增 source/tests/本 innerdoc；不修改封存源码/结果、共享文档或 Jason 的 `agent_kinodynamic_matched_physical_20261028_v1`。完整读取 experiment-code skill 与两份 references；采用先查实产物、定点接线、失败保留和回归，不执行其训练/调优/强制正结果/图表流程。无下级代理、网络、神经或物理作业。

## 查实资产：不重做 producer 或 manifest

已有原四个 known roots，**不是 moving 的两个 roots**：

- `agent_p2_runtime_guard_20261028_v1_known_MAIN01/report.json`，实际 SHA `0cfbe37a7783c6df331305e95e3c058a9b006f72bb1ca5b59d24fae047ee23a0`。
- wrapper prepare SHA `02269ae5d08d1f645e9bcf2473a7c7cbb2a16f3125c3fad62c23859380cc6c48`。
- MAIN 严格 saved 回执 `agent_p2_guard_saved_consume_20261028_v1_MAIN02.json`，实际 SHA `c7d183f5b18d1bfacd81f855025efdbbb403f97e5442650970080766db17088b`。
- `agent_p2_execution_assets_20261028_v1` 已实现该库的8 winner join、原RGB/root/goal/Q/完整history绑定、prepare/verify、MAIN calibration runner，source SHA `e8c1dff3f62de4caf89e83bed0ebaed6dc2d0c4a1da1ade02966f7c1d7e2452d`。

真实 saved build 已只读通过：8 plans、838 bindings，未导入 torch，未新建研究结果。已有480候选、1328预测的完整费用保留在继承库，不重新付CUDA、不宣称全资源相等。条件组合是在真实prefix预测后的完整mid context重编suffix；不是独立预编译库的任意多深度拼接。

Jason 的 matched physical 是两 moving roots、matched-domain beam 的另一份 authority/report/recipe，本模块不导入、不复制、不改该 stem。现有四根 assets runner 缺少独立 strict consumer；这才是本实现补齐的缺口。新 prepare 直接调用原 `assets.prepare`，无新 manifest/schema；新 run 只在显式 MAIN token 后委派原 `assets.execute`，没有第二个物理循环或旧 globals monkeypatch。

## 新 callable/consumer

`consume(output, report_sha256, prepared, prepare_sha256, main_review_sha256)` 只读保存的JSON/PNG；验证：

1. 调用者提供真实外部报告/prepare/MAIN review SHA；原 prepare 全量重建，原库/模型/代码绑定不变；report/execution与原schema精确对应。
2. 固定四根八臂 counterbalanced 顺序、各winner handoff/selection、raw10命令，不重选、不按结果选根。原无组合单full10边、条件组合双5边由既有assets join和MAIN saved receipt验证，不冒充本consumer重复独立CUDA复算。
3. exact artifact census、regular flat files、全部SHA、原fsynced JSONL。原runner无hashchain，consumer不会伪造一个或声称有链式原生记录。
4. 原始公共reset/step的attempt→return/error配对、全局phase索引、root/arm/selection binding、120 warmup/80 selected caps；每臂15真实warmup原命令与已验证normalized history、selected逐命令与branch journal一致，不退款/补步。
5. 逐步原PNG uint8 RGB224，重算原RoyalBlue guard（纯saved像素运算，不是神经/physics），严格initial→attempt→returned→observation顺序；native stop/guard abort不补到10。
6. 真7维state/goal、完整trace、末态/新鲜endpoint。复用原封存labels算术核对原native结果（含agent+object），wrapped角度只核对原pose诊断，不替换native成功或改变角度协议，不用object-only投影判断完整失败。没有新调用installed evaluator；其等价性仍依原runner及MAIN审读。
7. 初始history/goal RGB身份与原refs一致、初始state与journal一致、两臂reset身份严格相等。原warmup三张RGB本身未新持久，consumer只能检验原记录hash==refs，不声称独立重新渲染恢复。
8. 合法partial report仍保留八项；未执行分支结果/步数/native成功为null而非0。缺少/损坏足够真实性证据时直接failclosed，绝不产出伪完整回执。完整工程记录可以包含负结果、native early stop、guard abort，不能等于P2通过。

新返回含 `rows[8]`、public_resources、完整/成功计数、原继承预测账及source seals，全部scientific_admission/P1/P2/freshness/independent_confirmation/physical_execution_authorized=false。工程执行的MAIN token和review SHA是操作防误用/审读引用，不是科学放行证书。

原物理collector/root recipe的constructor、setter、render、goal bind、内部physics/reset工作与整段wall没有完整计量；新consumer将这些显式null，不能补0，也不把公共env.step说成内部物理总调用。900秒/100MiB/200MiB floor为原runner的边界检查而非抢占式watchdog或逐字节硬cap。失败写盘可能不足以生成可验证report，保留原产物，不重试或删除。新consumer可拒绝这类不完整证据，不能逆向补造。

## MAIN 精确命令

在 `/home/gbk/code/LeWorldModel/le-wm-main`；worker仅运行RAM测试和只读verify-inputs：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/agent_p2_composition_execution_20261028_v1_test.py tests/test_agent_p2_execution_assets_20261028_v1.py

PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p2_composition_execution_20261028_v1 verify-inputs

# MAIN审读之后生成原assets prepare；无需新CUDAproducer：
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_composition_execution_20261028_v1 prepare --output results/agent_p2_composition_execution_20261028_v1_prepare_MAIN01.json
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_composition_execution_20261028_v1 verify --prepared results/agent_p2_composition_execution_20261028_v1_prepare_MAIN01.json --prepare-sha256 ACTUAL_PREPARE_SHA

# 仅MAIN审核调度真实physics；worker不执行：
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_composition_execution_20261028_v1 run-calibration --prepared results/agent_p2_composition_execution_20261028_v1_prepare_MAIN01.json --prepare-sha256 ACTUAL_PREPARE_SHA --output results/agent_p2_composition_execution_20261028_v1_physical_MAIN01 --main-review-sha256 ACTUAL_MAIN_REVIEW_SHA --authorization MAIN_APPROVED_P2_KNOWN_ROOT_CALIBRATION_ONLY

.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_composition_execution_20261028_v1 consume --prepared results/agent_p2_composition_execution_20261028_v1_prepare_MAIN01.json --prepare-sha256 ACTUAL_PREPARE_SHA --output results/agent_p2_composition_execution_20261028_v1_physical_MAIN01 --report-sha256 ACTUAL_PHYSICAL_REPORT_SHA --main-review-sha256 ACTUAL_MAIN_REVIEW_SHA --receipt results/agent_p2_composition_execution_20261028_v1_consume_MAIN01.json
```

三种ACTUAL_SHA只能填MAIN实际输出/审读记录，不预填未来SHA。可省略`--receipt`仅stdout读取；可验证已有匹配schema的原assets output，无需重复物理。

## 精确未决

目前未找到上述四根新assets calibration的真实物理报告/prepare；本consumer的物理端只经RAM fixtures，不伪称真实执行消费已完成。MAIN仍须审读/prepare/物理调度/新saved consume。独立P1、fresh认证、原C2广义组合、多层真实child及全资源预算可比均缺失。没有新神经评分、实际任务收益或因果泛化结论；只有4个known roots也不能替代独立科学证据。sealed source hash变化必须由MAIN新版本明确处理，不能本地改pin迁就漂移。
