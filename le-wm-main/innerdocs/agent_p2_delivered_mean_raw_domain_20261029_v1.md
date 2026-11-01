# P2 delivered mean：离线 raw-domain / bit round-trip

仅新增 `agent_p2_delivered_mean_raw_domain_20261029_v1` source/tests/doc。MAIN批准的authority仅是**原P2 known-root工程约定**，不是Hi原StandardScaler或训练stats认证。冻结controller的四returned和两arbitrary missing全部保留，不挑target、不clip/tanh/reselect、无部署许可。experiment-code技能用于source-first、日志及RAM回归；训练、绘图、生产执行均不在本slice。

## 原始authority与检查入口

固定实际controller完整三件套（相对inner项目results）：

| 文件 | SHA256 |
|---|---|
| agent_p2_target_controller_cuda_20261029_v1_prepare_MAIN01.json | 1a021116784dfaa5b4ce976f694fbfde377e6b282ce3553cb57d49a5e033c0aa |
| agent_p2_target_controller_cuda_20261029_v1_cuda_MAIN01/report.json | b1358eb3cec42cdddb30b0d138c2bc2e68773f195aa985977bf9c090527382cb |
| agent_p2_target_controller_cuda_20261029_v1_consume_MAIN01.json | 34e0491ca05a1b381db3541997c6215c44f9927761d6123a559e604160f0be4a |
| agent_p2_three_program_physical_20261029_v1_prepare_MAIN01.json | a4fb18b5aedd53e1820717f827d4703774b8c41dd08b0fd3974cca58593ade93 |

导入以上**完整固定字节**与完整controller strict receipt，检查report→prepare/receipt→report、四local_check精确join、全六槽身份、全局原model before/after。此处不调用controller.verify/consume或core.validate_saved，也不读取模型NPY/weights或重建旧fullstrict。仅读每槽trace及实际selected mean/native_return actions/native_return mean三份小NPY，核文件inventory/SHA/shape/dtype/EOF/data SHA，要求三者逐位一致；绝不以elite均值/最优candidate代替返回mean。

原normalizer从 `experiments/run_pusht_counterfactual.py:34–35` 用AST读取 PUBLIC_ACTION_MEAN/STD 的float32原字面量：mean `[-0.0087,0.0068]`，std `[0.2019,0.2002]`。原文件SHA `0a4eb4592951139e596b35becede17d175df7b48a2fde16679ab0aac7c4dce34`，报告保存**真实float32值及bits**，不是十进制重新估计。不能调用该文件另一个 `normalize_public_action` 的历史 `/100` 分支。

原Jacobian映射是 `pusht_full_packet_compiler_20261027.py::make_transition:95` → `pusht_d2p_packet_contract.py::normalize_live_packet:90–115`：raw packet `[5,2]` time-major展平10，先减tile(mean,5)、后除tile(std,5)，无额外 `/100`。源码固定SHA分别 `f359e7b21d1970f95f065dbee206d2dc9359c2eaff7ddbcd7611e07686c6cf2a`、`eda631348e929b45591807dac0adcbb72d10d84b7d48a7a558a492eab9c04aef`。`preflight_pusht_d2p_p0.py:80–91`也明确同一坐标修正，其source SHA固定在SOURCE_PINS。

再核原program receipt的 `/plans/1、3` normalization与上述f32精确一致；task/blind身份与delivery一致、原physical prepare的整个planner_plan相等；原task压缩文件字节SHA核对但不decode环境/轨迹。用原warm10做一次前向算术，必须精确恢复原两条between-action history与本次controller causal action2。这绑定原task和实际Jacobian约定，不将公共常数冒认为Hi训练数据真值。

physical native source为 `.venv/lib/python3.12/site-packages/stable_worldmodel/envs/pusht/env.py`，固定SHA `c6d2104adc368ff1c352edc0e35ac74da9f07fda3a748ddbef0ba6b7a2d1daa2`，须匹配原physical prepare.bindings。AST核 `PushT.__init__:73–75` 的Box `[-1,1]^2/float32`、relative默认true、action_scale100。其step:318–319原生把relative raw乘100转位置目标；本工具不调用step，也不把100再带入归一化/逆变换。SOURCE_PINS全7份当前SHA还必须存在于原controller sources或physical bindings，不仅自称source正确。

## 一次f32算术和完整报告

输入只接受真实finite float32 `[1,3,10]`，按C索引展平为 `[15,2]`：H0的5个XY、H1的5个XY、H2的5个XY。无截断、padding、额外normalize、重复搜索或数值修复。

严格一次 NumPy `multiply(dtype=float32)` → 独立 `add(dtype=float32)` 得raw；再独立 `subtract(dtype=float32)` → `divide(dtype=float32)` 得roundtrip。五阶段normalized/product/raw/centered/roundtrip完整原字节hex保存，不融合FMA。每槽全部30坐标保存packet/within_packet/axis/raw_step、原/新f32值与bits、domain、bitexact、ULP与各阶段零的正负号；所有越界、所有bit差异、所有零出现与符号改变再完整列出，不只报告max值。

ULP定义为finite IEEE单调rank距离；`+0/-0` rank距离0但**bits不等**，signedzero差异独立记录，绝不以数值相等或ULP=0替代bitexact。derived inf/NaN保存bits、数值null/finite=false/ULP=null，domain阻断，不让JSON输出非法非有限数。

raw逐坐标finite且在闭区间[-1,1]才domain_ok，恰好端点允许，超出1ULP也拒绝。越界status=`blocked_native_domain`，roundtrip不逐位相等则 `needs_paid_actual_token_replay=true`；若同时发生两者均完整记录，status优先domain。domain合法但roundtrip有任意bit差异，status=`needs_paid_actual_token_replay`。两项均通过也仅 `bitexact_in_domain_checks_only_not_released`。

**所有状态**：raw_deployment_authorized=false、original_prediction_cost_reuse_authorized=false、CUDA_token_identity_certified=false。NumPy镜像不是实际CUDA的token重建证明；有差异必需MAIN另定预算对实际部署token做付费replay，不得沿用原mean预测成本；越界还须先处理科学/执行合同，不能clip后replay冒称原动作。本slice不自动付费、改预算或重选target。prior物理时距仍null，不声称公平/time-matched/三proposal收益。

## MAIN运行命令与边界

只读保存文件算术，无NN/physics/fullstrict入口。prepare输出本stem单个≤16MiB JSON，floor200MiB，排他写与file/parent fsync；verify完整重建所有绑定/选定数组/检查结果。失败或旧文件保留，不覆盖。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q tests/test_agent_p2_delivered_mean_raw_domain_20261029_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_delivered_mean_raw_domain_20261029_v1 authorities
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_delivered_mean_raw_domain_20261029_v1 prepare \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_delivered_mean_raw_domain_20261029_v1_prepare_MAIN01.json
# RAW_DOMAIN_MH 填上述实际返回SHA，不预造。
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_delivered_mean_raw_domain_20261029_v1 verify \
  --manifest /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_delivered_mean_raw_domain_20261029_v1_prepare_MAIN01.json \
  --manifest-sha256 "$RAW_DOMAIN_MH"
```

worker仅RAM算术/AST/窄metadata回归，不对四槽真实mean执行actual build或prepare；由MAIN审读测试后实际运行。测试不冒充实际roundtrip结果或部署证据。旧源/封存结果不改；未认证Hi scaler、raw release、actual CUDA token replay和真实物理部署。

实际回归：指定inner `.venv/bin/python`、`OPENBLAS_NUM_THREADS=1`，35项RAM/AST/metadata/fixture通过。含六槽build编排的显式合成metadata/array callbacks、返回mean三份来源不一致/receipt/slot/task/warmup突变拒绝；这些fixture不认证HOST结果。source初检和CLI不导入torch/pymunk。没有运行真实四mean的domain分析、actual prepare或verify。
