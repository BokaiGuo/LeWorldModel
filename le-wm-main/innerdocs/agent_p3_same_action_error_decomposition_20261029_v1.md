# H3 同动作误差分解：saved-only 数学与接口合同

状态：已实现独占 saved 分析器和 fixture 测试，未运行生产消费。编码三件套保持冻结。
本文件不加入已封存编码 prepare 的 source closure，不改变任何旧合同或结果。

## 1. 固定对象与权威

保留四原始分支（known2 roots × endpoints/midpoints），每分支 cp10/15/20 三时刻，
共12个预测误差槽；另保留四个 cp5 anchor 审核槽，总16输入行。
每分支均执行原 prefix5 + 固定 b5 H3 winner raw15，不是原 MPC 后续可能换动作的轨迹。
所有端点均是从 cp5 原观测 context 出发的同一开放环 H3 预测；不是在 cp10/cp15
用真实反馈重新预测。因此不能把相邻误差之差称为局部一步模型误差或反馈修复收益。

唯一字段权威为冻结 `agent_p3_h3_witness_encoding_20261029_v1.py`：

- source SHA256：`c3c7f4421ffe843acc845ba9ef1bc19424ae66e4d08fa96b7a3fc63f22bb4748`。
- prepare 的 `rows[16]` 保存原动作、goal、prediction、task/model/selection 身份。
- report 的 `rows[16]` 保存实际数组描述符；消费必须重建全部真实数组和原生 IO。
- own consume schema 为 `agent_p3_h3_witness_encoding_20261029_v1.consume`，
  status 为 `all16_H3_witness_RGB_verified_NOT_admitted`。
- own consume 的 `rows[16]` 是已分析数值行，不能误把它当 report 的数组描述符行。
- own consume 的 `actual_context_chains[4]` 仅是观测历史，不是新预测或新执行。

未来输入需 MAIN 提供真实 encoder prepare/report/receipt 路径及各自 SHA256；
目前不填未来 SHA、不生成占位执行回执。physical 输入由 encoder authority 继承绑定，
不拿单独 physics 成功标签替代 encoder strict。

## 2. 每分支每时刻的精确恒等式

固定分支 b、时刻 t∈{10,15,20}，d=192。把已保存 float32 向量先提升 float64：
预测 p、真实RGB编码 a、原目标 g。无新归一化、投影拟合或目标重编码。
定义 e=p−a，r=a−g，s=p−g=r+e；Q(x)=〈x−g,x−g〉/d。
符号严格沿用旧消费字段 `model_minus_actual_Q`，即预测成本减真实编码成本。

```
E       = 〈e,e〉/192                          # prediction_vs_encoded_MSE
A       = 2〈a−g,e〉/192                       # signed alignment term
D       = Q(p)−Q(a) = E + A                   # model_minus_actual_Q
D       = 2〈p−g,e〉/192 − E                  # independent equivalent identity
Q(a)    = Q(p) − A − E
abs_Q_error = |D|                            # not |A|+E
```

同时报告 E、A、D、第二种展开和恒等式 residual，不能只报告非负 E 或把 |D| 当 MSE。
D>0 表示模型预测成本更高，D<0 表示模型预测成本更低；不等于物理任务好坏。
A 可为负并抵消 E；Q差为零不证明预测向量准确。E=0 时严格报告 e=0、A=0、D=0。
可选 `alignment_cosine=〈r,e〉/(||r|| ||e||)`；任一范数严格为零则 null，禁止 epsilon。
不需要任何比值即可完成主分解；不计算在 D≈0 时不稳定的“贡献百分比”。

这些是实数恒等式；float64 点积的不同求和次序可产生舍入残差。
保存 `identity_residual=D−(E+A)` 及第二展开 residual，不强制跨求和路径 bitparity。
未来实现的数值门须事前固定由运算误差界推导的浮点检查，不能按实测结果放宽；
原 owned-versus-reference、数组位级身份等已有 strict gate 完全不变。

## 3. 跨时刻、跨分支边界

同分支 t1→t2 可描述（所有符号均为后者减前者）：

```
predicted_Q_change = Q(p_t2) − Q(p_t1)
actual_Q_change    = Q(a_t2) − Q(a_t1)
change_gap        = predicted_Q_change − actual_Q_change
                  = D_t2 − D_t1 = (E_t2−E_t1) + (A_t2−A_t1)
```

固定输出每分支全部三时刻对 (10,15)、(15,20)、(10,20)，共12对，缺值为 null。
这是同轨迹不同时间的描述，不是两个root配对，不增加独立样本量。
cp5 仅可作各自预测/实际的共同起点成本参考；不将它列成第四个 forecast-error 时刻。
实际或预测 Q 的下降、change_gap 改善均不是干预效果、物理收益或因果反馈增益。
不能从累计误差随时刻的变化反推出是哪条模型边造成误差。
跨 bank/跨 root/跨时刻不得静默混合；主表不输出跨分支 winner/ranking 策略。
如将来确需同根跨bank比较，必须另立合同检查同动作/同目标/同时间资格，
不能把本表12点复用成12个独立root，亦不能因某端点可用而删掉其他槽。

## 4. 与当前 saved schema 的精确接线

| 数学或身份 | own consume `rows[k]` 字段 | 额外来源/检查 |
| --- | --- | --- |
| 固定索引 | `index`, `plan_index`, `checkpoint` | k=4b+j；checkpoint=[5,10,15,20][j] |
| 分支身份 | `root_slot`, `condition`, `arm`, `task_id` | condition 固定 off；不按结果重新排序 |
| 原选择 | `selection_digest` | prepare 同行还含 `witness_sha256`, `model_hash`, `task_binding` |
| p,a,g | `predicted192`, `actual192`, `goal192` | 所有可用向量长度192、有限值；由strict绑定原f32数组 |
| Q(p),Q(a) | `model_Q`, `actual_encoded_Q` | 独立从向量重算，不把摘要当真值 |
| D,|D|,E | `model_minus_actual_Q`, `absolute_Q_error`, `prediction_vs_encoded_MSE` | 独立重算并核一致 |
| cp5 | `role=observed_anchor_reencode`, `observed_anchor_reencode_MSE` | 原prepare `observed_anchor192`；必须原精确锚点检查 |
| 缺项 | `latent_missing`, `source`, `physical_status` | predicted存在并不补全actual；保留null |
| 同动作依据 | prepare同行 `full20`, `raw_float32_bits`, `causal_raw10_bits` | report/receipt严格重建后交叉核对 |
| 原生物理 | `native_metrics`, `response` | 仅原样独立列出，不混入latent内积、不再定义success |

当前 consume 行也含 `causal_raw10_bits`，但不含全部 prepare provenance；
因此不能仅接一个无绑定的 rows JSON 就声称已完成同动作认证。

## 5. 纯 saved 分析接口（已实现；继承 MAIN 完整消费，不重跑上游）

```
build_saved_decomposition(*,
    prepared_path, prepare_sha256,
    output_dir, report_sha256,
    receipt_path, receipt_sha256) -> decomposition

decompose_point(predicted192, actual192, goal192) -> numeric_fields
```

前者仅保存文件运算：校验所有显式外部 SHA，验证已封存 encoder source SHA，
核 report 的 prepared/prepare_sha256 正好是参数指定输入，receipt 的 report_sha256
正好是输入 report。遵照 MAIN 后续明确授权，**不调用 encoder consume/verify/build**，
不导入 producer，不重做全上游 strict。MAIN 提供的外部 receipt SHA 是继承完整
消费的信任起点，绝不是本分析器新签发的神经/物理认证。
独立解实际 latent gzNPY，核文件SHA、arraySHA、shape/dtype与report inventory，
逐行核 prepare/report/receipt 身份、原固定 H3 winner/prefix/full20/causal raw10 的
float32 bits、原goal/prediction及所有Q摘要；对 report/prepare/receipt/source 前后重hash。
不重查全部上游bindings、完整encoder IO/事件链、预处理或物理日志；
这些证据仅继承已完成 MAIN receipt。输出 `evidence_scope` 明列上述边界。
后者纯数值，无文件、模型、physics、outcome过滤或阈值选择。
接口拒绝 bool/nonfinite/非192shape、非原生float32向量、重复JSON键，不广播修正维度。

实现 schema `agent_p3_same_action_error_decomposition_20261029_v1.saved`：

- `authority`：外部三个实际 SHA、encoder source SHA、继承 physical report/receipt SHA。
- `anchors[4]`：固定身份、cp5 availability、原精确reencode认证及原因；不伪装forecast误差。
- `points[12]`：固定身份和t、provenance引用、availability/reason、p/a/g、Qp/Qa、
  `error_mse`, `actual_residual_cross_term`, `model_minus_actual_Q`, `absolute_Q_error`,
  `prediction_residual_form_Q_error`, 两个identity residual、`identity_roundoff_bound`、`alignment_cosine`。
  实现字段 Qp/Qa 命名为 `model_Q`/`actual_encoded_Q`，保留原符号。
- `temporal_pairs[12]`：固定分支和三对时刻、两端可用标志及上述change恒等式各项。
- `coverage`：固定4分支/2known roots/12预测槽/4锚点/12时刻对，各自available/missing数量。
- `resources`：新增encode/projector/predict/AD/physics均0，saved CPU wall单列，继承资源不退款；
  `total_FLOPs=null`, `equal_total_compute=false`。
- `P1_admission=false`, `P3_admission=false`, `C3=false`, `physical_benefit_claim=false`。

缺a时依然保留p/g及已知Qp，涉及a的所有数值项null并列原因；
时间对任一端缺则差分项全部null。无可用子集重归一化、填零或终态padding。
必须完整保存全12点和全12对，汇总仅列固定分母和可用数，不通过平均隐藏unknown。

## 6. 数值误差与fixture验收

E=0；r=0；e与r正交；正/负内积；E与A精确相消而e非零；
两个恒等式与时间change恒等式；zero-norm cosine=null；缺项传播；
重复/缺少/错位身份、篡改goal/action/selection或外部SHA拒绝；
cp5不能冒forecast槽；合法ties保留；非有限/bool/错误维度拒绝。
固定数值恒等式检查使用 float64 unit roundoff u，gamma=4096u/(1−4096u)，
scale=mean(s²+r²+e²+2|r e|+2|s e|)，bound=gamma*scale+4096*最小f64正数。
这是保守算术检查，绝不是模型误差容差或对原编码gate的修改。
cosine 按原点积/范数直接报告（浮点可能略越±1），不clip、不加epsilon；
测试对非精确的cosine用固定浮点比较，不改变输出。
fixture 覆盖抵消、missing全分母、向量bool/nonfinite、Q/身份/动作/witness/预测篡改、
schema和外部三SHA篡改、真实临时gzNPY字节篡改及exclusive writer。
此处测试不导入旧producer，也不加载模型；生产结果仍由MAIN运行后给出。

## 7. MAIN执行命令

工作目录 `/home/gbk/code/LeWorldModel/le-wm-main`。
全部输入必须是MAIN已经完成完整编码消费的真实文件；以下shell变量是显式输入，
不是默认路径或预填未来SHA。`ENCODER_PREPARE` 为prepare JSON绝对路径，
`ENCODER_OUTPUT` 为含report.json的CUDA输出目录，`ENCODER_RECEIPT` 为完整消费JSON路径。
`ENCODER_PREPARE_SHA`、`ENCODER_REPORT_SHA`、`ENCODER_RECEIPT_SHA` 均需MAIN实际提供。

```bash
.venv/bin/python -m pytest -q tests/test_agent_p3_same_action_error_decomposition_20261029_v1.py

.venv/bin/python -m experiments.spectral_reachability.agent_p3_same_action_error_decomposition_20261029_v1 analyze \
  --prepared-path "${ENCODER_PREPARE:?actual absolute prepare path required}" \
  --prepare-sha256 "${ENCODER_PREPARE_SHA:?actual SHA required}" \
  --output-dir "${ENCODER_OUTPUT:?actual CUDA output directory required}" \
  --report-sha256 "${ENCODER_REPORT_SHA:?actual SHA required}" \
  --receipt-path "${ENCODER_RECEIPT:?actual complete receipt path required}" \
  --receipt-sha256 "${ENCODER_RECEIPT_SHA:?actual complete receipt SHA required}" \
  --saved-output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_same_action_error_decomposition_20261029_v1_MAIN01.json
```

只允许新独占结果文件，`xb`拒绝覆盖，文件内flush/fsync后目录fsync。
无CUDA/physics命令、无生产prepare子命令、无新fullstrict证明。

本设计按 documentation-and-adrs 记录符号、权威边界及接口决策；没有新实验、
没有root/超参选择、没有模型训练，不能据此放行 P1/P3/C3。
