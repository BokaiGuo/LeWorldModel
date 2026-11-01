# Moving live teacher：纯saved诊断解释与交接合同

## 状态与边界

本文仅定义怎样合并已封存控制结果与随后teacher结果，不运行模型/物理，不修改driver、analysis或teacher v1/v2。
本文初稿在teacher终结前制定解释映射；第6节在MAIN交付显式actual report后补入纯saved数值，不冒称结果前预注册。
原控制实验已完成不等于teacher成功；teacher完成也不等于P1通过。两个known roots是校准任务，不是12个独立样本或泛化验证。
没有新阈值、拟合、选择、补跑、显著性检验或CPU neural fallback。

## 1. 输入authority与读取次序

使用显式路径及外部SHA，不搜索最新目录、不按效果挑版本：

- 原控制report：`results/p1_moving_live_feedback_physical_20261027_v1_main01/report.json`，SHA `990843e8bba15955db5163b9b5d77c1f8eecaa1870d3745b49d46cc22ca895e3`。
- 原控制prepare SHA：`bc3fbc91d7a19ae256b24db22ad05b2526fcbe454e3b55d904136eba2f8fdff5`，路径沿其execution_binding并与显式authority闭合。
- 原描述性analysis v2：`results/p1_moving_live_feedback_analysis_20261027_v2_main01.json`，SHA `5df7dd90b3da010d1bafbb66198334272ec6470b7b0f7bc62991e4b0dae0b5d8`。
- teacher v2 prepare：MAIN交付SHA `b83bccf3b5edba63e59c0c3f4efc4e796fe99bfdd43a9a6096c8eb57f1b9fae6`；实际路径由MAIN显式指定并与teacher binding/report相符。
- teacher v2 actual report：`results/p1_moving_live_feedback_teacher_fidelity_cuda_20261027_v2_cuda_main01/report.json`，外部SHA `7323347a9e74c8f1751550642503426d07ddce55549ddaef4331f3c362567128`。第6节只读其保存结果；不能用v1失败或prepare成功替代实际report。

teacher v2的prepare是磁盘≤256KiB的单gzip/base64 envelope；应通过冻结v2 `verify(prepared, expected)`取semantic对象，不能把envelope直接当旧prepare。
完整semantic bindings保留，解码上限1MiB；外部prepare SHA与内部semantic SHA不是同一个摘要。
先运行冻结teacher `consume(output, report_sha)`作原有saved完整性检查，再合并描述性字段；consume不调用模型。
若需要独立重算，可只读原NPY用float64 NumPy算术；不得为缺失数组重编码/重新predict。
完整检查包括原task/model/source SHA、全预测seal早于任何episode_scored、原始module链和artifact集合；仅有report中的标量不替代这些证据。

## 2. 固定join：不能只按root或行号

完整cohort为2 source_slot ×2 prefix_arm ×3 mode，共12 episode；4个root/arm层，每层O/I/R。
arm字面值为`line_search_GN`、`full10_conditional_D2P`，mode为`O/I/R`。
固定顺序由冻结analysis `KEYS`确定，但join必须同时核身份，不依位置猜对应：

```text
teacher.rows[i].identity
  index, episode_id, source_slot, prefix_arm, mode,
  parent_task_id, parent_selection_sha256
            │
            ├─ teacher semantic inputs/evaluators：完整identity精确相同
            ├─ analysis.rows：(episode_id, source_slot, prefix_arm, mode)唯一相同
            └─ 原driver prepare.layers：(source_slot, arm)唯一，
               task_id/selection_sha256与teacher parent_*相同
```

episode_id严格为e00_O/I/R、e01_O/I/R、e02_O/I/R、e03_O/I/R，不能用另一个episode恰相等的NPY值冒充来源。
teacher预测是每条自己的actual prefix和actual suffix；标签是该条自己的actual terminal。
图/latent/commands重复不去重、不新增独立样本、不退款。
source/task/goal/model/normalizer任一身份不闭合、重复或未知额外row必须failclosed；不能靠数值近似相等放行。
完整结果应保12 episode、24原观测endpoints和12 teacher评分；不足时保固定分母、缺项null及工程原因，不缩为成功子集。

## 3. 精确字段与算术

下面是读取映射，不是新运行输出schema；若另写汇总应保原外部SHA及这些字段来源。

| 含义 | 冻结saved字段 |
| --- | --- |
| teacher自身H1误差 | `teacher.rows[i].metrics.factual_MSE` |
| teacher预测/实际终态goal Q | `metrics.predicted_Q` / `metrics.actual_Q` |
| teacher Q偏差，正数为预测更差 | `metrics.predicted_Q_minus_actual_Q` |
| 实际prefix到terminal的latent变化量 | `metrics.actual_prefix_terminal_mse` |
| 原prefix/terminal观测Q | `analysis.rows[j].endpoints['5'/'10'].observed_Q` |
| 原生及wrapped/object指标 | `endpoints['5'/'10'].native_recomputed`；原始终态还保`original_final_metrics` |
| 原控制suffix模型实耗/预声明预算 | `suffix_budget_actual` / `suffix_budget_expected` / `suffix_budget_matches_full_protocol` |
| 原成对描述 | `analysis.contrasts`按source_slot/prefix_arm/contrast定位，核`prefix_pairing_exact` |

设p为teacher预测terminal，y为自身actual terminal，g为原task goal，x为自身actual prefix，均为原native192。
在原float32数组上转float64计算，mean分母192，不把sum、L2 norm、投影拟合或新权重替代原Q：

```text
M = mean((p-y)^2)                     # factual_MSE
Qhat = mean((p-g)^2)                  # predicted_Q
Qactual = mean((y-g)^2)               # actual_Q
b = Qhat-Qactual                     # predicted_Q_minus_actual_Q
V = mean((x-y)^2)                     # actual_prefix_terminal_mse，非第二次模型预测
```

teacher actual_Q必须与同episode analysis terminal observed_Q闭合；prefix使用同一NPY receipt且原goal不变。
复算先核数组file/array SHA、dtype/shape和规范算术；若出现不一致，保差异为工程/绑定问题，不先解释成模型误差。
V只是实际移动的latent尺度参照，不是预声明误差阈值；不得事后设M/V界给出“合格”，V=0也不得构造无穷或丢行。

原native_distance严格为state7-goal7全7维norm，含末两维velocity；success仍原norm(d[:4])<20与原angle项<pi/9。
object_position_error仅d[2:4]，object_pose为state[2:5]，不能与native_distance混称同一量。
native_angle_term仍原min(abs(d[4]),2pi-abs(d[4]))，超2pi可能负值；wrapped角诊断单列，不悄悄修正原生success。
native终止/成功与saved recomputation的来源要保留，不能把latent Q或object-only改善替代原生成功。

## 4. 分开解释三条证据链

### A. 预测：冻结模型在已执行动作上的factual transition

M刻画自身actual context、实际suffix下的transition预测误差，b只刻画沿本goal的标量偏差。
令e=p-y、u=y-g，则b=M+2mean(e*u)，因此小b不保证小M，M也不决定b符号。
几何一致性检查可用`abs(b) <= M + 2*sqrt(M*Qactual)`（允许说明浮点残差），这不是接受阈值。
没有授权数值阈值时只报告原值、排序和描述，不将“低/高”机械标注为合格/失效。
误差大可说明本条factual预测与观测有差异，不能独立定位为训练域、normalizer、视觉别名或动力学哪一环的因果故障。

### B. 度量：同一实际终态的latent Q与物理指标是否一致

在同root/arm且prefix精确配对的O/I/R中，对照actual_Q、native_distance、object位置/角及原生success。
若某对比Q改善而物理指标未改善，可描述该对比的目标排序不一致；不能从单次排序判定表示普遍失效。
若teacher M较小但native仍失败，也不能宣称动力学足够好或唯一问题就是Q：候选/预算、约束、目标可达性和度量均未被排除。
若teacher M较大又Q/native不一致，两种描述可以同时成立，不能二选一地归因。

### C. 控制选择：已部署O/I/R的实际成对结果

原`I-O`、`R-I`、`R-O`均为lhs-rhs；Q/距离等负数表示该标量减少，布尔success差不能当概率提升。
只有`prefix_pairing_exact=True`才解释同层后缀对比；相对prefix的实际变化为各自Q10-Q5。
R与I的部署context和选中suffix都可能不同，差异只能描述既定协议包的结果，不能断言单独context/J/谱机制的普遍因果效应。
teacher为I也输入actual prefix，这是事后factual诊断；不能倒写I部署时见过actual prefix，亦不能当I当时的imagined预测。
teacher Qhat不是原planner选中时预测分数；未知原预测继续unknown，不用teacher补填。
不同O/I/R使用不同suffix，所以它们M的差异不是同动作反事实预测差。
只有已执行动作有真实label，未执行候选没有；不能据此证明selector错选、另一个候选本可成功或全局最优。

## 5. 汇总与错误交接

建议按4层逐层列O/I/R的M、b、Q5、Q10、Q10-Q5、全7native距离、object位置/角、native_success和预算attempt/completion。
再列原三种成对contrast；teacher M/b差仅可另列为“不同自身轨迹的描述差”，不冒充同动作对照。
可以做全12固定清单的描述统计，但说明重复任务/相关分支；不报告独立n=12、显著性、置信泛化或P1/P4通过。
旧控制预算、原24图编码成本与新teacher12 predict/AE/predictor/pred_proj单列，API/子模块不相加作24次预测。
新teacher encoder/AD/J/SVD/physics应为0，但只能在实际ledger闭合后陈述达到；不预填实际资源。

teacher缺seal/工程失败时：原控制结果仍可单独描述，teacher对应项null/unknown；不能把工程失败当模型预测错误或理论负结果。
teacher评分阶段失败时：保已封预测、错误与缺项，不补label或重跑模型。
source/目标身份冲突时：禁止生成合并解释，回报精确绑定冲突；科学12分母不因失败降低。

MAIN最终交接只需实际teacher report路径/外部SHA、冻结consume状态、各row完整性及任何工程差异。
本doc不预填teacher结果，不更改下一轮控制，不授权追加模型、阈值或样本。

## 6. Actual saved补记：同一模型预测，不等于正确实际排序

MAIN交付teacher实际终结并确认ownconsume通过。本doc读取上述report并独立核外部SHA；实际report状态为
`teacher_factual_recorded_not_admitted`、errors=[]，四项AE/API/predictor/pred_proj各attempted=completed=12。
这些CUDA工作由MAIN完成，本doc作者只做NumPy保存数组核对，未加载模型或执行物理。
原控制report和analysis v2外部SHA也按第1节逐字核对。

### 6.1 全12自身轨迹的原数值

以下保留12分母；展示值四舍五入到9位小数，精确值以原report/NPY为准。

| episode | factual_MSE | teacher Qhat | actual Q | Qhat−actual Q |
| --- | ---: | ---: | ---: | ---: |
| e00_O | 0.065113129 | 0.604785097 | 0.482174596 | +0.122610501 |
| e00_I | 0.081734405 | 0.664413096 | 0.633924778 | +0.030488319 |
| e00_R | 0.150446031 | 0.490135683 | 0.565551149 | −0.075415466 |
| e01_O | 0.137096121 | 0.199897467 | 0.182658563 | +0.017238904 |
| e01_I | 0.137096121 | 0.199897467 | 0.182658563 | +0.017238904 |
| e01_R | 0.050770463 | 0.060671085 | 0.039441458 | +0.021229627 |
| e02_O | 0.027669037 | 1.284849149 | 1.374904060 | −0.090054911 |
| e02_I | 0.019864161 | 1.357722332 | 1.378029943 | −0.020307611 |
| e02_R | 0.019558960 | 1.330095476 | 1.332704614 | −0.002609138 |
| e03_O | 0.055428581 | 0.220834617 | 0.198951944 | +0.021882673 |
| e03_I | 0.055428581 | 0.220834617 | 0.198951944 | +0.021882673 |
| e03_R | 0.090498240 | 0.260056094 | 0.129025324 | +0.131030770 |

纯saved重新读取12个prediction、12个label和各自原encode5/encode10 NPY，校验相应file SHA；
teacher label与本episode原encode10的float32字节全相等。
每层goal取原R response artifact绑定的`branch_00.goal.npy`，用原float64 mean192算术复算五指标。
全12×5=60个标量与teacher report精确相等，不是仅显示精度内相等。
这里的核对不替代MAIN已完成的完整consume/hashchain工作，也不是新增模型实验。

### 6.2 R−O实际排序：三层反转，仅一层同向

| 层/原任务 | Δteacher Qhat (R−O) | Δactual Q (R−O) | 排序关系 |
| --- | ---: | ---: | --- |
| e00 / slot0 GN | −0.114649414 | +0.083376553 | 反转：模型偏向R，实际Q偏向O |
| e01 / slot0 D2P | −0.139226382 | −0.143217105 | 同向：两者Q均偏向R |
| e02 / slot1 GN | +0.045246327 | −0.042199447 | 反转：模型偏向O，实际Q偏向R |
| e03 / slot1 D2P | +0.039221477 | −0.069926620 | 反转：模型偏向O，实际Q偏向R |

特别注意e02也反转，不能写“e01/e02同向”。恒等式为
`Δactual Q = Δteacher Qhat − (b_R−b_O)`；四层bias差分别为
−0.198025967、+0.003990723、+0.087445773、+0.109148097。
例如e02的R factual_MSE比O小、R自身Q偏差也接近零，但O的乐观偏差约−0.09005，
仍使模型的两动作相对排序与实际相反。这说明单条误差较小不能直接推出动作排序可靠；不引入误差接受阈值。
四个层是两个根上相关控制分支，不将3/4写成泛化排序失败率或理论否证。

### 6.3 原R dispatch预测确实可与teacher逐位对上

原response不含顶层`dispatch_prediction`或`dispatch_model_Q`，不能据这些不存在的字段宣布缺失。
真实预测保存在`eXX_R_suffix_response.json`的`row.evaluations`，stage为
`old_suffix`、`new_suffix`、`new_suffix_replay`；每项有context/cost/latent_array/token_array。
dispatch字面值为`new_suffix_unconditionally`，实际控制并非在old/new预测Q中择优：
因此e02/e03新动作预测Q更差却仍执行，是冻结协议行为，不能直接称selector代码错误。

本次四层都按原live dispatch_seal的response_sha256（packet compact canonical摘要）闭合response，
并逐文件校验response.artifact_sha256后读取数组，完成下列纯saved精确比较：

- R namespace `branch_00.observed_latents.npy`与teacher该R input.latents的float32字节相等；observed_tokens与input.history相等。
- `branch_00.new_suffix.npy`与teacher该R input.raw5相等，原封存response的dispatch_raw5/实际执行绑定仍沿原source合同。
- `branch_00.new_suffix.context_latents.npy`及`new_suffix_replay.context_latents.npy`的末192维，均与teacher对应R prediction逐位相等（4/4）。
- 四层`old_suffix.context_latents.npy`末192维与teacher对应O prediction也逐位相等；原cost分别精确等于teacher O/R predicted_Q。

原四个R response文件SHA（不同于packet digest；两者不得混用）：

| 文件（相对原live output） | 文件SHA256 |
| --- | --- |
| e00_R_suffix_response.json | `1e89259f9c9208141d8a4d20cc5afae6f3bd9309835ea0fff0b1198e782e7718` |
| e01_R_suffix_response.json | `b97ff0700e5eb5b473c0589d665b99f05df91585e713964a076f0738d49ab148` |
| e02_R_suffix_response.json | `474d4bf29950ea04e6aefb21bdf2346c0e79ac06eb68b191aeeb850a902b468d` |
| e03_R_suffix_response.json | `1ffb7a2adf90146f25ed72ba75baf86d429ff1e67a485ba0980dd3d0b437b855` |

故本次R/O的teacher排序不是新teacher context或推理数值路径改变后另造的分数；它重现了原R compiler
在该observed context对old/new动作的保存预测。预测与实际排序差异在这些已执行动作上可直接陈述。
仍不能据此声称I当时的imagined预测也重现：teacher I使用actual context，此处未把I dispatch预测等同teacher。

### 6.4 与物理指标分开看

原analysis v2的terminal指标如下，仅列O/R便于解释上述contrast；I仍完整保留在原12行中。

| 层 | native_distance O→R（全7维） | object_position_error O→R | wrapped角误差 O→R |
| --- | ---: | ---: | ---: |
| e00 | 105.218088→201.529756 | 39.788663→36.424485 | 0.279523→0.083553 |
| e01 | 166.541341→136.162856 | 16.925102→9.213750 | 0.001532→0.060849 |
| e02 | 367.439715→381.369603 | 134.735301→135.723830 | 0.445289→0.453206 |
| e03 | 508.413146→481.266475 | 36.788671→31.179388 | 0.074637→0.292395 |

e02的实际latent Q改善而上述三个物理误差都变差，是本条目标排序不一致的具体例子；
e00则object位置/角改善但全7距离变差，不能只取一种指标宣告全面改善。
e01/e03位置与全7距离改善而角误差变差，也不能由latent Q减少推出姿态全部改善。
原12条native_success均false，teacher诊断不改这一结论。

可以支持的窄结论是：本批自身轨迹上存在已保存且可复现的预测误差、三层R/O latent-Q相对排序反转，
并且latent-Q与物理指标的相对变化未一致。它们是预测/度量/固定派发协议三个层次的不同描述，
不互相替代，不证明唯一瓶颈、机制成功或失败、可达性、替代候选必胜、独立泛化或P1科学准入。
