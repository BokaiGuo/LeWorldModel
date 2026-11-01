# P5 rolling history资产前置：固定两份manifest的saved-only清单

只交付本stem source/tests/innerdoc；不改任何H1/H2/whole/latest/MAIN封存，不做零early特征消融，不执行模型、物理、生产inspect/verify或full strict consumer，不训练、不改normalizer/Q/gate、不按outcome筛根/分支。本工具没有run模型入口，不是新实验prepare。原独立goal API是无关blocked目标，不覆盖/完成它；个人交付仅为本实现与fixtures。

## 固定输入和检查范围

H1 manifest绝对路径：
`/home/gbk/code/LeWorldModel/le-wm-main/results/agent_p5_raw164_jepa_cuda_20261029_v1_prepare_MAIN01.json`

SHA：`222d917f262e2a7c8e347fd351e3b0927ec4ffe200035e7b3b971a7abf0000fe`

H2 manifest绝对路径：
`/home/gbk/code/LeWorldModel/le-wm-main/results/agent_p5_raw164_jepa_h2_cuda_20261029_v2_prepare_MAIN01.json`

SHA：`09070780c3ce782a71ae01492e0c4b801de976e9a36cdf9ee1bf1a53be98e924`

只读取这两份明确SHA的JSON及其绑定的H1/H2 runtime、原AdaLN/attention/loader源文件hash；不扫描其他目录，不读取候选分数、标签PNG/latent，不调用owner.build/verify/consume，不反向认证全项目训练/loaded bytes。JSON拒绝重复key、非有限值、SHA漂移；schema/全部164顺序/两root×off-on/原80pair/任务selection/动作/原normalizer/图像时标与condition一致性都检查，缺失或错配直接报错。

检查原raw commands与原source float32归一化：`(100*u64).astype(float32)/float32(100)`后使用原stats，逐位复算packets/proprio。H1 raw6..20必须与H2 raw1..20的后15相同。共同label只核metadata身份，不decode。warmup10与plan0不能凭文件名视为相同：manifest中对应pixel SHA与原生agent proprio须exact对齐。PNG本身/像素hash不在本工具复验范围，报告明确`PNG_or_latent_decoded=false`和`full_strict_consumed=false`，不能用它替代旧strict。

## 当前资产能说明什么

全164个request的H1实际context为5/10/15、终态20；H2实际context为0/5/10，原本已声明的滚动预测为15、20。两份manifest明确提供actual0/5/10/15图像身份、原native agent[x,y,vx,vy]与原raw1..20，未提供独立封存的causal pred5→pred10滚动历史。

必须区分“缺独立滚动history”与“没有数值数组”：H2第一次调用保存完整full3，其前两个输出槽具有successor5/10的时序语义；但其中第二槽以**actual5**输入，而非由前一槽生成的pred5。因此不能把该full3前两输出改名为已经认证的连续roll5/10，更不能将actual5/10换名当predicted。旧H2原滚动只把最后槽pred15追加到actual5/10后，然后预测20；这也解释为何whole/latest输入实际相同。

本工具仅查manifest声明和绑定source，不打开H2实际输出或full strict，因而`available_rolled_prediction_semantics=[15,20]`明确只是原协议声明，不是新完成资产认证。缺口结论严格限于这两份manifest，**不是全项目不存在其他早期history资产**。

## 合法时序ABI：区分固定full3与明确的新short-start协议

若要求起点每次都保持旧固定三槽，首次预测5必须输入actual[-10,-5,0]，对应raw[-9..-5]、[-4..0]、[1..5]。这两份manifest缺actual-10/-5的RGB及native agent proprio，也缺raw-9..0，更缺连续前驱轨迹exact接到旧frame0的认证。不能倒积分物理、复制frame0、补零或改时间标签。`full3_only_alternative`完整列出四次滚动及该缺口。

但不能因此断言所有合法起滚都必须有负帧。原绑定`AdaLN_vit.py` forward从实际输入读`B,T,N,D=x.shape`（310–325），按输入token数截mask（340–344），再按实际T恢复输出。原`modules.py` RoPEAttention按实际T生成位置（189–200），原mask允许过去、屏蔽未来。原v8 loader为feature proprio/0condition tokens。结构上可使用真实T=1、2，不需padding；**尚未在HOSTCUDA验证，且不同于旧runner强制三槽的执行合同**，必须MAIN另行批准新协议并检查真实shape/计账后才可运行。这里只提供纯NumPy ABI/拼接函数，不执行它。

新short-start ABI固定如下，不根据结果选起点：

| 预测 | 完整context来源 | 完整动作packet（raw闭区间） |
|---|---|---|
| pred5 | actual0，T=1 | 1..5 |
| pred10 | actual0、sealed pred5，T=2 | 1..5、6..10 |
| pred15 | actual0、sealed pred5、sealed pred10，T=3 | 1..5、6..10、11..15 |
| pred20 | sealed pred5、sealed pred10、sealed pred15，T=3 | 6..10、11..15、16..20 |

视觉输入`(1,T,1,16,16,384)`；本体输入`(1,T,256,16)`；动作`(1,T,10)`；原生返回visual`(1,T,256,384)`、None、proprio`(1,T,256,16)`。只追加每次full返回的末槽visual和proprio，保留最多三槽；不能从别的teacher-conditioned输出槽偷取pred10。全4call真实输入/输出必须seal并明确来源，不能先看共同actual20 label再决定轨迹。

模型内部时间位置采用原native当前序列局部坐标，不伪造负帧position embedding。short-start不是原H1/H2同horizon比较，也不称原模型训练支持已证明。运行需要原actual0**原生编码与本体feature实际IO**的外部路径/SHA、模型loaded state与normalizer身份闭包；这两份prepare本身不足替代该IO绑定。本工具不重建原权重、不读取该IO，明确保持blocked。未来若验证可复用这些已有实际tensor，可0新encoder/physics；不能预先声称复用已通过。

原warmup0/5/10是pre-constructed-goal-bind观测，保留原默认goal_pose渲染；plan阶段原goal绑定在global10。新ABI必须保留原source的时间/条件/goaloverlay记录，不把旧warmup重画成plan观测，也不凭数值相同抹去该语义接缝。predictor本身不接goal label作为输入。这里不认证goal语义影响不存在。

## fail-closed预检API及成本边界

- `missing_history(...)`同时报告fixed-full3缺口、short-start已有actual0/raw1..20和仍未批准/未验证的接线。它永不直接返回新contrast已认证。
- `bootstrap_packets(...)`仅接受明确short-start actual[0]与完整raw1..20，生成四个无padding动作窗口。不得给0/0/0复制帧；不生成视觉feature或预测。
- `advance_history(...)`只作原位来源明确的NumPy拷贝/concat，要求两种完整native预测返回，绝不产生模型预测。
- `contrast_preflight(...)`用于将来的全量输入预检/fixture：模型、root/task/condition/selection、时间、normalizer、共同label、goaloverlay身份必须相同，完整动作必须逐位相同。视觉/本体前两槽至少有真实数值变化才允许叫**新的early-history输入contrast**；仅末槽变化、全部相同或仅signed-zero字节差异一律拒绝。函数不认证输入物理可达、预测有效或新实验已完成。将来调用者必须全164通过，不能删失败分支来形成子集；本工具未生产执行此预检。

short-start未来完整rollout需4×164=656 predict、(1+2+3+3)×164=1476 token-frame samples；不是授权或实际receipt，额外AA/对照、模块IO/wall/cap需另行事前合同。本交付0NN/0physics/0training，metadata cap32MiB、exclusive写入、磁盘floor200MiB。

full3-only若需新采集前驱，单branch至少额外10 raw controls，沿原每control10substeps就是至少100substeps；constructor/reset/observations/close成本未定，且根本**不保证能抵达旧exact frame0**。不能把此下界说成已有可执行采集预算。若只能生成不同root，必须独立新panel，不能回证当前164。

混合/滚动context仅model-internal，不保证物理可达；额外native位置/速度权限不是RGB-only，不是控制收益、跨模型公平排名或P1/P5准入。

## MAIN CLI（worker未运行实际inspect/verify）

工作目录`/home/gbk/code/LeWorldModel/le-wm-main`。

```bash
.venv/bin/python -m pytest -q tests/test_agent_p5_roll_history_preconditions_20261029_v1.py

.venv/bin/python -m experiments.spectral_reachability.agent_p5_roll_history_preconditions_20261029_v1 inspect \
 --h1-manifest /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p5_raw164_jepa_cuda_20261029_v1_prepare_MAIN01.json \
 --h1-sha 222d917f262e2a7c8e347fd351e3b0927ec4ffe200035e7b3b971a7abf0000fe \
 --h2-manifest /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p5_raw164_jepa_h2_cuda_20261029_v2_prepare_MAIN01.json \
 --h2-sha 09070780c3ce782a71ae01492e0c4b801de976e9a36cdf9ee1bf1a53be98e924 \
 --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p5_roll_history_preconditions_20261029_v1_MAIN01.json

.venv/bin/python -m experiments.spectral_reachability.agent_p5_roll_history_preconditions_20261029_v1 verify \
 --inventory /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p5_roll_history_preconditions_20261029_v1_MAIN01.json \
 --inventory-sha ACTUAL_INVENTORY_SHA
```

inspect成功表示阻塞清单已生成，status固定为`blocked_missing_causal_pred5_pred10_and_short_start_validation`，绝不表示可运行/准入。verify仅重建同metadata清单，不隐式调用full strict或模型。新文件不覆盖，错误schema/SHA/缺项则直接拒绝；不伪造未来prepare/report。待MAIN实际执行的产物SHA未预填。
