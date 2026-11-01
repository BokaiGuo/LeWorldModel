# Early-history factorial之后：时间槽可识别性的最小增量协议

状态：独占NEW设计文档，待MAIN批准实现；没有生产runner/未来receipt。本轮只读旧协议/source/mask，不打开新factorial report的score/pairs/summaries或独立复算结果，不依据新outcome选择角、root、h、condition或预算。现4cell不重做。旧source/tests/doc/results均不动；不运行NN/physics/网络，不创建agent，不训练/拟合/修改Q或科学门。

## 1. 已有设计与证据等级

ROOT=`/home/gbk/code/LeWorldModel/le-wm-main`，以下相对路径均相对ROOT。

已封存`experiments/spectral_reachability/agent_p5_early_history_factorial_20261029_v1.py` SHA `39dbb6d61121ea9dbd6763be9b0104c45a8b3dbcf20ad3b090f21bed5a5645b7`，tests SHA `d19535839b9c0a99df3dae60d88733ed87767b7bdffa16afc564fa8283a9aa2e`，doc SHA `24a60779fd85df3a4dad58ab2405491e3c40d556de2117ff883f3e276475a490`。其原始合同是164枝/80对/四层，early5/10视觉×proprio四角，全cell最新15均固定short pred15，原动作packet[1,2,3]，全部656新预测封存后读共同H2 label20。RR重新执行并逐位核完整native IO。

MAIN已提供实际完成身份（本轮不读取分数）：prepare SHA `c8376a96a8320d4f147bf5e526954fbe8e31d37db565428ad82916dcc9d0f9c0`；report `results/agent_p5_early_history_factorial_20261029_v1_cuda_MAIN01/report.json` SHA `4fc86a7fdc69fbe95003e417ee38e3a4d430360a8eb5bd0c89f73ee845926d0f`；完整strict `results/agent_p5_early_history_factorial_20261029_v1_consume_MAIN01.json` SHA `d9461f7a21fa13454eca1f4371488431cf833a08ee758990c2dcb999b342600e`。MAIN报告656/1968、43.006s、errors/source_errors为空；这里不冒称本worker重新验证这些实际run字段。

独立标量核验器`agent_p5_factorial_independent_scalars_20261029_v1`已交付，source SHA `6beb1f3e52820e0cba1d8541f70d5b2df20bd41da1bbf2869998161311beeb79`；其MAIN实际analyze/verify结果本轮尚未提供，**pending**。下一prepare需绑定其实际输出/verify证据或由MAIN明确接受仅完整strict作起点，不伪造未来SHA。本设计不是已完成效果或科学通过。

## 2. 现有四角能识别什么，不能排除什么

现设计中A/R为actual/causal-rolled，字母分别控制visual和proprio，且每个因素同时替换5和10两个槽。只要完整strict及独立标量核验成立，实际可识别的是：**冻结模型、当前已知panel、固定latest15和原动作条件下，整块early视觉/proprio输入替换对final20输出及共同label误差的条件效应和交互**。是模型内部受控输入效应，不是自然轨迹物理因果效应。

| 可以控制/排除的替代解释 | 仍不能据此排除或证明 |
| --- | --- |
| 四角之间latest15、动作buffer和label/batch不是变化因素 | latest pred15本身可能错误；其生成依赖原rolled早期history，替换后不重算，混合history可能不可达或分布不一致 |
| 模型权重、source、normalizer和原root/condition固定 | 训练覆盖/泛化/不同root效果、特征MSE与原生任务成功的联系未认证 |
| RR全native IO bitwise复现排除“本次RR与旧short输入/输出发生数值漂移” | 不是所有其他cell或将来job都逐位确定的通用证明；requested MATH也不等于实际kernel证书 |
| 共同H2 label避免四角评分用不同标签batch | actual编码与predicted特征的分布/编码batch差仍内含于A/R因素，不能唯一分离为动力学误差 |
| 所有656已seal后label，排除本run用未来label数值驱动预测 | 不能排除研究者已知历史结果；仅新协议可前瞻固定，known roots不是盲测 |
| 输出差异非零可否定“这些具体输入替换在此panel完全无数值作用” | 非零不定位哪一个时槽、patch或模块；零也不证明架构屏蔽/全域不敏感，可能补偿、局部不敏感或范数下投影抵消 |
| 模态交互exact算术可分解此有限四角 | 不能把MSE交互称物理交互，不能把cosine符号变化线性归因，也不是微分Jacobian或无限小局部线性定理 |

源码依据：`external/jepa-wms/src/models/utils/modules.py::build_action_block_causal_attention_mask`以`t2<=t1`填time block；`external/jepa-wms/app/plan_common/models/AdaLN_vit.py::forward`切片attn_mask并送入各block。当前T3允许最后槽读取early5/10，不能把可能的零效应直接解释为mask阻断。新执行仍需实际loader/source哈希闭包，不能抄此文字当当前加载证明。

## 3. 唯一建议主对照：early时间槽5×10联合模态交叉

不按新outcome决定先试5或10。两时间槽都事前固定；在每个时槽，visual与proprio**成对**取A或R，仍固定short R15和原actions。将这次新增cell命名为`T_AR`、`T_RA`，T字母按时间5、10排序，绝不能与旧视觉×proprio的AR/RA混淆。

| temporal cell | slot5 visual+proprio | slot10 visual+proprio | slot15 visual+proprio | 来源 |
| --- | --- | --- | --- | --- |
| T_AA | actual5 | actual10 | 固定short pred15 | 复用early-factorial旧AA，非新call |
| T_AR | actual5 | short pred10 | 同上 | 新164 calls |
| T_RA | short pred5 | actual10 | 同上 | 新164 calls |
| T_RR | short pred5 | short pred10 | 同上 | 复用early-factorial旧RR，非新call |

新增两个角与现有模态AR/RA通常不同，不能借用旧输出。最小问题是“5/10时槽联合模态替换的条件效应能否分开，以及二者是否有有限交互”，而不是再次运行early模态四角或继续重跑已证实相同的whole/latest。相同枝保留为零干预；若整个新角与旧角全164输入相同，则prepare判为不具新增可识别性，保留blocked证据，不将其改名新对照。

## 4. 可落地ABI、来源与封存门

建议独占新stem待MAIN命名批准，不创建生产文件。本次无新增代码。实现可复用已封存loader、完整native Trace和codec，自有两call/branch Meter；不改旧全四角guard或monkeypatch globals。

`prepare(authority_report, report_sha, full_receipt, receipt_sha, independent_scalar_evidence?, budget_mode, output)`只读取完整持久receipt及source/prepare/artifact/schema/计数身份，hash未来label但不decode。不嵌套旧fullstrict；未重建的上游输入/chronology/model语义明确继承完整strict。新`verify`重建本prepare；`run`仅MAIN HOSTCUDA；新`consume`全量检查本次IO/journal/输入/输出/label/计账和独立算术，不能用旧strict冒充新成功。

branch i旧native inputs从已封存early report按`/calls/{4*i+j}/inputs/$tuple/0/$tuple/{k}/$array`抽取：j=0为旧AA，j=3为旧RR；k=0/1/2为visual/action/proprio。descriptor绑定directory/file/SHA/bytes/shape/dtype及JSON pointer，验证whole NPY后slice；不是按文件名推身份。旧输出同call的`/outputs/$tuple/{0,2}/$array`，行prediction树与call交叉核；label从旧`/rows/{i}/label`取共同H2 batch2 slot1，并继续绑定原H2 source，不换batch。

输入ABI恒为visual(1,3,1,16,16,384)、action(1,3,10)、proprio(1,3,256,16)，float32 contiguous：

- T_AR：visual/proprio槽0取旧AA，槽1取旧RR，槽2取旧RR；T_RA反之（槽0旧RR，槽1旧AA，槽2旧RR）。两新角动作都取旧RR全buffer，需与旧AA动作byte相等。
- 所有branch root/task/selection/condition/raw20/normalizer及时间5/10/15必须exact；原20个raw时刻×2维动作自身证据不改，packet对应raw6..10/11..15/16..20，不假称1raw=1packet。
- 新all328 full3视觉/proprio输出先durable seal；之后才读取/复制共同label并统一计分。旧endpoints也先绑定输出SHA与完整strict，禁止先看新label决定输入。
- 失败保留全164/80及两个新槽，禁止换root/h/overlay、padding、补跑或选择“有效pair”。prepare的差异census不删individual相同枝；input bytes相同的输出比较只能视为复现，不视为新对照。

## 5. 最少预算与跨job边界（尚未授权运行）

建议最小新增为**328predict API=328samples=984frames，0encoder/proprio_encoder/AD/physics/training**；原四角656成本完整单列，不计成新调用。每call保存全3inputs/full3 visual+proprio output=2,457,720B，328共806,132,160B；共同batch2副本128,974,848B，共935,107,008B tensor。无需重编码actual5/10/label；不重做0→15滚动。

这个最小模式复用旧AA/RR endpoints，跨job系统数值环境仍有限制；即使旧RR先前bitwise相同，也不能证明下一job同样。若MAIN要求本轮baseline bridge，必须**事前**另选`328+164=492calls/1476frames`，额外固定只重放RR（不是按效果补跑，也不是重做现四角）。RR全3inputs和2outputs必须逐位等于旧RR，差异即partial/blocked、不加容差。该模式native1,209,198,240B+label128,974,848B=1,338,173,088B；它认证RR在本job复现，仍不是任意其他输入跨job数值的全域证明。

预算模式必须MAIN在任何新结果前选择并封存；本doc不替MAIN默选、也不因当前结果决定。两模式均可事前cap2GiB tensors/3GiB total output（包括prepare、NPY headers、journal、label副本、report）、900s/floor200MiB/reserve64MiB；若新实现额外bank/复制使上界失效，先报告，不降精度/缩分母。实际RSS/GPU peak、attempt/return/validated、模型加载bytes/source前后、请求MATH和observed guard分轴记账，不宣称等FLOPs。

## 6. 固定统计，不用效果决定设计

两个新角加两个旧endpoint全164/80完整重评分，全部cell用同一H2 label20；全256×384 float64 C-order MSE，exact-zero→null cosine，无epsilon/可调阈值。保全体与原四层、全部signed pairs/两h/十坐标、observed-zero与nonzero计数（不把56/24当硬编码准入），每cell response norm/dot/cosine/error与/2h差分、全部符号转移。

记时间顺序值a=T_AA、b=T_AR、c=T_RA、d=T_RR，固定：slot5=`((c-a)+(d-b))/2`，slot10=`((b-a)+(d-c))/2`，interaction=`d-c-b+a`，joint=`d-a`。每branch标量及预测向量效应均保留；向量L2不等于MSE效应，cosine不作线性分摊。所有正负零均报告，不选“最佳时槽”作为方法、不做事后显著性。

可识别的是固定latest条件下**每个早期时槽的联合模态**影响；仍不能分离slot5视觉vsproprio、slot10视觉vsproprio或任意更高阶跨模态×时间交互，也不能证明换成actual15能修复。要回答这些另需新固定设计，不能从本两角外推出16角全factorial，更不追加zero-early OOD消融。

## 7. 封存、独立根和科学外推前置

本known-root下一轮仅calibration/内部机制定位，**不要求P1通过才能工程准备**，但不得称独立确认。生产前须封存协议、预算模式、两新角顺序、全164/80 identity schedule、normalizer/model/config loaded-byte接口、完整源tests/doc SHA、输出目录/caps、所有scalar与零规则；全部预测先seal后label。旧失败/负结果/当前所有角保留。源码/receipt任何改变只能新版本；不根据MAIN新summary选择协议分支。

要将本设计推进为独立根确认，最少还缺：MAIN批准的独立root数量和采样分布/seed事前承诺、producer/factory/env/constructor/config/模型精确来源、完整reset/load尝试ledger（包括失败/未选及其范围上界）、所有载入状态字节及历史排除closure。不能把“未找到历史记录”认证为fresh；不同seed/root名不自动是独立状态或独立采样。旧freshness缺口不可由此文档、当前known-root成功或新CUDA重放回证。

新根若无既存actual0/5/10/15/20和原动作/proprio的完整合法资产，则本次0physics/0encoder预算**不适用**：需另授权真实reset/warmup/action逐步观测、编码原预处理/批次、实际model roll与完整计量，且重新建立two-h×10coord±及nominal、off/on分母。此处不凭未知采集实现猜reset/physics或编码API总数；明确缺原采集runner具体调度、root数及batch/dedup合同。根数/排除规则必须在看新根outcome前固定，无按成功挑根。

对已具有同协议完整saved bank的每个root×condition组，最小temporal两新角增量为41×2=82predict/246frames；若批准RR bridge则123predict/369frames。当前四组共328或492。这个增量公式不包括新根采集/编码/causal-roll/端点建立成本，必须继承单列，不伪称新根全实验只需82。独立根确认仍不能跨架构比较MSE绝对值，不能自动开放P1/P5或宣称task success因果改善。

## 8. 本次实际交付

只新增本doc。只读了旧factorial合同、mask/forward源码并核旧交付文件SHA；未打开新factorial outcome、未执行独立标量生产分析、NN/physics/consume。文档预算算术和固定两角覆盖可用stdlib轻量检查；无新增runner或需模型的测试。下一可落地动作是MAIN选择328最小模式或492带RR bridge模式后，授权新stem实现fixture+prepare/run/consume；不是直接启动生产，也不等待或宣称科学准入。
