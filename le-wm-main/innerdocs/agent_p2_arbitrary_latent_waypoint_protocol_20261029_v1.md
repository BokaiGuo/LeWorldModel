# Exp1/P2 arbitrary latent：最小事前协议草案，待MAIN批准

独占stem `agent_p2_arbitrary_latent_waypoint_protocol_20261029_v1`。
本次只有协议，不新增runner、状态wrapper或实际优化结果；不覆盖旧missing。
无NN、physics、生产fullstrict、子代、训练或结果选参。

## 1. 已读原件与实际缺口

原附件：`/home/gbk/.codex/attachments/861f6764-1b25-4227-bf15-f1a237ba7d8f/pasted-text-1.txt`，
SHA256 `6cd1ebb67192485f19fa8debb1d50ee6586a5099efa5816099bd19c3511a9aae`。
527–574行Exp1要求arbitrary直接在latent中优化目标，区别于learned prior和动作见证产生的中间状态；
最终问题是共同低层controller能否实现、执行偏差和闭环成功，不只是latent距离。
该段没有规定arbitrary的loss、初始化、步数、时间尺度或Hi head。

实际代码已存在：

- `experiments/spectral_reachability/agent_p2_direct_latent_optimizer_20261029_v1.py`，
  SHA256 `0f38e93e500ecfc75ec35581552d04a03ba2868dc81192659acdfc36c46a9b76`：
  已有真正直接latent参数优化、解析梯度、固定64步及RAM tracking seam。不是尚待重写的空接口。
- `experiments/spectral_reachability/agent_p2_target_only_delivery_20261029_v1.py`，
  SHA256 `e69b2d1e5fcd5d853004a0b3b2126c8506ffd7068a39ac867b60134d3358f4c7`：
  真实离线join，controller payload只允许z/a/target；无dispatch。
- 实际`results/agent_p2_target_only_delivery_20261029_v1_prepare_MAIN01.json`，
  SHA256 `82b36fa3030f8ebb2da37bdc91ba3e8d582f129d6bc9e5c9f8e774b363ef3f62`：
  roots0/1、off、六槽；prior_FIRST与program_first_postcondition四槽assembled，arbitrary两槽missing。
  reason为没有共同ABI上的认证直接latent输出，不表示数值优化代码不存在。

以上仅只读源码/保存JSON核对，未调用build/verify/consume或加载原数组/模型。
保存manifest及SHA不等于本worker重建其完整上游strict。

## 2. 推荐唯一最小路线：复用已有几何核，补真实输入到单目标交付

不引入新loss或寻找新超参。MAIN若批准，复用上述冻结`optimize`，
以原两known/off根的真实共同history和原goal执行一次/根；交付固定第一向量。
本草案新增的是未来真实输入/输出的严格接线选择，不冒称新优化方法或原提案唯一算法。

令a为原真实full3 context末latent，g为原goal，优化变量X=[X1,X2,X3]，每个192维，X0=a。
Q使用当前共同identity192投影因子，不解释成未经证明的metric平方根，不拟合Q。

```text
L(X) = mean((Q @ (X3-g))**2)
       + sum(k=1..3, mean((Q @ (Xk-X[k-1]))**2))
X(0) = [a, a, a]
for t=0..63:
    X(t+1) = X(t) - 16 * analytic_gradient(L, X(t))
retain exactly all iterates0..64, their losses and gradients0..63
delivered_target64 = X1(64)     # fixed FIRST; no best/nearest/posthoc choice
target = float32(delivered_target64)[None, :]  # one declared transport cast
```

Q=I时公式及运算次序沿用原核；原核Python binary64，无tanh、L2球、投影、
manifold/LayerNorm重归一化或float32每步更新。传输cast仅在生成完后做一次，保存前后向量和rawbytes摘要；
cast后finite不满足即失败，不clip或修复。不把raw动作cap误套到latent。
平滑系数1、64步、eta16、最终iterate及FIRST索引全部事前固定。
已有核的逐步非增检查只作数值失败守卫；失败不降学习率、重试或回退别的iterate。
a=g时保留全部64次零更新；非有限/坏输入失败后仍占原分母，不拿goal/random替补。

该凸目标的每坐标Hessian最大特征值小于8/192，eta16低于保守1/L=24；
步长理由来自固定公式，不来自本两根结果。解析极小值Xk=a+k(g-a)/4只作RAM测试参照，
未来runtime仍执行原64步，不能直接用插值闭式值假称付过优化。

**科学弱点必须保留：**这个loss只表示几何平滑与原goal相关性，完全不表示冻结动力学可达性。
它实质上趋向平滑插值，而非一个学习到的subgoal prior。若MAIN要求“模型价值head优化的任意latent”
才是目标对照，目前原提案未给对应head/loss，不能用本方案冒充；届时需另立事前合同。
本草案仅请求批准上述单一路线，不并行尝试多个loss后按效果选择。

## 3. 固定输入来源及禁止泄漏

每根只读delivery中固定prior_FIRST行的公共z:[1,3,192]、a:[1,2,10]，
以及其authorities所指parity report `/rows/{root}/pixels/3/paths/hi/output`的原goal数组。
**prior_FIRST行的target不是goal，禁止取它作为优化goal。**
goal须按原parity数组descriptor/SHA读取，和原parity slot/goal来源相联；不重编码。
两根task身份、history/action bytes及RGB/encoder/preprocess/normalizer坐标沿原delivery authority核对。
原task_id分别为：

```text
root0 f5145aab0b44d21d0492d591590a73b4d4c9109903d5a97f18d01618f3bd5cc7
root1 697557899923163376df67cba615f13ebf1580166ca4de40a6759d12c927292b
```

未来优化函数的输入白名单仅root/off、原fullhistory、原goal及coordinate_refs。
不接prior target/program postcondition、未来raw见证、未来真实state/RGB、success或native标签。
父delivery含其它臂来源，只能用于身份连接，不能将其target/command送入优化核。
fullhistory原样保留；本几何loss只使用最后latent，不假称使用全部历史推理。
共同tracker接完整z/a，而非只接a末帧，过去真实action history不等于未来动作warmstart。

模型/encoder/projector/preprocess/normalizer/Q均继承原parity/tracker实际source、state与数组authority；
生成器不调用任何模型，也不把Hi和原P2不同权重模型叫同一个模型。
normalizer不重fit、不再归一化既有history，不增训练mapper，不用归一化动作当合法raw。
本次不认证raw inverse；原delivery的raw_scaler/raw_inverse缺口继续保留。

## 4. 费用与时距：只承诺明确边界，不称公平

生成阶段每根：65次loss、64次解析梯度、64次576参数更新；两根合计130/128/128。
这是CPU向量算术，不是神经backward；实际attempt/return/失败分别记账。
新增Hi high head/low predict/encoder/projector/J/VJP/神经AD/SVD/solve/physics/训练均0。
不需要Hi模型head或J，也不能把没有这些调用称为获得相同动力学约束。
旧已付proposal/编码/strict费用只继承引用，不退款、不虚记新调用；不同算子FLOPs=null。
优化阶段inclusive wall须包含读取、核算、运输、落盘的独立边界，核内部wall另列；不重叠相加。

若以后MAIN另批准真实共同tracker运行，建议不重调既有固定spec：
H3、samples8、iterations2、topk2、var_scale1、seed0、segments(1,2)，
全horizon zero/no-witness初始化，原unclipped_normalized_gaussian域及identity192 terminal MSE。
每根12 predict API/54 forward samples，包含两个各3步的mean replay；
两根24 API/108 samples；每根action_encoder/low_pred_proj各12调用，
action_encoder tokens216、low_predictor context tokens162、projection rows162。
新增encoder/high/J/VJP/AD/physics为0。这些是未来完整成功路径的固定预算，**本轮未授权执行**。
若新跑三臂六槽同spec则72 API/324 samples，不能用旧goal-target smoke的24/108冒充四target或三臂执行。
不能只跑arbitrary后将旧其它target的未dispatch状态标为已比较。

prior_FIRST物理时间仍null；program首postcondition来自source raw5；共同tracker terminal为H3。
本X1只是三潜在优化位置中的FIRST，**未认证物理cp5**；原核checkpoint_steps=[5,10,15]
仅保留为原结果字段，不升格成本新三臂时间等价证据。
不新增X2/X3三个依次tracker任务，不偷加调用。三目标的生成费用/时距不匹配，
same_generation_budget=false、time_alignment=false、equal_FLOPs=false。

## 5. MAIN批准后的最小交付接口与封存顺序（本轮不实现）

1. MAIN批准本协议后，新独占实现绑定本协议、冻结数值核/三件套及原delivery完整实际SHA；
   导入MAIN既有receipt证据并核选定数组，不伪称worker重建fullstrict。
2. 解析已存z/a/g，先封存两task、goal/坐标摘要、上述固定算法/预算；再调用一次optimize/根。
   两根均保留，包括失败。保存0..64全历史、实际费用、float64目标及cast32目标；不择优。
3. 新target delivery companion返回两行，而不是覆盖旧六槽missing：
   `{root,condition,task_id,parent_delivery_sha256,history_sha256,action_history_sha256,
   goal_sha256,optimizer_source_sha256,optimizer_result_sha256,target64,target32_sha256,
   controller_payload:{z,a,target},source_time:null,status,resources}`。
   failed行payload/target保null并存first cause，不换根或target。
4. 来源/数值验收通过也只表示target可交付，不能叫低层可实现。
   实际tracker/HOST CUDA、raw inverse/physics及tracking/native评估仍须MAIN后续单独审读授权。
   所有跟踪误差针对本臂实际cast32 delivered target；未知/earlystop不padding、不造ratio阈值。

未来RAM测试至少核：旧核公式/完整64步不变、同goal/fullhistory、FIRST固定、
goal与prior target混淆拒绝、额外future/outcome字段拒绝、cast前后SHA与finite、
两known/off完整分母、坏来源/坏schema/NaN/bool、失败不退款、原missing逐字未改。
协议阶段未新运行这些测试，也没有新增source/tests或实际目标结果。

## 6. 批准请求与当前结论

可行：已有直接latent数值核可复用，真实history/goal的来源已存在；最小新增工作是合法实例化与单目标交付，
不是再造“优化器存在”证书。待MAIN批准的问题只有：是否接受**此明确的几何surrogate+固定FIRST**作为
新探索性arbitrary对照，而非声称原提案给过确切loss/时间公平性。

若批准，后续按本协议实现，不看结果改超参；若不接受几何基线，则科学缺口是缺指定的latent价值/可达性目标
及相应已认证head，而不是可以用随机target、action endpoint或未来真实state填补的资料缺失。
P1/E1/P2/scientific_admission保持false；known2不是独立确认，不引用目标距离下降作为效能。
