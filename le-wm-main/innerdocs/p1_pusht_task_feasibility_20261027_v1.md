# PushT 两根：P1任务可行性与下一批见证优先设计

仅saved arrays/源码审计，不调用模型、autograd、环境构造或physics，不改sealed产物。
experiment-code用于独立日志/测试；experiment-design用于分阶段及对照设计，任务约束优先，
不采用技能通用模板里的训练/调参/作图要求。下面是新设计建议，不是已获准运行的protocol。

MAIN追加方向已落实为独立 `p1_constructive_reachable_source_20261027_v1.py/tests/doc`：
优先固定2–4个witness生成reachable goals，用原生success验证，不要求direct困难或tau authority。
下方较大stratified设计仅保留为后续P1确认设计草案，不阻塞这一小探索producer。

## 原proposal要求，不是事后把失败改成任务不合格

原文 `/home/gbk/.codex/attachments/861f6764-1b25-4227-bf15-f1a237ba7d8f/pasted-text-1.txt:681–687`
明确P1为：模型已有对应动力学，当前直接梯度易停滞、但两阶段动作可达，再检验短期
重配置→后继progress→真实闭环优于单层搜索及普通latent subgoal。§4.2强调PushT接触
不连续时用实际短动作rollout和顺序差异，不能仅拿连续Lie bracket或rank替代。
当前两根没有达成这个前提；不能把空的任务资格当成理论失败，也不能以新容易任务追认旧P1。

## 当前source / environment / action合同

原 `p1_conditional_calibration2_main_20261027_v1/sourcepack.json` 固定SHA
`728f14cb83c4fcd0ea4f5348bad2bff5b5dd372ab0185b6e6b3a26dbfda58057`。
generator `sample_state` 复制native随机goal；物体XY加各轴uniform[-16,16]、角加
uniform[-π,π]，再把agent放到物体随机边附近。domain只检查画布、contact gap[-.5,3]和
object位置>20或wrapped角>π/9。有限first-legal，不读取未来，但不检查任务可完成。
无条件角扰动有8/9概率超过20°、1/2超过90°；这是recipe先验，不是两根的总体估计。
原native goal agent位置独立从[50,450]²采样；靠近物体的agent初态与该goal没有到达证据。

两根原NPZ的初态/goal、selected_attempt、source seed/hash在新报告逐根保存。旧历史是
causal padding；后来的observed0/5/10 zero warmup确为真实执行且恰好原数组不变，
不把这两种来源描述混为一谈，也不把warmup算进未来10步reach预算。

installed `stable_worldmodel/envs/pusht/env.py` 固定SHA
`c6d2104adc368ff1c352edc0e35ac74da9f07fda3a748ddbef0ba6b7a2d1daa2`：
relative action target = agent.position +100*raw；PD kp100/kv20、dt.01、10Hz，
每public command10个physics step，未来10命令=1秒。不是10次位置瞬移，也不是绝对像素action。
H2为两包各5×2独立坐标，每包logit L2≤.75√5=1.677050983，float32 tanh后dispatch。
这不是整个raw box，也不是full20一个共同球。逆tanh数值核验容差2e-6仅用于float32舍入。
∑100||raw_t||≤750px仅是名义target-offset和的上界，不是agent真实可达距离上界；
没有接触冲量/角速度上界，不能据3rad误差推出1秒必不可达。

原任务normalize为(raw-mean)/std，mean[-.0087,.0068]、std[.2019,.2002]。
`run_pusht_counterfactual.py`中的旧helper另有/100说明，但当前桥不调用该helper，只读常量；
不能据那段注释单独断言本批缩放错误。checkpoint README只标训练集quentinll/lewm-pusht，
没有本地已认证的训练action分布/normalizer provenance。动作合法≠训练支持已证：新报告只给
实际normalized范围，不把几倍std当成分布外证书，也不建议无证据改为/100重跑。

## 已保存实际：目前无合法native reach见证，不是不可达证明

新脚本复用独立sameprefix审计：2 roots×3 roles×41 probes=246完整H2，738PNG和34196
hash-chain事件全核验。另绑定原witness8、sampling4、B8四分支的固定report SHA、prepare、
完整journal、原command/goal/root/奖励/终止，合计262分支、2620 public steps。
这些分支仅两根重复实验，绝不是262个独立样本。补充三批未重新逐PNG审计，范围明确分账。
原生success和独立wrapped诊断分别复算；不更改官方termination/reward。

|root SHA前缀|初始agent误差px|object XY误差px|wrapped角误差rad|sameprefix最佳末角误差rad|sameprefix最大实测转动rad|
|---|---:|---:|---:|---:|---:|
|02a13916|242.044137|9.167074|3.017657|2.938905|0.078752|
|05844ca2|318.982782|13.691348|1.782513|1.782513|0.141427|

第一根greedy/two-stage的最好末allXY误差仍分别179.122/165.991px；第二根两role均
262.281px，native阈值是<20px。第二根zero角色有物体转动，greedy/two-stage在这些probe
内最大转动均0：不能笼统声称模型选中prefix提高了接触能力，更不能用root1另角色转动拼机制。
41探针是零+10坐标×正负×两尺度(.25,.5)，不是完整十维suffix搜索。最大转动是离散返回时刻
wrapped位移，未声称连续路径弧长/未观测子步角速度。

262分支原生和wrapped成功均0、负native角项均0（实际数字由测试断言）。这只回答所检查
这四批没有见证；不声称穷尽全仓每条未来，也不声称全合法球不可达。当前两根旋转过大/agent
目标遥远与短局部动作支持明显不匹配，属于任务前提未证的具体风险，不是已识别的唯一因果。

native success同时要求agent/object合并XY误差<20且角<π/9；reward是负完整state7范数，
含agent速度。静态goal RGB也有agent。因此agent不是可以任意删掉的纯nuisance：它可能主导
latent排序、造成agent-only progress，但去掉agent goal就是新任务，不能沿用旧success。
保留official+wrapped+object-only+速度分解各自标签，不把object-only诊断升格为主成功。
B8 GD仍改善、GN固定拒绝重复proposal的既有证据也不能自动称direct plateau；本审计不发明
epsilon_plateau/escape/tau，不把逆tanh容差拿来当科学阈值。

## 下一批预注册建议：witness-first，随后冻结模型测试

状态：DESIGN_ONLY，不能直接launch。下列支持/分层是建议值，不追认旧任务，不按本批成败挑根。

1. **先冻结任务生成，不用待测模型生成资格。** 新seed/source lineage与这两个已看根彻底隔离；
   先列固定slots、attempt上限和拒绝原因，禁止补满成功数。预声明H=10/20/40三个独立strata，
   各阶段边界按5步packet，逐packet保留相同logit cap；不在失败后给同根追加时间。原10步主张
   只能由H10回答，长H成功不反填。每stratum先8个calibration slots；正式confirmation数在pilot
   前由功效/精度目标另行封存，当前不能虚报统计功效。
2. **构造真实forward witness而非采样任意goal。** 每slot最多16条预声明接触/绕边/推送/转动
   动作序列；所有动作按同合法球、原freshRGB guard执行。先从实际安全root执行完整H，使用
   真实terminal几何生成包括agent的goal RGB/state，再从原recipe独立reset+10步observed history
   重放两次。目标重设会触发native reset内部physics，必须逐字段核root RGB、state、已记录
   dynamics与goal render，并在新目标任务本身重新验证成功；仅旧轨迹终点截图不够。
   不假定T角等价于完整state，也不添加free settle：收敛/settle动作若需要必须包含在H。
   若固定序列都无有效object操纵，slot保留失败；不能只造agent移动容易任务来冒称P1。
3. **隔离task builder与部署。** 可用state/接触几何构造任务及评价见证，必须标为privileged
   offline task construction；证书动作/未来RGB/body dynamics/筛查标签不进入模型selector。
   部署只收自己的真实RGB/action history、原goal RGB、已声明动作域；不能把witness当候选答案。
   对外主张限“witness-generated feasible distribution”，不称随机PushT分布上的泛化。
4. **独立冻结模型资格。** 先封存可行source bundle和真实重放，再封存checkpoint/config/
   exact transform/normalizer/支持证据。对有限B1/B3/B8及预注册强direct配置完整记录轨迹，
   阈值authority只来自独立calibration，不读取confirmation未来。分开三门：物理见证存在、模型
   能表达对应响应/两阶段目标、direct有限预算plateau+escape。任一未证不进入P1；有限预算
   plateau不是KKT/全局最优证明。若均不能满足，报告空eligible池，不扩大搜索直到出现正例。
5. **冻结P1比较及能力探针。** 同root同H同action球同真实反馈，单层强搜索、普通latent subgoal、
   greedy/zero与重配置使用预声明资源向量，先封存选择再执行；first阶段必须按冻结任务指标
   验证非贪心，真实own-prefix后重新编译suffix，并以共同探针测量真实目标方向响应。
   诊断state不返回选择器，不按physical probe挑动作。保留全部根/失败，root配对与多重比较
   按冻结规则，必须同根满足非贪心、能力改变和最终native任务收益，不能跨根拼合。

**资源账本草案（不授权执行）：** 每slot16 attempts，每attempt构造1 episode+独立重放2
episodes；每episode warmup10+未来H，reset最多2，显式RGB每返回步+初态各1。
因此每slot上限48(H+10) public steps、96 resets、48(H+11) explicit renders，另96 reset内goal
renders、192 reset内physics steps；public内部若保持此env即480(H+10)physics substeps。
按每stratum8slots计，三strata总公共步38,400；这是较昂贵的上限，不是建议立即花满。
goal生成额外render/reset、失败中部分子步、源文件/像素存储必须由未来producer具体代码补齐
并预检后封存，否则上表不构成完整cap，拒绝运行。模型阶段encoder/projector/predict/AD/
engine/SVD/solve、wall/显存/磁盘上限尚待具体算法配置；不以未知cap假称ready。
source生成阶段模型调用必须0。MAIN可先审批更小固定pilot，但不得看pilot结果后重定义主比较。

## 复核命令与边界

cwd `/home/gbk/code/LeWorldModel/le-wm-main`；unittest不创建tmp或pyc/cache：

```bash
.venv/bin/python -B -m unittest discover -s tests -p 'test_p1_pusht_task_feasibility_20261027_v1.py' -v
.venv/bin/python -B -m experiments.spectral_reachability.p1_pusht_task_feasibility_20261027_v1 --output results/p1_pusht_task_feasibility_20261027_v1_audit01.json
```

去掉--output只输出stdout、不写文件；写报告独占新路径、<200KB、200MiB floor。
这不是神经/物理新实验，输出明确P1=false、global_unreachability_proven=false。
