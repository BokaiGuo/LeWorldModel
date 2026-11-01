# bounded-native-CEM known2：待MAIN审读的前瞻协议

独占stem：`agent_lewm_bounded_native_cem_20261029_v1`。
本阶段只新增本protocol；MAIN批准后才写该stem源码/tests/innerdoc，不先实现或生产。
动机是旧官方returned mean root0 raw越界；不是对旧动作clip后重用cost，也不只保留root1。
这是新bounded对比方法，不称官方复现、等预算/等FLOPs、公平排名或Hi训练scaler认证。
本协议在已知旧结果之后提出，不是独立确认或旧gate修复。

## 1. 固定输入和优化参数

两根仍为原moving/off source_slot0、1，固定该顺序；同原goal、actual0/5/10 full latent history、
原raw10 warmup及其两个normalized incoming packets，不重选目标/根、不新encoder/physics。
原task00/01导出与批准blind输入、loader、weights/config、source SHA闭包沿原已认证P2材料绑定；
不将164 branches或两个known roots冒充fresh。
P2 PUBLIC_ACTION_MEAN/STD float32来自`experiments/run_pusht_counterfactual.py`及原task，
由`pusht_d2p_packet_contract.normalize_live_packet`处理，xy按时间平铺5次，不额外除100。
保留`training_provenance_verified=false`，不借Hi/sklearn scaler名义认证。

安装版`CEMSolver`采样/elite更新代码不改，不monkeypatch全局。
N=300，I=30，topk=30，batch_size=1，var_scale=1；每root新建solver，显式seed=0。
这里seed0是MAIN本轮事前固定的前瞻协议，不是用户指定；安装版此前默认seed1234保留，不能倒称原默认就是0。
H2、每packet5raw、每raw二维，优化参数shape `(1,2,10)`。
本H2与P2共同controller H3的预测时距及资源不同，不得直接作性能排名或称同预测合同。
初始化完整零logit mean和单位logit std；每轮sample0强制为当前logit mean，沿安装版代码。
elite按实际cost topk，均值和torch.std correction1沿官方实现；不加std floor、momentum、重试或补跑。
相同cost的tie按实际原生indices记录，不事后择优改tie规则。

## 2. 严格一次mapping与实际指令

优化变量是无界logits，不是normalized动作或raw动作。
每轮全部300候选：float32 `raw = torch.tanh(logits)` **只做一次**，reshape为两个raw5x2。
不clip、不另加packet球投影、不重复tanh、不做inverse-normalizer。
包含端点的raw[-1,1]逐元素精确检查；tanh饱和到±1仍属合法域，不用epsilon放宽。
通过原P2 `normalize_live_packet(raw,mean,std)`一次生成模型tokens，和原warm2拼成四tokens。
实际保存logits、mapped raw、normalizer输入输出、全部native IO，不以CPU重算tanh冒充GPU字节相等。

30轮后返回的是安装版**logit elite mean**；对它做一次tanh，得到最终交付raw10 float32。
该raw不是raw elite算术平均，也不是最后一轮已评候选winner，不用两者cost替代。
先durable封存final logits、raw10、bits/SHA；后续实际评估从该raw字节重新加载，不再tanh。
调用原P2 normalizer得到实际部署tokens并保存，正是下节whole/segmented实际使用的tokens。
不做“normalized mean逆变换→再正变换”的捷径，不沿用旧官方mean或旧mean cost。

## 3. 预测封存与评分：避免自适应CEM的逻辑矛盾

自适应CEM下一轮依赖上一轮cost，因此不能宣称先完成全部30轮预测再统一计算planning cost。
每轮Cost adapter分开调用官方cached `LeWM.rollout`与官方`criterion`：
先将本轮全部300个完整H2 predicted_emb和模块IO durable封存，再调用criterion、保存cost、更新CEM。
不直接用合并get_cost后再倒称cost之前已封存；模型`get_cost` API此协议调用0次。
规划cost仍是原native192 terminal float32 sum192，不fit Q；mean192只作另列诊断。
目标只进criterion，不作为predictor输入；不读取未来physical outcome。

最后两个root的交付raw均固定后，各做一次独立whole2和segmented1+1：

- whole：官方cached rollout，真实原RGB3/context，warm2+部署tokens2，AE4一次、predict/proj各2次。
- segmented：AE3→predict3，再用自己的pred15移窗，独立AE3→predict3；没有伪造pred15的RGB。
- 四路全部完整预测输出先durable封存，再统一计算四个final criterion。

最终cost只绑定这份actual raw/部署tokens和布局；末轮elite平均只保留其原名，不作mean评分。
whole/segmented按全部future384元素、frame15、terminal20报告bitwise/maxabs/ULP；
若非bitexact如实报告，不调整AE布局、不引入parity容差、不升级旧receipt。
全部两根和失败/partial槽保留，不自动生成可部署子cohort或物理授权。

## 4. 固定新增付费预算（整个两根job）

| 轴 | CEM规划 | final raw交付重放 | 合计 |
| --- | ---: | ---: | ---: |
| 自有cost oracle API | 60 | 0 | 60 |
| 官方model.get_cost API | 0 | 0 | 0 |
| 官方rollout API | 60 | 2 | 62 |
| segmented model.predict wrapper API | 0 | 4 | 4 |
| criterion API | 60 | 4 | 64 |
| action_encoder API | 60 | 6 | 66 |
| AE samples | 18000 | 6 | 18006 |
| AE tokens | 72000 | 20 | 72020 |
| predictor API | 120 | 8 | 128 |
| predictor samples | 36000 | 8 | 36008 |
| predictor context tokens | 108000 | 24 | 108024 |
| pred_proj API | 120 | 8 | 128 |
| pred_proj rows | 108000 | 24 | 108024 |

规划每native predictor batch300、context3；交付重放batch1、context3。
wrapper API与内部模块不能相加称“总预测”；sample/token也不能当API数。
每root完整9000候选，全部两根18000候选；交付重放额外8 predictor是付费成本，不偷藏在规划内。
没有额外best/nominal replay、padding、重试。原8encoder/8projector继承单列一次，
不把旧官方native CEM的已花费job当本新方法的免费规划或同预算依据。
新增encoder/projector/AD/physics/training均0，冻结模型/normalizer，无CPU神经fallback。

## 5. 拟采用的工程上限与交付边界

建议本job output1GiB，含64MiB final reserve；disk floor200MiB；wall900s含verify/load；
GPU allocated3GiB/reserved4GiB。原native CEM已保存约同量full IO，新增mapped raw/tokens约数MiB；
实现时用完整shape/字段census测试保守上界，若不足先报MAIN，不缩分母/精度或减少候选记录。
CUDA cooperative守卫不冒充kernel抢占，partial保原attempt/return完整分母。
MATH是requested context，实际TF32off/autocastoff/inference/CUDA-f32守卫另列，不冒充kernel证书。
文件SHA与模型tensor hash不冒充同buffer loaded-bytes或训练统计证明。

MAIN批准后新runner最小接口：`prepare / verify / run / consume`，原输入identity闭包沿旧合法材料，
新独占Store/journal，不调用旧CEM结果筛选；RAM tests包括mapping一次、所有raw域检查、
normalize输入输出接线、CEM原生callback更新、全9000候选分母、实际raw重载和seal-before-score，
及真实Store写盘→reader回归。新结果与作者报告值隔离，不能称P1准入/控制收益。

当前无runner、无production prepare、无NN/physics/fullstrict/下载/后代agent；等待MAIN审读协议。
