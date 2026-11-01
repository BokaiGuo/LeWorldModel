# Experiment1 下一执行切片：共同低层控制器的两臂校准

状态：只读核查及本独占文档完成；未实现新runner，未运行模型、物理、网络或训练。
这是交MAIN审定的具体实现规格，不是运行授权，也不是已封存实验预注册。
旧matched RGB、三臂contract/runner/assets及全部结果不改。

## 原问题与唯一下一步

原始依据：`/home/gbk/.codex/attachments/861f6764-1b25-4227-bf15-f1a237ba7d8f/pasted-text-1.txt`
527–572行。Arbitrary臂是直接在latent中优化目标；learned臂来自已训练subgoal/proposal；
compiled臂先有动作序列再有中间状态。关键测量是共同低层的真实可实现性、执行偏差及最终成功。

下一步只做一件事：新增一个calibration-only共同controller适配器，让原moving两个known
root上的arbitrary目标与已有compiled目标，真正通过同一个144-predict低层求解器输出raw10。
先交prepare/verify、RAM测试及MAIN CUDA入口；本文件不实施、不启动它。不是把旧两套
controller结果拼表，也不是再写registry。该切片只关闭“没有共同controller实装”缺口，
不关闭learned-prior、独立根、完整E1或P1缺口。

## 固定输入与两种目标

唯一输入来源是原moving off两份blind task，按source_slot 0、1全取；使用
`experiments.spectral_reachability/p1_moving_sampling_baselines_cuda_20261027_v1.py`
的`load_export(path, expected_sha)['gn_task']`和原moving encoder的`load_task`闭包。
必须绑定实际原task/source/model/config/normalizer/goal/full-history；不得用zero-history loader。
原非零warmup为前5 raw `[.1,0]`、后5 `[0,.1]`，保留三状态与两incoming tokens。
goal、root、原task_id始终不变；tracking target是单独字段，不能覆盖task.goal再冒称同task。

下列arbitrary搜索是一个有限、弱的本地latent-search基线，不是官方Hi-LeWM/SAGE，
不是学得先验，也不是声称穷尽所有任意latent优化。参数必须在任何新预测/物理前由MAIN接受并封存：

1. `D=192`，`z=root_context`末状态，`g=task.goal`，原identity mean192 Q。
2. 每根PCG64 seed分别2028102801、2028102802，一次按行生成20×192标准正态`u`。
3. `s=norm(g-z)/sqrt(192)`，候选`w_j=z+s*u_j`；运算float64后一次转float32。
   不投影到模型可达集，不从动作rollout构造，不读取native结果或证书。
4. 直接按`(mean192((w_j-g)^2), j)`最小值选一个目标；保存所有候选、标量及原始随机draw。
   这是20次有限latent随机搜索优化，既不是梯度优化，也不是无搜索的任意噪声标签。
   若`s=0`仍保留原20项，不补抽、不改根。它可能是很弱的优化器，不能据此否定所有latent方法。
5. compiled目标固定使用已封存beam每根选中路径的prefix5预测；入口是
   `agent_kinodynamic_matched_domain_20261028_v1.consume(output, external_report_sha)`。
   原报告SHA为`d4976aeca357b5324c74be8b2b7e3bfc254e82b834b3acfe1b0ab90a1ef4614c`。
   从consume的prediction及完整节点/context/action witness提取，不猜result行号，不重选beam。
   task/root/goal/history/normalizer不等即拒绝，而非转换latent补配对。

compiled目标已由旧模型搜索选出，存在继承搜索成本和选择优势；这是本切片公开的限制，
不能把它说成两种目标产生过程等预算。目标均先seal，再开始共同controller；不能看新执行结果换目标。

## 共同controller：可实现的确切调用合同

四个dispatch槽固定`root-major × [arbitrary_latent_waypoint,dynamics_compiled_witness]`。
每槽从原root完整context重新开始，不把arbitrary执行结果传给compiled。
每候选为两packet、每packet五raw动作，共10×2；保留原logit球、tanh、raw界限和单次normalizer。

使用新的有限CEM loop，而非修改旧optimizer的全局目标：3轮×24候选，每候选完整H2，
每次真实packet都调用同一冻结transition。排序目标事前固定为
`0.5*mean192((pred5-w)^2)+0.5*mean192((pred10-g)^2)`。
两臂只有`w`不同；终点goal仍是原g。此复合tracking目标是新适配目标，不能称原beam搜索目标。

CEM参数继承`moving_prediction_matched_sampling_cuda_20261027_v1.contract()`：elite4、
初mean0/std1、std裁剪[.05,1]、ddof0、无平滑/incumbent注入、原cap与投影规则。
每根两个臂使用相同controller seed（2028102811、2028102812），独立初始化随机状态；
首轮随机draw应逐字相同，后续分布可因目标不同分叉。保存全部72候选/评分/动作/context。
发布所有72项已评候选的稳定全局argmin；不再best replay、不评nominal、不padding。

可以纯复用旧matched采样的`NativeCuda.guard/_step`、Store/Meter模式和模型加载闭包，
但新runner必须有自己schema/声明/四组meter；不能调用旧固定四组budget后冒用旧arm身份。
逐candidate保存pred5、pred10及其完整rolling context；不是用最终latent插值伪造prefix。

|资源轴|每root/arm|本切片总计|
|---|---:|---:|
|predict API、predictor samples（batch1）|各144|各576|
|action_encoder、pred_proj调用|各144|各576|
|新image encoder / image projector / AD / physics|0|0|
|选中动作额外replay|0|0|

144是成功完成路径的实际调用数，同时是predispatch上限；异常后不补齐、不退款。
原编码成本继承单列；arbitrary的20次CPU数组标量评分、旧beam 144predict/root及所有旧选择
成本分别披露。相同的是低层求解预算/动作支持/信息，不是proposal总成本、FLOPs或全部墙钟。
输出64MiB、free floor200MiB、dispatch/write wall900s可沿旧实现语义；不声称抢占阻塞native call。
attempt/return/failed、模块sample/frame/token数与first/secondary error都要真实记录。

prepare必须拒绝错误外部SHA/不一致root/非有限候选/不完整history/未知模型authority。
run只允许MAIN HOST CUDA、冻结eval、MATH、TF32off，无CPU神经fallback。
consume只读重建20点latent选择、72项CEM更新与argmin、576资源回执及完整raw10，不能重跑模型。
RAM验收至少覆盖两臂相同首轮draw、同一controller代码/预算、坏history/SHA、目标偷换、
少候选/额外predict、资源超限、错误后保留四槽及未派发项。测试通过不证明真实跟踪成功。

## learned缺口不藏进两臂结果

完整E1科学分母仍是2root×3arm；本切片仅有4个工程dispatch，另2个learned槽明确unknown，
不能把2臂结果送旧三臂score时删臂、填fallback或合成learned回执。
旧`agent_subgoal_gap_runner_20261028_v1.run`要求真实P1/freshness/physical authority，当前不可
伪填true；新calibration-only实现不得改旧gate，也不得借RAM_METADATA执行真实神经。

SAGE已有缓存官方接口是`load_subgoal_prior`后调用generator(history,goal,lowdim,offsets)，
不是可直接启动的完整本地包。当前H10在官方tau15/20/25下触发final-goal fallback。
后续learned的最小资产请求仍为实际visual generator、配套object及可信配置/统计/来源：
`pusht_visual_generator.pt`预期SHA
`8c834bac6f63f6dfa092140a9b2687a13cfed35b1c658bbeccd85d41fd297adb`，
`pusht_lewm_object.ckpt`预期SHA
`4656a13b878d83e417ba97d96257e77fdadc7a9ecc997031eb611c6494a4c270`。
它们的声明hash不是已持有认证。若沿本地冻结模型，必须先证明encoder/projector/预处理配对；
否则三臂共同迁移官方配套backbone并重做，不能训练mapper。官方首个已配置非平凡H50/[25,25]
也需共同重新声明任务时间语义，不能将procedural goal重标dataset future50。

本切片完成后仍需另行MAIN授权真实own raw10物理tracking及RGB消费，才能报告执行偏差和native成功。
native保持原7维状态/goal语义，成功位置包含agent+object；不以object-only或RGB Q代替。
两known roots不是独立确认样本；任何零/负结果全留。不存在当前可安全宣称完成的三臂性能比较。

## 交接状态与下一条操作

本次实际仅阅读原proposal、现存contract/runner/assets文档及matched采样/beam接口；
未生成prepare、未运行测试或任何实验。没有新可执行runner，故不提供伪造的run命令。
下一条操作是MAIN批准或退回上述唯一切片参数，然后授权新独占校准runner实现；
不是现在启动官方SAGE、绕过旧gate或重复现成registry。当前本文件可以直接只读核对与SHA冻结。
