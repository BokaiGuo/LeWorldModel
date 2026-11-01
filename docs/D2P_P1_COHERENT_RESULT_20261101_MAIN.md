# 包常量候选与真实前缀干预

P1仍未通过；以下是单个已知平移根上的开发结果。

## 候选改法的真实闭环

只改变初始化：structured用五步求和的Jacobian生成常量包响应，
其余臂高斯初始化也变为包内常量；优化、反馈、评分和预算保持不变。
四臂均真实执行20步，16节点通过保存核验，0/4原生成功。
新旧H4相关测试79项通过（工程证据，不是科学通过）。

|臂|原H4终态jointXY|包常量终态jointXY|
|---|---:|---:|
|structured|403.37586|138.68464|
|native|224.98585|215.79410|
|random|257.09238|200.17730|
|immediate_progress|414.65936|296.71078|

共同初始jointXY为153.35168。structured的cp5物体位置和角度均未改变，
jointXY却增至159.40150；终态物体位置误差从153.17534降至102.34505。
不能用这一个任务替代独立泛化或强基线比较。

结果：`le-wm-main/results/p1_coherent_analysis_20261101_main_MAIN01.json`。

## 同动作预测误差已补齐

原runner只有0/5/10/15编码，20步仅保存PNG；因此首版诊断仅完成24包。
补做8次真实终态CUDA编码和8次预测后，完整32包有：

- 4包预测Q变化与真实RGB-Q变化反向；
- 15包真实RGB-Q变化与物理jointXY变化反向。

这些包共享一个根和相关轨迹，不能当32个独立样本或泛化失败率。
coherent structured首包：预测ΔQ=+0.053408、真实ΔQ=+0.024407、
jointXY变化=+6.049819，预测MSE=.011789。
所以这个首包的非即时进展不是只由预测符号错误造成的。

完整结果：`p1_centered_terminal_cuda_20261101_main_MAIN01.json`，
SHA `d44bc567500977ca7d9a32ac86a09e7927dce80fa6b2db8d7090c74545894b44`。

## 五前缀共同15步后缀与zero后缀

固定四臂前缀加zero前缀，各执行共同structured已执行尾部和zero尾部，
共10条轨迹，全部完成20步，无原生成功。共同尾部是固定动作干预，
不是为其他前缀各自优化的计划。

|前缀|共同后缀终态jointXY|zero后缀终态jointXY|共同−zero|
|---|---:|---:|---:|
|structured|138.68464|160.07524|−21.39060|
|native|177.24124|173.02227|+4.21897|
|random|225.83618|207.53336|+18.30281|
|immediate_progress|355.90474|315.39471|+40.51002|
|zero|286.81569|153.33622|+133.47947|

这支持“前缀改变了该后缀的任务后果”，不等于“后继状态的可规划能力更强”。
immediate_progress是短视筛选臂，不冒充严格greedy最优解。
随后对五前缀使用真实cp5 RGB、同一优化器与预算各自生成后缀并真实执行，
结果见下节。共同尾与各自规划是不同实验，不能互相替代。

轨迹：`p1_coherent_intervention_20261101_main_RUN_MAIN01/report.json`。

## 五前缀各自规划后缀：CUDA与真实执行均已完成

每个前缀重新编码实际cp5图像；四个已有前缀的编码与先前保存值逐位一致，
zero是新实际编码。统一coherent structured H3优化器、六次更新、相同Q和预算。
先前计划参考统一规定为当前实际前缀加zero15，不冒充原模型选出的计划。
每前缀单列1次encoder调用；节点各390次predict、240次backward、
81次replay/selection，五份编码及五份节点durable账本均复算一致。
这是节点调用账本，不是独立GPU kernel计量或完整方法总成本优势。

模型报告：
`p1_coherent_own_suffix_20261101_codex_RUN_MAIN01/report.json`，
SHA `e14a4e701ed4983e48801b2ed813f9daf8c0373db1771af4415118690b19cdc7`。
封存后独立核验候选交换、argmin、hold参考和选中动作，再执行全部五条计划。

|前缀|共同后缀终态jointXY|各自规划终态jointXY|各自规划原生成功|
|---|---:|---:|---|
|structured|138.68464|120.45650|否|
|native|177.24124|134.01678|否|
|random|225.83618|232.59507|否|
|immediate_progress|355.90474|409.56537|否|
|zero|286.81569|184.34910|否|

五条均完整20步，无停止或删失；实际cp5状态/RGB与规划输入精确绑定。
另复核全部100个执行动作、25个保存检查点图像、来源和物理指标。
物理报告：`p1_coherent_own_physical_20261101_main_RUN_MAIN01/report.json`。
该比较是cp5后的open-loop15，不能冒充整个后缀持续反馈的验证。

现在有同一已知根上的正向链：
模型事前选中首包，预测Q与真实Q都暂时变差，物体无即时进展；
相同后缀的效果改变；同预算各自规划时structured仍优于其他四种前缀。
但仍无原生成功、没有独立根统计，也没有同协议公平强规划基线；
immediate_progress亦不是认证的same-bank greedy。因此P1保持未通过。
训练来源未知也不因上述开发实验而被解决。

## 补齐真正的同候选库greedy：共同后缀物理已完成

从原structured的八个受保护首包中，按同一个模型Q的首包预测代价
确定性argmin选出index3；原structured选中index6。这个定义是模型即时
greedy，不保证真实物理即时最优，不应与原immediate_progress初始化臂混用。
选择实际计费8次predict/action_encode/replay/selection，无新训练。

greedy的共同15步后缀与zero15都已完成20步，无原生成功、停止或删失：

|指标|same-bank greedy|
|---|---:|
|初态jointXY|153.3516843180|
|cp5 jointXY|209.4689452362|
|共同后缀终态jointXY|356.7486622755|
|zero后缀终态jointXY|214.7750495214|
|共同−zero|141.9736127541|

主线程复核了40个实际动作、10个检查点PNG、共同/zero的cp5状态一致性、
protocol散列与全部绑定来源。该结果仍只是共同后缀干预，不能替代greedy
各自规划后缀后的比较；其真实cp5 RGB独立规划尚待完成。

物理报告：`p1_samebank_greedy_physical_20261101_main_RUN_MAIN01/report.json`，
SHA `9366d4017fba6b5733a7174af21c0a0466f82fcb8bd2db9a46a52190e163c667`。

## same-bank greedy各自规划和物理执行完成

随后完成该greedy实际cp5 RGB的主机CUDA编码、同预算H3 own-suffix规划，
以及封存计划的20步物理执行。两项规划源测试、三项执行消费器测试通过；
这些是接线测试，不代替以下实验。节点实耗与其余五前缀一致：
390 predict/action_encode、240 backward、81 replay/selection、1 Jacobian/SVD，
另计实际cp5的一次encoder调用。冻结模型哈希前后一致。

|前缀|共同15后缀终距|各自规划15后缀终距|原生成功|
|---|---:|---:|---|
|structured|138.68464|120.45650|否|
|same-bank greedy|356.74866|348.51032|否|
|zero|286.81569|184.34910|否|

greedy own-suffix选中index2，真实执行满20步，没有删失；cp5图像与
状态和实际动作历史精确绑定。structured在同一根的这一条件后缀比较中
优于greedy和zero，但它们不是独立样本，亦未证明强基线或泛化优势。

新反馈编码也给出明确预测—观测失配：
根Q=0.7930806385；greedy首包预测Q=0.6937566646；
真实cp5 RGB-Q=0.8195884878。其物理jointXY增加56.1172609182。
即模型预测即时改善，实际编码与物理指标都变差。不得因为该对照实际
变差，就把它称作“真实即时greedy”，也不能据此宣称整个模型失效。

CUDA报告：`p1_coherent_own_suffix_samebank_20261101_codex_RUN_MAIN01/report.json`，
SHA `7402564db52792ed3a7027b63aa7328d6c9c7b59b0e7ca881a000a16515919e0`。
物理报告：`p1_coherent_samebank_own_physical_20261101_codex_RUN_MAIN01/report.json`，
SHA `8ed6bf4cc60a33204908464e3e3a7a69bb7b4f85200d86abcd07e31e112e3501`。

主线程新增六前缀汇总器，重新核验18条真实轨迹的动作、90个检查点PNG、
来源以及全部物理指标，并按实际前缀/cp5匹配共同、zero和own分支。
汇总：`p1_coherent_six_prefix_summary_20261101_main_MAIN01.json`，
SHA `b668efc76320fca501567ad74b35b035f19b8beeb9dd4b079661af23af563ebe`。
包含首包各项即时变化、共同绝对终态、相对zero的变化、own终态和停止状态。
仅一个已知开发根，18条轨迹均无原生成功，P1仍未通过。

## cp10/H2、cp15/H1真实反馈：六分支全部完成

实际重放各自前10步并精确核对cp0/5/10，再在线编码当前RGB、重新规划、
实际执行并用新cp15图像规划最后一包。全部12个节点/6条轨迹完成，
没有模型训练或CPU神经降级。旧计划的实际已执行首包和当前图像、历史、
尾部散列绑定；旧尾参与当下重评分，费用计入节点账本。

主线程从封存数据独立重建了上下文、carry、候选交换、最终argmin和
实际派发动作，核验24份编码/节点账本及物理帧/指标。
额外实耗12 image encode、1956 predict/action_encode、2880 backward/VJP、
588 replay/selection、12 Jacobian/SVD；这些是dispatch计量，不是GPU内核计量。
物理操作另计，不由账本中的physical_steps=0推断没有执行。

|前缀|cp5后open-loop15终距|增加cp10/15反馈终距|反馈选中index(cp10,cp15)|
|---|---:|---:|---|
|structured|120.45650|138.68464|8,6|
|native|134.01678|208.71029|4,8|
|random|232.59507|232.59507|8,8|
|immediate_progress|409.56537|421.31271|8,1|
|zero|184.34910|201.98006|6,6|
|same-bank greedy|348.51032|289.40610|2,6|

index8表示保留旧尾，其余为当前新候选。六分支均原生成功false。
所以真实反馈不是统一有效的修正；不能把greedy改善外推到其他前缀，
也不能把额外反馈成本忽略。structured终态恰与原coherent闭环一致，
不算另一个独立成功或新的根。

原报告：`p1_six_prefix_feedback_20261101_codex_RUN_MAIN01/report.json`，
SHA `bc5b3a8af924c7651793ed50541a3c6fc52374f26d2f62816427a2e03d7c6a4b`。
保存数据复核：`p1_six_feedback_saved_20261101_main_MAIN01.json`，
SHA `374425052e2f2a7b2df6156227dd099bae4a4588d32171bc2a15f014a3a19df0`。

## 完整六前缀×七后缀干预：42条实际轨迹

预先固定zero15及六个已封存own后缀，施加于所有六前缀，未按结果选尾。
42条全部完成20步，无原生成功或删失。核验840个实际动作和210个帧，
重算每个检查点的物理指标。下表是jointXY绝对终距，不是候选评分。

|前缀＼尾部|zero|S-own|N-own|R-own|I-own|Z-own|G-own|
|---|---:|---:|---:|---:|---:|---:|---:|
|structured|160.08|120.46|197.89|170.14|218.13|194.69|233.66|
|native|173.02|179.48|134.02|168.69|199.02|200.39|237.29|
|random|207.53|211.88|238.14|232.60|259.19|253.77|305.83|
|immediate_progress|315.39|343.00|346.36|323.86|409.57|369.62|418.40|
|zero|153.34|284.43|242.92|168.82|208.62|184.35|333.84|
|same-bank greedy|214.78|349.96|365.55|239.00|272.98|266.87|348.51|

S/N/R/I/Z/G按表中六前缀排序。全部六条非zero尾部下，structured终距
优于same-bank greedy；但相对zero仅3/6条绝对终距更好，按各自zero尾
调整后为4/6条。不能宣称对所有尾部都更有能力。相对native的zero调整
优势仅在structured自身尾部出现，说明已观察的能力作用具有动作依赖性。
这些计数是单一根上的有限尾部描述，不是六个独立样本或正式统计检验。
禁止据此把真实最优尾部回填为模型选中结果。

汇总：`p1_suffix_family_saved_20261101_main_MAIN01.json`，
SHA `09e07b75fcd924a4f54f612735edcc076c5e7de1935e7174d4c481a32d1b3930`。
两条新证据链仍未打开P1。下一项固定H3前视、40步滚动控制是另开的开发
合同；延长时域不改变或追认本20步合同的失败，也不解决来源/泛化缺口。

## 固定H3前视、40步滚动控制：完成并复核

新合同六前缀均重放同一已封存前10步，在实际10/15/20/25/30/35边界
固定前视15步。每次重新编码真实图像；实际旧尾两包加zero一包参与重选，
这是明确标记的hold-extension，不冒充未改变的原计划。
真实边界与优化器逻辑边界5分开绑定并复核。

运行前审查修正了新runner的状态汇总：合法提前成功/安全停止不能自动
归为基础设施partial；现分开记录实验所有分支有效终止与是否全部跑满。
4项合同测试通过。此前20步实验均跑满，因此此报告修复不改变它们的结果。

本轮五分支完成40步，random第31步canvas-censored，零基础设施失败、
零原生成功。每组固定20步和40步的实际结果如下：

|前缀|新控制器cp20 jointXY|新控制器cp40 jointXY|终止状态|
|---|---:|---:|---|
|structured|164.94051|88.96179|completed40|
|native|212.64884|104.01392|completed40|
|random|217.67909|缺失|第31步canvas-censored|
|immediate_progress|334.16064|408.37763|completed40|
|zero|241.72702|158.97831|completed40|
|same-bank greedy|363.36224|444.34978|completed40|

structured的cp40物体位置误差42.93563、角度误差0.14406、
推杆误差77.91490，仍不满足原生成功。不能只取其角度改善宣称任务完成。
固定H3在cp20并未优于原open-loop15的120.45650；cp40降低到88.96179
同时用了更多真实步骤和计算。因此不能归因成等预算优势或旧合同通过。
random最后观测jointXY=442.06414属于cp31，绝不填入cp40或参与完整终点排名。

共完成35个主机CUDA规划节点，实际dispatch：
35 image encode、13650 predict/action_encode、8400 backward/VJP、
2835 replay/selection、35 Jacobian/SVD。模型哈希未变。
主线程独立核验所有保存节点的真实上下文、旧计划及zero延展、候选交换、
当前argmin、派发动作、账本、图像和物理指标；未做额外模型重跑或内核计量。

原报告：`p1_fixed_h3_40_20261101_codex_RUN_MAIN01/report.json`，
SHA `1106877d113223c2d9bb833930bd1f889b7c1a335c442a19363fae9a1dc4772f`。
复核：`p1_fixed_h3_saved_20261101_main_MAIN01.json`，
SHA `2785bf839acb6f73303f40dcecb12bdff7a58d3d673c962342d22aee3cc42edd`。

**本轮结论仍是P1未通过**。这些实验支持继续研究结构化前缀，但没有独立根
机制合取、原生任务成功优势、完整方法强基线和总资源匹配。
未发现可把本批有效负结果直接消除的动作/RGB/carry接线错误；
不据此宣称排除了所有代码问题或理论不可行。确认方案草案
`D2P_P1_COHERENT_CONFIRMATION_DESIGN_20261101_HYPATIA.md`仍未注册；
来源认证缺口仍见`D2P_P1_SOURCE_AUTHORITY_GAP_20261030.md`，不得用
开发根重复试验填补独立确认。
