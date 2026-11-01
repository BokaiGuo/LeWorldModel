# Native offline→live：四episode输入合同与动作条件fidelity诊断设计

日期：2026-10-27。**只设计，不实现、不prepare、不运行。** 本轮仅写本文，不改sealed源码／数据／任务／结果，
不运行神经、物理或网络。采用experiment-design技能的最小诊断与后续因果检查分层，不训练、不拟合Q／readout，
不新增科学通过阈值。未来所有实际CUDA和涉及native环境的执行均MAIN-only、另行授权。

## 1. 要回答什么，不回答什么

当前constructive使用with_target=False；本地H5已观察到绿色目标marker，但checkpoint实际训练run→
dataset revision→collector/render合同未闭合。public normalizer与MAIN全H5统计也不同，哪组是checkpoint
训练统计仍unknown。因此先检验**本地归档输入能否按当前native接口无歧义地送入冻结模型**，再测
在该归档上factual动作的next-latent误差及动作条件消融；不直接把live任务失败归因overlay或统计差异。

分三层输出：

1. **输入conformance**：帧／动作时间对齐、RGB transform、states3/actions2+outgoing、normalizer、设备、
   model接口和账本是否符合明确合同。这是工程检查，不等于误差小。
2. **归档条件fidelity／动作敏感性**：真实nextlatent标签下factual与固定zero/mask消融的描述性比较。
   文件名含train不证明这些episode确实训练过，也不证明它们是held-out；统一标`training_membership_unknown`。
3. **offline→live与overlay因果**：须另有同状态、同目标、同历史／动作的paired native render，且分别核实
   archive再现和live接口一致性。阶段A结果本身不是live泛化、控制成功或P1证据。

不读constructive witness，不给任何selector未来RGB／latent／native outcome。不把归档factual轨迹当
模型发现的动作；这些是dataset提供的标签和条件输入，不是新规划结果。

## 2. 固定选择：先封存metadata规则，不能看outcomes挑四条

唯一H5：`/home/gbk/.stable_worldmodel/datasets/pusht_expert_train.h5`。
只根据episode数E选择原episode ordinal：`[0, floor(E/3), floor(2E/3), E−1]`，按此顺序。
现存metadata记录E=18,685，故预期为`[0,6228,12456,18684]`；实现必须核对E，不能静默换成另一份库。
episode ordinal不是自行猜测episode_idx标签，二者映射均记录。

每条只取原relative step `0,5,10,15,20`的五张RGB和action primitive `[0:20)`，共20帧、80条二维raw action。
预检只核offset／length／step索引／shape／dtype／有限值；每个固定episode必须有step20（length≥21），
且这段不跨episode。若某条短、重复、缺失或损坏，保留四分母并fail-closed，**不换下一条满足条件的episode**。
不以成功、物体移动、接触、预测误差、action方差或green像素多少选episode；不挑最活跃窗口。

阶段A无需读取state/proprio/reward/goal字段值，episode_idx/step_idx只作时间身份核验；不据hidden state生成模型输入。
字段名、H5 attrs、文件stat及所选slice SHA写manifest；H5完整46GB SHA若无既有批准记录，不宣称已做全文件身份认证。
metadata、20个decoded RGB bytes、80条raw action bytes、offset/length映射分别绑定SHA，末尾重验所选内容／文件stat。
selected-slice身份不冒充整个dataset不可变revision。prepare可只封小manifest＋小数组，不复制旧大protocol或H5。

## 3. 两个teacher-forced窗口：精确时间表与因果隔离

记`I_t`为原step t RGB，`z_t=projector(encoder(transform(I_t)))`，
`A_i=raw_action[5i:5i+5]`为五步×二维time-major packet，`T_i=normalize(A_i)`。

| 窗口 | 输入observed latents | 两包between-action history | outgoing | evaluator标签 |
| --- | --- | --- | --- | --- |
| W10 | z0,z5,z10 | T0,T1 | T2=action10:15 | z15 |
| W15 | z5,z10,z15 | T1,T2 | T3=action15:20 | z20 |

两窗口都teacher-forced：W15用真实observed z15，而不是W10预测输出；不能将teacher forcing收益叫
open-loop两步预测准确性。W15的z15是在其合法过去，W10中的z15只作标签，不得反向进入W10模型输入。
z20从不进入任何预测输入。z0..20虽均需编码，模型调用参数由白名单window builder构造；
标签数组放独立evaluator payload，selector不存在，也不因误差结果改变动作／normalizer／checkpoint。
先durable封存所有window条件和消融定义，再预测；每次预测保存输入引用／SHA、完整输出和选用last-token索引，
最后独立evaluator才join目标latent计算误差。不能仅凭“脚本知道这是label”就声称OS级隔离。

**action时间语义是前置条件，不靠误差选shift。** 当前HDF5 reader的训练取slice／action不下采样／
packet reshape路径支持上述对应方式，但文件本身未给完整collector lineage。
NEW实现必须记录现存collector／writer合同或MAIN明确批准的alignment scope：确认action[t]是I_t之后
到I_{t+1}的outgoing动作，而非入帧动作／terminal padding。如果只有loader约定、没有归档生成证据，
可以明确按“loader-convention fidelity”运行，但不能标“dataset temporal provenance verified”。
不得扫描shift±1后挑误差最小者来修对齐；缺失来源继续unknown，不补造证据。

## 4. 固定native输入：不用全H5新统计，不用目标任务Q

保持原冻结checkpoint与config；采用constructive批准的same image_transform/_image_tensor、RGB224、
ImageNet归一化、原resize顺序、float32，CUDA batch1、MATH SDP、TF32关闭、eval/no-grad、model tensor hash前后相等。
不为使归档更好而新增crop、去green、颜色转换、encoder pooling或不同projector。
所有20张各encoder+existing projector一次，不dedup，即使像素相同也保留固定帧分母。

原public float32 mean `[-.0087,.0068]`、std `[.2019,.2002]`，每包：
`T=(A.reshape(10)−tile(mean,5))/tile(std,5)`，native context states `[1,3,192]`，
history `[1,2,10]`，拼outgoing后action_encoder输入`[1,3,10]`。
不额外÷100，不将归一化零当raw零，不重复每个坐标五次，不把单个动作复制代替五条实际primitive。
H5 expert可能超出constructive tanh支持；本阶段是**factual数据评估**，直接保留原raw，绝不clip/tanh成另一条轨迹。
因此不宣称这四段是constructive动作球内的部署计划，动作域不匹配单独列出。

调用语义必须与当前native `_step`一致：
`pred = model.predict(z_history, model.action_encoder(cat(history_tokens,outgoing)))[:,−1,:]`。
不额外encoder／projector，不重复pred_proj；native model.predict内部包含predictor及pred_proj。
不得用直接predictor输出冒充已pred_proj的native192 successor。若新runner不借task接口，也须保存上述context schema，
无需为计算下一帧误差伪造goal task或把z20装成goal传入planner。

public常数的training_provenance_verified保持false；MAIN本地全action统计只能作为已有来源诊断，
不在这次重新fit、替换或并行择优。训练数据真实normalizer未知时，结果准确命名为
`frozen_checkpoint + frozen_public_mapping on local H5`，而非“认证训练管线上的复现”。

## 5. 三种条件：动作消融不是新的物理反事实标签

每episode每窗口固定执行F/Z/M，顺序固定，无候选选择、无early best：

- **F factual**：两包真实history＋真实outgoing，action_encoder正常计算全部三包。
- **Z raw-zero outgoing**：history仍两包真实token，outgoing用5条raw `[0,0]`再按public归一化。
  仅改变outgoing；不是normalized token全0，亦不把全部history清零。
- **M outgoing embedding mask**：输入仍为F三包token并实际调用action_encoder一次；仅将输出action embedding
  的最后一个time token置0，前两个history embedding不变，再调用同model.predict。
  这是显式中间表征干预，不是attention mask、合法raw动作，也不自动等价于某条raw命令。

mask发生位置、dtype、shape及前后数组都保存，不能让测试／报告笼统写“masked action”而实现另一种消融。
禁止全历史mask或normalized-zero额外臂悄悄混入；若需要应NEW预声明，而不是看F/Z/M后补最好解释。
M只在输出已通过原action_encoder后置零，仍计其一次调用，没有“mask所以免算”的成本豁免。

三个条件都与同一**factual**z_next比较。只有F有对应实际执行轨迹；Z/M并无真实zero/masked successor标签，
故它们只是动作条件消融，不能用其对factual标签的误差称“zero物理预测准确性”或因果动作效应。
输入相同的factual history不代表它们的下个真实世界状态相同。

## 6. 保存什么、如何解释而不拟合或设阈值

每窗口存所有输入raw/tokens/history、三个action embedding输入／mask输出、三份完整predicted3×192及last192、
真实next native192、文件／数组entry SHA、module journal、模型前后身份。无训练、无decoder、无Q fit。

预声明算术：

- `E_F/E_Z/E_M = mean((pred_condition−z_next)^2)`；所有8窗口逐项列出，不能只报最有利均值。
- signed消融差`E_Z−E_F`、`E_M−E_F`；正值只表示在这个factual标签上F误差更小，无新通过margin。
- last-observation persistence `E_P=mean((z_current−z_next)^2)`，无需模型；另报真实latent change magnitude。
- `mean((pred_F−pred_Z)^2)`及F−M输出差，用来区分“对动作不敏感”与“有动作依赖但方向／幅度错误”。
- factual误差相对persistence、两窗口误差变化并列；target几乎静止时不能把小绝对误差当动力学识别。

可汇总为每episode两窗口的均值／范围及四episode等权均值，但独立单位最多4 episode，不是8窗口或24条件。
训练成员身份unknown、相同数据祖先可能相关，不做泛化显著性或P1统计通过判定。
不由这四条拟合新normalizer、选checkpoint、调renderer、调history、定误差阈值或选择下一批“好episode”。

F优于Z/M并优于persistence仅支持本地四段上动作条件factual预测有用；不证明对应训练数据一致或live泛化。
F不优于Z/M可提示动作利用、调用语义／分布或小运动片段问题，不能单独证明模型不懂动力学。
即使conformance全过，normalizer、归档生成／训练run的来源unknown仍保留；工程PASS和低误差是两回事。

## 7. 阶段A可执行预算与失败合同（未来MAIN）

预检saved数据不加载模型；NEW prepare≤256KiB，source closure只存manifest/hash，不复制大模型或H5。
CUDA独占一个NEW output，cap32MiB含partial与final（final预留1MiB），200MiB floor，启动至少cap+floor，
每次写入／调用前再验；单PNG≤1MiB，20张PNG/20个raw latent全部持久保存。

| 资源 | 固定cap／完整预期 |
| --- | ---: |
| archive episode / RGB / windows / conditions | 4 / 20 / 8 / 24 |
| encoder calls+frames / existing projector calls+frames | 20 / 20 |
| model.predict entry / predictor.forward | 24 / 24 |
| action_encoder calls / packet tokens | 24 / 72 |
| pred_proj calls / projected token samples | 24 / 72 |
| AD / grad / engine / optimization / physics / reset / render | 全0 |
| allocated / reserved GPU memory | 3GiB / 4GiB |
| 内部wall / MAIN外部timeout | 600s / 650s |

cap是工程上限，不保证硬件一定满足；拒绝CPU neural fallback、不下载、不隐式batch合并、不重试。
model.predict与predictor分别记录，不双计成48次预测；同理内部pred_proj不是额外existing projector。
每次module dispatch/return都durable记账并同步；20+24个主样本固定分母，失败保留not_started/partial。
模型/source最终hash、首错和cleanup后错分别保留；缺latent或label不填0，不丢episode，不替换目标。
禁止阶段A导入环境或任何physics包；numpy/h5读取和saved误差计算可CPU，神经仅MAIN CUDA。

## 8. 阶段B：paired-render因果，而非直接把旧source改with_target

阶段A不改变H5图片。不能用“用白色盖掉green”冒充native无overlay配对：marker和物体／agent遮挡重叠，
去色或inpaint会改物体信息。也不能把with_target=True直接打开到旧constructive source追认新结果。

NEW paired-render需要先冻结：

1. 四固定episode各0/5/10/15/20的可认证native场景几何、goal_pose／goal_state、body姿态、renderer参数／版本／
   shape／颜色／viewport。H5只存state7可能足以提出几何重渲染，但**不足以认证同collector场景**；缺goal_pose
   来源或场景合同就标missing，不从预测误差或future terminal反推出有利goal marker。
2. 每个固定state产生两张native图：相同env实例／相同完整snapshot，一次overlay off、一次on；顺序按
   episode ordinal及step parity交错，事前固定。唯一干预是渲染marker开关，不能改物理goal、动作、agent或物体。
3. 核对renderer实际读的goal_pose与目标对象：原生_set_goal_state只改goal_state的风险必须解决在NEW桥中，
   不能默认二者同步。每次render前后snapshot逐bit一致，禁止space.step或动作推进发生在pair内部。
4. 先报告两个native版本与archive RGB的file/pixel/transform差异。如果没有任何版本再现archive，
   仍可研究**该新renderer**的overlay干预，但不能称H5→live差异已被单因素识别。

两套20帧均用同固定模型／public normalizer和同一F/Z/M矩阵，各自在其view下的真实nextlatent作标签。
必须报告overlay引起的target latent变化、persistence及F−Z/M误差差分，不能只看E_F更小：稳定marker可能
使表征更容易保持、改变标签坐标，而不改善动作动力学。两个视图的latent没有同一个固定标签，不可混用
off历史预测与on标签后把误差解释为真实动力学变化；cross-view误差如报须明确为表征分布差。

阶段B是另需批准的最小新实现／实际配对，不是本轮授权。建议native render-only桥精确预算如下：
每episode一次native reset（其内部2physics、2setter、1goalrender按原生路径计），然后每个固定state一次
setter、两次explicit render；四episode合计4reset、8内部physics、28setter、4goalrender、40explicit render、
0public action。这不是“零物理”：reset内physics也必须MAIN授权并hook计账。
若实际native setup有额外操作或无法符合此封存合同，prepare拒绝／另freeze新版cap，不静默漏记。
重渲染使用state仅为evaluator干预，不进入部署selector，亦不称从state7恢复完整动力学轨迹。

阶段B CUDA额外40encoder+40projector、48model.predict／predictor／action_encoder／pred_proj calls，
action_encoder与pred_proj各144 token samples；0AD、0优化。输出cap64MiB（含partial/final1MiB），
prepare≤256KiB、disk floor200MiB、allocated3GiB/reserved4GiB、每独立render/CUDA阶段wall600s。
A+B合计60encoder/projector、72predict及各216 action/pred_proj tokens；总输出cap96MiB，成本分阶段列出，
不将A的20个archive latent复用充当B新渲染的40latent。

在缺少认证场景材料时，阶段A仍可按明确scope开展；阶段B暂停，不靠dummy goal或任意native默认goal补齐。
进一步从相同prefix真实执行action检验factual物理重现／闭环效果，需要另物理协议，不能由render-only桥代替。

## 9. 实现前应交付什么，以及与P1的关系

本文无可运行NEW CLI，不能对不存在的producer给出假命令。MAIN若批准后续实现，应先交NEW producer/tests/doc，
提供`collect_saved → prepare → pure verify → MAIN CUDA run → independent saved audit`，阶段B另显式授权。

内存测试至少覆盖：四episode规则不读outcome且无替换、episode跨界／缺step20、action shift不自适应、
两窗口label firewall、time-major/tile/float32、raw-zero≠normalized-zero、mask仅outgoing embedding、
teacher-forced W15不取W10预测、predict与pred_proj计账、no-dedup、budget/floor/partial拒绝、
first error不被cleanup覆盖、模型/source漂移、paired render快照不变与goal_pose错绑、未具场景资料不得假配对。
独立saved auditor应不用producer.verify、不导入环境或模型，核20/40帧SHA、NPY entry及24/48调用、固定全分母。

本地归档fidelity可以检查冻结系统的输入合同，paired render可以界定marker干预在新场景上的影响；
两者均不提供严格非贪心、共同真实能力改变、强资源对照或新root独立确认。
不能把H5可能含训练episode的误差当泛化，也不能把witness构造成功归给冻结模型；P1门槛与旧seal保持不变。

## 依据（本轮只读文本）

- [action stats来源审计](/home/gbk/code/LeWorldModel/le-wm-main/innerdocs/p1_constructive_action_stats_provenance_audit_20261027_v1.md)：public来源与训练run unknown、单位和normalizer边界。
- [render training合同审计](/home/gbk/code/LeWorldModel/le-wm-main/innerdocs/p1_constructive_render_training_contract_audit_20261027_v1.md)：H5固定帧green marker与checkpoint lineage缺口。
- [native backend](/home/gbk/code/LeWorldModel/le-wm-main/experiments/spectral_reachability/p1_conditional_direct_screen_20261027_v1.py:284)：states3/between-actions2及outgoing送入model.predict。
- [安装native LeWM](/home/gbk/code/LeWorldModel/le-wm-main/.venv/lib/python3.12/site-packages/stable_worldmodel/wm/lewm/lewm.py:44)：predictor和pred_proj内含于predict。
- [当前train.py](/home/gbk/code/LeWorldModel/le-wm-main/train.py:18)：teacher target与predict路径；当前源码不充当发布权重历史训练证明。

本文只有设计证据；未读取所选四episode的RGB／动作／outcomes，未生成任何实验结果。
