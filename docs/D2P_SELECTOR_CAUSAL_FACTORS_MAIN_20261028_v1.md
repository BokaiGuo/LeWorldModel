# P1 排因：先区分实现、近似与科学命题（2026-10-28）

2026-10-29 新终态：固定80对的原AST/detached几何分析及完整verify发现24对浮点
shape变化在整数primitive层完全消失，且对应同RGB；其余56对RGB变化，无unknown。
仍未认证完整contact/debug callback流，不称全部P1失配唯一原因。全40跨尺度比较中
模型方向cosine最小0.999999653，真实编码小h21零/大h3零，而物理pose40均非零。
另实际HOST CUDA固定48次动作辨识及strict/MAIN独立算术通过，原12 factual全3token
逐位复现，archive/nativeoff/nativeon中factual更近分别10/12、11/12、10/12；
不是“模型忽略动作”的支持证据，也不代表物理counterfactual/控制排序已修复。
完整路径、SHA与并行P2/Hi实际运行见[本轮终态](D2P_LOCAL_SCALE_AND_DOWNSTREAM_MAIN_20261029_v1.md)。

固定normalizer双臂现已MAIN主机CUDA328预测与完整strict通过，原public全164
context/objective逐位复现；archive统计未充分修复：反转34→36、2修正/4新增，
24真实零响应保留。终态平均MSE略降不等于排序/物理收益。训练血统仍未认证，
不替换部署统计、不以此否定理论。完整SHA、八条件及独立5测试复算见
[实际双臂结果](D2P_FIXED_NORMALIZER_RESULT_MAIN_20261028_v1.md)。

最新更强的固定锚点替换反例：保留全80对的模型响应，只oracle替换同slot的
真实baseline后仍21对方向反转（原34中19持续，另2新增），24真实零响应原样
保留。这比“误差项相消”更直接排除了本面板上“只减共同偏移即可根治”的
解释；是回顾代数，不是部署策略或物理因果干预。新9测试通过，完整SHA与
Hi-LeWM已实测的layout转写错误分开见[本轮记录](D2P_ANCHOR_FACTORIAL_AND_HILEWM_REPAIR_MAIN_20261028_v1.md)。

## 最新完整164探针闭环（覆盖下方“物理启动/编码未执行”历史快照）

模型164预测、真实164分支、真实164终态主机CUDA编码及各自完整strict均完成。
MAIN独立分析80对：34反向，固定共同baseline分解45对异号相消、80对均非精确
常偏移。新增可观测性分栏又独立核物理报告：24对不可见、22同向、34反向；
24不可见均有pose变化、不是只有velocity改变；相同RGB编码不一致0。
可见且目标方向非零的56对中34反向，不能把另24对零响应计为支持。
模型FD/AD仍一致，故不是本批粗大梯度接线错误；也不能把其余失配归为理论失败。

新代码严格区分不可见/encoder不变/目标tie/同向/反向，拒绝相同RGB却非零
真实响应等矛盾输入；不更改封存文件、阈值、模型或planner。五文件联合43测试
通过，完整SHA与分条件结果见[164探针终态](D2P_RAW_COORDINATE_CAUSAL_RESULT_MAIN_20261028_v1.md)。
已修的新驱动接缝不冒充历史P1根因；尚需追训练数据/normalizer身份，渲染非单射
具体来源，以及动作响应失配与任务度量tradeoff各自的可干预贡献。

## 新增目标方向复算（旧固定FD，不与80对面板混算）

MAIN已独立重算旧四根/八上下文的32项scalar Q secant：8项模型/真实
编码方向严格相反，25项三类误差异号相消，一个条件的4项实际RGB响应为零。
恒等式残差最大6.25e-14，10测试通过；无新增神经/物理调用。新增分析器
捕获并修复“原observed只有两分支”的错误schema假设，真实原报告为16分支，
FD仅固定其中selected/zero；不是历史P1损害根因。详见
[scalar方向分解](D2P_SCALAR_FD_CAUSE_MAIN_20261028_v1.md)。
实际cp5/raw10逐坐标对照现已完成模型侧164次主机CUDA/完整strict及MAIN原数组
80对独立复算，四baseline exact，FD/AD cosine最低0.9999997906，暂未见本批
粗大梯度接线错误。物理driver独立29测试通过，164真实分支已主机启动；
真实终态编码未执行，不能把模型FD通过当作物理方向或P1通过。
详见[本批实际执行记录](D2P_RAW_COORDINATE_CAUSAL_RESULT_MAIN_20261028_v1.md)。

## 最新干预索引（下方保留历次证据）

真实cp5 history条件预测、240请求scalar FD、observed suffix重编译72/32、
held-old2历史漂移4预测均已主机CUDA/完整strict；重编译的新四条full10也已
完整真实执行/264PNG strict。具体来源、SHA、独立复算与边界汇于
[真实观察后缀干预](D2P_OBSERVED_SUFFIX_INTERVENTIONS_MAIN_20261028_v1.md)。

新增确认：old2重编码漂移对本固定预测的误差影响仅约1e−8，不能解释
约.01–.17的条件预测变化；scalar梯度四固定锚点未见粗大接线错误。
新物理driver确实捕获并修复跨schema JSON hash误用和cp5检查时点后置，
但二者是新接缝问题，不是已证明的历史80对排序反转根因。真实观察后
suffix优化使模型Q全降、jointXY三优一劣、angle全损害、原生成功0/4，
仅2known根；不能把反馈接线补齐等同完整机制成立。新终态RGB桥与MAIN完整
独立分解均已结束（分解SHA `55ee1bef2ef77b16fe2153a4d2eecfa494c909da3b82ef2c514bfd56eac24d22`）：
同observed cp5 history下root0两条件预测Q下降却实际编码Q上升；root1两条件
两层Q都下降但angle均损害。四对误差相消、无精确共同offset。两类断点仍在，
不能声称已找到唯一代码根因或理论已否定；新动作真实成功仍0/4，科学门不变。

## 本轮实际完成的定位

不是“所有因素都已排除”，也不是“理论不成立”。本轮新增工具不调用模型、
不读取物理 outcome，对原主机 CUDA mixed222 的全部 **888 候选**重算。
两个已见物理根 × off/on 是四条件，不是四个独立根。

结果：`le-wm-main/results/main_selector_gap_decomposition_20261028_v1_MAIN01.json`，
SHA256 `6724b8f9275b97d22c2811065ca74d55ccb9467870b86828d2ce753ed6e51655`。
工具与16项回归：

- `le-wm-main/experiments/spectral_reachability/main_selector_gap_decomposition_20261028_v1.py`
- `le-wm-main/tests/test_main_selector_gap_decomposition_20261028_v1.py`

仅使用原始封存的模型候选、goal、Q、Jacobian、参数与预测；没有读取实际
terminal RGB、state、成功或奖励。原 CUDA report 的逐字节 SHA 被固定。
这是独立代数核对，不冒充原全量 pipeline consume，也不是新 CUDA 实验。

## 精确分解，而不是统一偏移拟合

记旧目标 O=stage+L+ridge，完整非线性动作见证的终态代价为 T。
L 使用原 float64 ridge 解 u，实际动作使用 `tanh(float32(u))`。
将线性残差记为 r，参数舍入引起的预测变化记为 a，有限步模型非线性
余项记为 e；全部先乘原 Q，维数 d=192。

```
T = L + (2 r·a + ||a||²)/d + (2(r+a)·e + ||e||²)/d
O − T = stage + ridge − rounding_cost_change − nonlinear_cost_change
```

逐候选与逐选择对的恒等式均验算；交叉项保留符号，不能把平方项大小当作
独立因果贡献，更不能把相消项删去后当作可部署模型。
其中 e 还包含模型有限步/tanh曲率与浮点运算差异，不单独证明物理模型错误。

| root / condition | 旧 stage+ridge 选择 | 完整终态 Q 选择 | 新减旧预测终态 Q | 新减旧旧目标 O |
|---|---:|---:|---:|---:|
| 0 / off | 51 | 51 | 0 | 0 |
| 0 / on | 51 | 47 | −0.016992972 | +0.020801431 |
| 1 / off | 23 | 61 | −0.029397651 | +0.239528688 |
| 1 / on | 23 | 69 | −0.123287275 | +0.383670436 |

后三行 O 与 T 的排序差来自以下**带符号的评分差分**，不是物理因果干预：

| condition | stage 贡献 | ridge 贡献 | −舍入贡献 | −非线性贡献 |
|---|---:|---:|---:|---:|
| 0/on | +0.009285722 | 0 | −1.55e−9 | +0.028508683 |
| 1/off | +0.140371247 | 0 | −1.21e−10 | +0.128555092 |
| 1/on | +0.248218208 | ~0 | −1.51e−9 | +0.258739504 |

全部候选舍入代价变化绝对值最大约 **3.28e−9**，非线性代价变化最大
**0.356075133**。因此，在本批封存候选中，浮点参数舍入不能解释上述
0.02–0.51 量级的选择差异；局部导数正确也不能保证有限动作的线性评分准确。
这不授权修改 dtype、damping、Q、beta 或 gate。

四个完整终态优胜者的 prefix 都没有比 root 当前 Q 更差。各条件中严格
`predicted stage > root Q` 的候选数量为 77/105/118/119；即便只看其中
终态最好的候选，其预测终态 Q 仍比原选择差 **0.617956/0.503647/
0.902318/0.986562**。这说明：

1. “旧评分与完整动作终态目标不一致”是本批明确存在的实现设计近似。
2. 不能据此说“本批存在被评分误杀的有益非贪心程序”。目前库内没有
   预测证据支持这个更强说法；真实物理尚须新选择独立执行。
3. 即使新终态选择更好，也可能只是更好的短时优化，仍不是 C2/P1。

## 因素覆盖与因果证据需求

以下覆盖当前已知主要因素族，不声称数学上穷尽一切潜在因素。
“本批未发现”限定于已验证 cohort，不转授给旧 Reacher 或其它 checkpoint。

| 因素族 | 当前可核事实 | 尚需证据 / 修改边界 |
|---|---|---|
| 实验身份、来源、版本漂移 | 历史 prepare 因测试 SHA 改变被拒已保留；是工程失败 | 实现冻结后重新 prepare；不得改封存 receipt |
| CUDA、TF32、autocast、模型冻结 | 新164预测全主机CUDA、MATH/禁TF32及autocast、参数和buffer未变 | 新神经实验继续只主机CUDA；无CPU fallback，不认证全局物理正确 |
| 动作单位/尺度/归一化次数/展平顺序 | raw10、float32 tanh、normalized history 已逐值核 | 训练 run→dataset→scaler 的身份未认证，不能称已排除 |
| 时钟、frameskip、动作与观测 shift | 已查 full3 outgoing convention、实际0/5/10时点 | 每个新驱动认证自身时序，不因旧驱动通过而豁免 |
| 渲染、goal overlay、颜色、预处理、CLS/patch | 已有 renderer×goal 与原 PNG 预处理核对 | 当前 RGB 合同不能代替训练视觉域溯源或任务充分性 |
| 起点、目标、RNG、接触、物理重放 | 已有 exact prefix/state/RGB 及 native dispatch 对照 | 新选择必须完整执行；不能拼接不同行动轨迹 |
| 隐藏速度、角速度、接触与可观测性 | 真三帧及动作 history 不等于隐藏状态可恢复 | 需要匹配输入的物理信息充分性/条件响应干预；state 不进 planner |
| AD、转置、符号、链式因子 | 新actual cp5原suffix的80对raw十坐标FD与AD一致，四baseline exact | 不推出物理方向正确；两h及实际float32分母分别报告 |
| 参数舍入与数值精度 | 本轮全部888项舍入影响至多3.28e−9 | 不可把有限步曲率误叫浮点 bug |
| 有限动作局部近似 | 本轮非线性代价差至0.356，明显影响排序 | 新算法按完整动作 terminal witness 评分，不改旧算法定义 |
| 初始线性化点与缺下降保护 | incumbent/zero × guard 的真实2×2已完成 | 不能把 anchor、guard、交互合并成唯一根因 |
| stage 惩罚与原科学选择目标 | 本轮3/4旧目标拒绝 terminal 更优动作 | 新 selector 与旧方法分栏；不用 beta 扫描挑收益 |
| 候选覆盖、suffix 质量、时域 | 222候选仍未生成预测有益且真正先退步的优胜者 | 新候选/更长时域/非线性优化要预定预算并单独验证 |
| imagined 前缀误差与反馈闭环 | actual-mid重优化四终态已真实执行：root0编码排序反转、root1编码改善但角度损害 | 同actual context仍须区分动作响应与任务目标，不归为统一前缀偏移 |
| 动作条件响应与共同偏移 | 原76可比对有70相消；新面板仍动作条件反转 | 不做按 outcome 拟合的 pairwise residual 校正 |
| latent目标与物理任务 | 实际 latent Q 改善仍可能 object/angle/velocity 变差 | 保留全部原生轴，不删除不利指标或改成功阈值 |
| native label、停止规则与consumer | 新164物理完整strict、10824PNG及全native7标签通过；四原动作cp10精确复现 | 新驱动修复≠历史控制根因；已知本批复现不转授其它环境 |
| 强基线、同反馈、预算记账 | 新真实RGB反馈gradient MPC已完整20步，1/2known根成功，费用逐轴分栏 | 不证明P1/C2；预测次数相同不等于总FLOPs相同 |
| 根难度、样本量、来源污染 | 本批仅2known roots，off/on相关 | 保留缺失历史reset authority；不自签全局fresh或按结果换根 |

## 修改准入

新 terminal-witness selector 是探索性算法分支，不是修复后重写历史方法。
先冻结所有候选与模型选择，再独立 native 执行及 host CUDA 终态编码；
逐对检查 predicted→actual latent→physical 三层，并报告所有固定条件。
不训练冻结模型、不拟合残差、不调 gate，不以测试通过代替实际实验。

并行推进的 feedback gradient MPC 是强基线补齐；它不能替代 P1 核心机制。
P1/P2/P3 科学准入继续 false，完整研究目标仍未完成。

## 新 selector 的 MAIN 实际执行记录

MAIN 完整审读 source/tests/doc 后，最终联合回归 **36 passed / 6.52秒**
（selector20 + 独立分解16；不累计历史测试）。新 selector 的 analyze 和
完整 deterministic verify 均已 exit0。产物：

`le-wm-main/results/agent_p1_terminal_witness_selector_20261028_v1_MAIN01.json`

SHA256 `a3a317b557da8bb6eb7a38d582f7df4d230b1b849cced00eca1add85403e63d9`。
单次 analyze 读取58,520个原数组 descriptor、105,191,296字节，完整核
原生模型数组/动作/J/时序，并独立重做 bank/ridge。verify 再付同一 saved
代数/I/O费用；没有任何新神经或物理调用，也不是“CPU降级跑实验”。

allmode 四条件原选择22/22/23/23，新选择2/2/23/23；mixedmode 原选择
51/51/23/23，新选择51/47/61/69。全部8个新winner无相对root的预测先退步。
两库重叠首42候选与相同任务分母完整保留，不能当作8独立根或1056独立样本。
下一真实执行固定 mixedmode 全4条件，不能根据后续物理收益在两库间挑选。

## 完整新物理执行及独立原生数值对照

新 physical 三件套 MAIN 完整审读，独立45 passed /16.64秒；prepare
SHA `b644dc1954f560448bca62a0131cee71319dc5f0af02c709672b91082af1f7a9`
与完整 verify 均通过。实际主机原生四条全10步结束，80 public、816 physics、
264 新PNG、errors=[]、8.667秒（物理执行窗口；前置saved核验另付）：

- report：`le-wm-main/results/agent_terminal_witness_physical_20261028_v1_MAIN01/report.json`，SHA `d02868f33f5b4b348c53d3684f30d2681124fc391b4f8d0f9b1a9e8992a04850`；
- 全264PNG/原state7/rawbits/事件/资源strict：`le-wm-main/results/agent_terminal_witness_physical_20261028_v1_consume_MAIN01.json`，SHA `0f644f89d5864f818805ce95b663d19d4a5d094b65e8f320ef30e3ef3491e8d4`；
- MAIN独立state7算术：`le-wm-main/results/main_terminal_witness_physical_contrast_20261028_v1_MAIN01.json`，SHA `375ba4e7edc5ba1005755a58c7712072534220c0baaa5d424fd139f121cf6558`，6项回归通过。

全部4条件保留，负数为改善；旧两条继承物理行保持其真实来源，不算新增执行。
新四条全新执行，不继承，index0相同task/raw10的终态state7精确相同。

| root/condition | 预测终态Q变化 | jointXY变化 | objectXY变化 | angle变化 | native7 distance变化 |
|---|---:|---:|---:|---:|---:|
| 0/off | 0 | 0 | 0 | 0 | 0 |
| 0/on | −0.016993 | −6.410426 | −10.503400 | +0.049616 | +16.665647 |
| 1/off | −0.029398 | +15.730537 | +8.241206 | +0.012990 | −11.761195 |
| 1/on | −0.123287 | +8.459656 | −0.840588 | +0.038834 | +22.967004 |

新旧均0/4原生成功、仅2known根。**换完整终态评分没有稳定修复真实任务**：
三条改变动作的条件角度均损害，jointXY一优两劣；不事后按objectXY选方案。
终态实际latent编码桥已审读，先捕获装饰器共用code的误拦截问题，修复为
`inspect.unwrap` 后独立35 passed /2.34秒；这属于新守卫接缝，不是历史
P1负结果根因。新四终态host CUDA与旧四终态的配对桥继续补齐，不把
预测Q直接当实际编码Q，也不把两层误差混成一个偏移。

## 真正带反馈的梯度 MPC：主机 CUDA 与完整 strict 已完成

MAIN完整审读新 source/tests/doc；首次独立测试1 failed/39 passed，定位
为原blob codec把运行时0维scalar变成保存的(1,)后，新验收器仍要求()。
新driver保留运行时 `loss.ndim==0`/fresh rawleaf/`scalar_outputs=1`守卫，
只按原codec的精确(1,)解码；raw和梯度仍严格10×2，不改封存codec。
修复后 MAIN 独立46 passed /37.77秒，三件套冻结后才prepare。

实际证据：

- prepare：`le-wm-main/results/agent_p3_feedback_gradient_mpc20_20261028_v1_prepare_MAIN01.json`，SHA `921785c816c74f957ebb6d453f8aa13dcfa24409a14f57b2daad041b0f86e719`；
- host CUDA/native report：`le-wm-main/results/agent_p3_feedback_gradient_mpc20_20261028_v1_MAIN01/report.json`，SHA `4f87f1d6edca9164b9a29c8692136f5d736de3123307f94af9974e8fe2ae2dc9`；
- 全量strict：`le-wm-main/results/agent_p3_feedback_gradient_mpc20_20261028_v1_consume_MAIN01.json`，SHA `09b3a3a5479a21e16959d94989f4da5fd36c97597e89d930d2591fbe8a22620d`，`complete=true`。

实际12节点sealed、4个native成功后not_reached节点保留；1632 predict、
384 scalar reverse、14 encoder/projector、100 public（warm40+control60）、
1016 physics、108真实RGB、errors=[]、132.189秒，模型前后完全一致。
完整原生IO/梯度/每节点递推/observed context/预处理/资源/dispatch均通过。

| 物理根 | 重复协议槽 | 原生结果 | 真实控制步 | actualQ20 |
|---|---|---|---:|---:|
| 0 | fixed/adaptive均跑同一GD | 两槽均未成功 | 各20 | 各0.423783669 |
| 1 | fixed/adaptive均跑同一GD | 两槽均在第10步原生成功早停 | 各10 | null，不能补至20 |

fixed/adaptive这里只是原参考槽标签，算法相同且重复轨迹精确相同：不能写成
四个独立样本、不能称2/4独立成功率。仅为 **1/2 known物理根**有成功。
也没有与原P3的132predict/节点强行声明等总预算；新GD为136predict+
32scalar reverse/节点，与原策略的导数费用不同，所有轴分别报告。

这条正证据反对“该冻结模型在这些任务上完全无法规划控制”的笼统解释，
推动继续比较候选库、局部ridge后缀与选择目标；不单独排除动作条件模型
误差/任务表示失配，更不证明非贪心重配置、程序组合或自适应预算成立。
P1/P2/P3 科学准入仍false；下一实现固定prefix并做非线性suffix优化对照，
预先固定更新/预算，不训练模型、不按物理outcome选动作。

## 新四终态 host CUDA 编码与完整 strict 已完成

桥三件套冻结后prepare SHA
`a498e7390dc5f652e93b4bde6d6adec684eace3f57f63e3b935f61825e2a257f`，
完整verify通过。主机CUDA实际4 encoder/4 projector，predict/AD/physics=0，
模型前后相同、errors=[]；report
`le-wm-main/results/agent_terminal_witness_terminal_rgb_20261028_v1_MAIN01/report.json`，
SHA `8d933d9d9bd83e5b68eb71161a7edf9f6f60662d9e97dc183afc0a5cdd458cab`。
完整strict含PNG→独立预处理→257hidden→CLS→projector→原NPY全链，回执
`le-wm-main/results/agent_terminal_witness_terminal_rgb_20261028_v1_consume_MAIN01.json`，
SHA `e44de69224bdcab89d6dd204099f8f725124db79043a0ae7370de221aef645dc`。

| root/condition | 预测终态Q | 实际编码Q | 预测endpoint相对实际latent MSE |
|---|---:|---:|---:|
| 0/off | 0.368376272 | 0.208154325 | 0.681083435 |
| 0/on | 0.403660004 | 0.380040963 | 0.374864769 |
| 1/off | 0.136669154 | 0.191906256 | 0.134908805 |
| 1/on | 0.118495955 | 0.158300919 | 0.056814954 |

这些是新选择的完整终态，不单独构成新旧动作的实际编码排序比较。
旧surrogate四终态尚缺新编码；已交独占companion补齐，其中两条实际
物理为原allmode whole-row继承，必须认证全部task/context/rawbits与真实
PNG来源，不能拿空trajectory当作终态缺失，也不能拼接。

MAIN六文件最终联合 **169 passed /56.31秒**，`git diff --check`通过。
源/tests/doc、旧失败和封存结果保留，不训练模型、不放宽科学门槛。
本轮真实物理/CUDA及明确实现接缝修复是实质进展，整套研究仍active。

## 旧四终态补齐后的新旧真实配对：本批断点在第二层

MAIN完整审读旧方案control RGB三件套并独立37测试通过。真实mixed
consume重建并核MAIN summary的**整个内部receipt**；旧后两行的完整
allmode继承行与原PNG目录被认证，不把wrapper空trajectory当成无图。
四张旧cp10真实图像主机CUDA编码完成：4encoder/4projector，预测/AD/
新物理均0，模型前后相同、errors=[]。prepare SHA
`9ed1bb0ccf762d7d95822d6184cd95e188297e1e82e6c5567ce27267bd5084bc`；
report `agent_terminal_witness_control_rgb_20261028_v1_MAIN01/report.json` SHA
`67904a04ae37c7bb3e9f1216c193b47cb7f93161dbffe23cf0ca9e3ecaa8e9b4`；
完整strict `agent_terminal_witness_control_rgb_20261028_v1_consume_MAIN01.json`
SHA `4f40159b636dc5252ce2c24a6be29f2374c7850b3723f81e8b062f745b8ebec8`。

MAIN新工具 `main_terminal_witness_rgb_contrast_20261028_v1.py` 独立读取原
gzip NPY并核文件/数组SHA、dtype/shape、全部四槽的task/goal/context/Q/
checkpoint10身份，分别绑定各自动作，重算预测与实际Q及原生轴。
其12项算术回归通过；结果同stem `_MAIN01.json` SHA
`b740fe3e086d8e49c6bb4b0c7dbd3016d1f5761194060bba8991384bb022c67d`。
这是saved原数组独立复算，不冒充另一次上游完整strict重建。

| root/condition | 预测Q变化 | 真实编码Q变化 | jointXY变化 | angle变化 |
|---|---:|---:|---:|---:|
| 0/off（相同动作） | 0 | 0 | 0 | 0 |
| 0/on | −0.016992972 | −0.062626698 | −6.410426 | +0.049616 |
| 1/off | −0.029397651 | −0.007045688 | +15.730537 | +0.012990 |
| 1/on | −0.123287275 | −0.033899857 | +8.459656 | +0.038834 |

**三条改变动作的配对均未发生预测→实际编码排序反转，但编码改善仍不
保证物理任务改善。** 相同动作的完整state7、预测latent、实际latent均
精确一致，构成本批重放对照；不能据此泛化旧80对或其它checkpoint。

另保留完整带符号分解。设旧预测/实际为p0/a0，新为p1/a1，目标g，
e0=p0−a0，da=a1−a0，de=(p1−p0)−da，d=192，则

```
ΔQpred − ΔQactual = [2 da·e0 + 2(a1−g)·de + 2 e0·de + ||de||²] / d
```

三条改变动作的response-error MSE分别1.286573280、0.153648457、
0.149179264，且带符号项全部相消，不能解释成一个统一平移，也不能
删除某项生成事后修正器。排序同向不等于预测端点准确，更不证明单一
根因或全局模型无误。当前至少有两项独立缺口：局部stage+ridge评分与
完整终态目标不一致；完整编码目标又与物理任务不充分一致。

正在固定全部42prefix×4条件，原ridge suffix保持iterate0不修改，固定
8步非线性suffix优化、全9iterate保旧选择，再逐级验证；只改优化器
不能预先宣称能修复上述表示缺口。MAIN还捕获一处可误拒原合法witness
的新协议风险：原float64合法suffix经原float32 cast后，81/168的
postcast norm略超数学半径（至5.45e−8），应按原solver+原cast/rawbits
认证，不能事后加严格postcast gate或悄悄重投影iterate0。新投影迭代
仍按声明的inward float32规则严格可行；该风险不作为历史负结果原因。
P1/P2/P3科学门仍关闭，未改变冻结模型、Q、成功阈值或独立根要求。

## 固定suffix对照首次启动的工程失败（不算负向实验）

MAIN联合5文件108项回归通过（9.69秒），suffix v1三件套冻结后prepare
SHA `00958938fda085e303c3e56525999dcceeb99eef797b93e760642e5ef1301446`。
主机CUDA实际加载模型后、第一条candidate之前，加载类源码闭包检查
拒绝：`ValueError('presealed loaded model classes')`。失败report
`agent_p1_nonlinear_suffix_library_20261028_v1_cuda_MAIN01/report.json` SHA
`bb155464bef333add0cb78b0390c78fd3e4495d9a709b61cc54a0833ce233e96`。
168行全not_started，候选attempted=0，前后模型hash完全相同。

MAIN另在主机CUDA只加载同一冻结模型并列出实际未绑定class路径，定位
为torch内置activation/batchnorm/container/conv/dropout/linear/normalization
与transformers.activations共8个源文件遗漏。原allmode闭包只覆盖自定义
模块，而新驱动要求所有loaded class源均封存，build与run合同不一致。
这是新驱动封存代码问题，不是suffix算法/模型预测失败，更不是理论负结果。
v1源码/prepare/失败结果保持，v2补齐真实模块source SHA并重建verify，
不跳过class检查、不改算法/预算；物理adapter等待真实完整v2而非partial。

v1 partial saved消费已完成，SHA
`412aca32657c6ea7918cc6a8e8c545a9e710408907cd843b1513994252a88819`，
仍168行not_started、plans=[]、complete=false，不伪装完整实验。
v2经MAIN逐diff审读，新增安装源解析不import torch/transformers；原所有
函数/类除build外AST一致，build仅合并八源hash与记录来源表。v2 source
SHA `0a3e475af9271a0769a3829b4e714fa4e2d6f00eaed38ec8c3ea5d7b9a163fbc`；
prepare SHA `3da0299d73a865621f60b7bbd804a0f757044455dad30e562986635c1093eb4d`。
实际host CUDA重跑已越过原阻断，已出现真实sealed候选，尚未完整报告，
不得按已启动计为完成。新物理adapter三件套MAIN审读并67回归通过，
v2/helper/control/MAIN算术五文件联合184 passed /34.03秒，diff-check通过。

同state双renderer目标集合设计亦完成MAIN审读，只接受其A（全1056
既有预测的独立目标loss选择对照）进入实现准备；不实施额外suffix B，
不借新loss替换原P1合同，不按物理outcome选目标。其文档末尾“本次用户
只要求文档”仅能解释为worker上一有界文档任务，不能描述用户实际要求；
用户总体目标仍是实际研究落地。新分支还须真实原目标来源join、代码
审读/封存、saved执行与必要实际验证，设计文档本身不算已修复任务。

## 固定42prefix非线性suffix：实际CUDA、全strict及新物理均完成

v2实际主机CUDA334.264秒，3376predict/1344scalar reverse，全168候选、
四个最终double replay均sealed，errors=[]，模型全部参数/buffer未变；
report SHA `f91c44779de94e4abb87f4250b0db9b12f9e8cbe4f980628c031cb78301882de`，
完整数组/梯度/时序/资源strict回执SHA
`337330eda5e8f191deec615422cae53e0c260364f63db0cd14d58218033ff019`。
两文件分别位于同stem的 `_cuda_MAIN01/report.json` 与 `_consume_MAIN01.json`。

MAIN独立 `main_nonlinear_suffix_effect_20261028_v1`（8回归）重建完整strict，
四槽每条42prefix均保留原ridge iterate0；全部168条至少一个新iterate
的模型Q更低。四条最终winner为prefix2/2/3/3、iterate均8；仍无相对
root模型Q先退步。结果SHA
`86651769dbd4357675f02b7b72f854bf8d18abd74a93d71ad7c2651e7db4b449`。

| root/condition | 原allmode Q | 只改终态选择Q | 非线性suffix后Q | 固定原winner前缀的suffix Q变化 |
|---|---:|---:|---:|---:|
| 0/off | 0.654757470 | 0.534658170 | 0.386614428 | −0.023175931 |
| 0/on | 0.599394859 | 0.547018710 | 0.374633970 | −0.029237902 |
| 1/off | 0.166066805 | 0.166066805 | 0.135648054 | −0.008407896 |
| 1/on | 0.241783231 | 0.241783231 | 0.174545877 | −0.036083641 |

这是局部ridge后缀及选择近似确实限制模型目标的证据，但仍不是物理收益。
不把同时改变suffix和selector的新最终winner差异唯一归因于优化器。

新native四槽全部fresh env/full10，无继承：80public/816physics/264PNG，
8.863秒、errors=[]，完整strict验全部rawbits/state7/goal/renderer/事件。
prepare SHA `f168189aee796f3979dde759ea7d8b5c6e2274874be89913e29d078530118e89`；
report SHA `30f5131d6f70abdbe669ce0aef86a430c1ba6593e40149c2402801b6d0c37c8a`；
strict SHA `2add0915ecd4d1fa16d72d77a44e1db5cc4d41b1f1961e1ad06e216fa76119d4`。
路径stem为 `agent_nonlinear_suffix_physical_20261028_v1`。

MAIN同task/goal/root/time的原state7独立复算（8项saved回归）结果
`main_nonlinear_suffix_physical_contrast_20261028_v1_MAIN01.json`，SHA
`25fda7b108ebbe9461b6c04934a43264d032d98c31d65c78fc44327cb5846a34`：

| root/condition | jointXY变化 | objectXY变化 | angle变化 | native7变化 |
|---|---:|---:|---:|---:|
| 0/off | +4.821254 | −11.695113 | +0.018623 | +13.083493 |
| 0/on | −11.539865 | −29.479968 | −0.098276 | +34.235036 |
| 1/off | +1.504355 | +3.401825 | −0.061446 | −203.220045 |
| 1/on | +13.421142 | +11.493414 | −0.071535 | −89.846686 |

关节XY一优三劣、角度三优一劣、原生成功仍0/4，已知2根而非4独立样本。
root0物体误差虽降至5.23/1.07，agent仍远离目标；不能改为object-only
成功标准。native7包括速度，不能把其下降替代原完整任务成功。
本轮更好的非线性模型优化并未稳定改善完整物理任务；尚须新终态RGB
编码判定本轮预测→实际编码排序与编码→物理两层，不能套用前一mixed
对照的无反转结论。新RGB companion MAIN59测试通过并完整审读，prepare
已提交；尚未有CUDA/strict终态，保持P1/P2/P3科学门false。

### 新suffix四终态RGB已实际CUDA并全strict

上述“尚未编码”现已完成：prepare SHA
`c24a8bbb03543108a30eb419ea4ee07c453d14cb2d4a753fa24ba8037b3f0ae2`；
实际4encoder/4projector，预测/AD/新物理0、模型未变、errors=[]，report
`agent_nonlinear_suffix_terminal_rgb_20261028_v1_cuda_MAIN01/report.json` SHA
`78bb8cfcb051623e64017da0061dea4ad00ea90daaf89bfbf774b20a50727ba5`；
完整strict原PNG→独立preprocess→257hidden→CLS→projector/192和precision/
事件/guard回执SHA
`18d281427d34c1afce2054308208e7a9561df24a5d117cc851a5d143f17939e4`。

| root/condition | 预测Q | 实际编码Q | endpoint latent MSE |
|---|---:|---:|---:|
| 0/off | 0.386614428 | 0.092856397 | 0.449716220 |
| 0/on | 0.374633970 | 0.074589146 | 0.334035675 |
| 1/off | 0.135648054 | 0.153108141 | 0.162147192 |
| 1/on | 0.174545877 | 0.238967411 | 0.072497078 |

这还不是四条相对**原allmode**的完整编码排序：已补的旧control是mixed
51/51/23/23，原allmode为22/22/23/23。不能因为root相同拿前两条mixed
真实latent替代allmode，不能将旧mixed的无反转结论迁移到本新对照。
正在核查旧allmode精确raw10/task/time是否已有可认证编码，避免重复获取。
新增真实终态和表中物理损害均保留，不据低actualQ宣称P1通过。

## 双renderer目标集合A：全1056真实saved重评分与verify完成

MAIN完整审读A源/tests/doc并独立37测试通过（7.05秒），生产prepare完成：
SHA `1cf3f054b6620109dc0ed19e221471b4d2a95d0d014c5aa39d0a3a441646d4c9`。
实际analyze与从来源完整重建的verify均exit0；结果
`agent_renderer_goalset_selector_20261028_v1_MAIN01.json` SHA
`639fc9d15ea4ab02bddd8428f547acbc081d3c06cb324032fd45c651777508d1`；
完整selection（1056候选、8panel）摘要SHA
`88d1026e2b3341cd4378d65179c480ade850adac3d8f13b39a9d4ef1f0411224`。

同一完整物理goal的off/on/off与全部16原图像素/预处理/257hidden/CLS/
projector/latent链及当前slot context/goal/Q/task来源均核验。新loss取
两目标编码距离的min，原single-goal Q与候选不变；全程不读取动作物理
outcome做选择，无新神经/物理/训练/拟合。CPU仅原数组/来源算术，非
CPU神经降级。新算法身份与原P1合同严格分开。

allmode原terminal winner2/2/23/23，新仍2/2/23/23；mixed原51/47/61/69，
新仍51/47/61/69，八panel完整raw10选择**全部未变**。两panel的active
goal分支改变（allmode root1/on→off，mixed root1/off→on），模型集合
分数降低，但没有动作选择的first-stage效应；不重跑相同物理动作来
冒充新修复收益。该固定库上的renderer目标容忍本身不能解释或修复原
选择损害；不能据此断言其它目标表示/真正重优化B普遍不可能。本轮不
临时加第三目标或扩展B来挑结果。

MAIN最终八文件联合 **288 passed /50.70秒**，`git diff --check`通过。
没有删封存失败、覆盖旧结果、调整成功阈值或训练模型；完整研究仍active，
P1/P2/P3尚未科学放行。

## 原allmode精确历史编码配对：缺口已补齐

前文“原allmode编码尚须认证”现已完成。MAIN工具
`main_nonlinear_suffix_rgb_reuse_20261028_v1`重跑历史与新RGB的完整consumer，
核验原raw10、task/blind/source、step10完整snapshot/state、PNG像素及独立
preprocess、目标192坐标，认证历史image9/15/21/27确为原allmode
22/22/23/23，而不是mixed51/51的替代。同时间指同一执行步，不是同一
墙钟事件。没有新增神经或物理调用；不是fresh实验。

结果 `le-wm-main/results/main_nonlinear_suffix_rgb_reuse_20261028_v1_MAIN01.json`，
SHA `f2665a3c4f500f80f55da2f8b5519edf91fdb2203b3727bfcdc10b6c1281ed49`。

| root/condition | 预测Q变化 | 实际编码Q变化 | jointXY变化 | 预测排序反转 |
|---|---:|---:|---:|---|
| 0/off | −0.268143041 | −0.089802167 | +4.821254 | 否 |
| 0/on | −0.224760889 | −0.328156391 | −11.539865 | 否 |
| 1/off | −0.030418752 | −0.045843803 | +1.504355 | 否 |
| 1/on | −0.067237354 | +0.046766635 | +13.421142 | 是 |

四对均有异号误差项相消，动作响应误差MSE分别为
0.166972890/0.204762758/0.119058103/0.150145202；均不满足精确的
动作无关统一偏移。旧80对的70/76相消结论没有被拿来拟合偏移，亦不
能把本四对外推至新根。

本轮可区分的断点：①模型类源码绑定遗漏是已修复的工程失败；②原
stage+ridge选择和局部后缀近似确实限制模型目标，已以terminal选择和
有界非线性后缀作独立版本实现；③新优化仍有1/4预测→实际编码反转；
④另外2/4是编码Q改善而jointXY损害，不能仅修预测偏移；⑤同物理
目标双renderer集合在现有1056候选不改变任何最终动作，不能声称这项
修复有效。③④是当前协议/模型/表示链的实测失配，尚不是理论不可能
证明，也不能承诺继续找必然有代码bug。未改门槛、训练模型或删负结果。

包含新增历史复用工具的最终九文件联合回归：**292 passed /50.86秒**；
`git diff --check`通过。测试验证工程约束，不替代上述真实CUDA/物理证据。

2026-10-29后续：新三程序H3在模型Q两根均改善后，四条raw15实际物理已完整执行并通过strict，
MAIN另不导入原模块独立复算60次原生奖励/终止及raw位串。组合对jointXY的作用在两根
分别+16.094521/−22.658762，物体位置也一损一益，原生成功0/4。尚不能把这次模型优化
视为机制闭环；也不能把不一致直接归为理论失败。下一处测量是相同封存动作的cp5/10/15
模型Q→实际RGB编码Q→原生任务指标12槽对应，尚在实现而非已完成。详见本轮后续记录。

2026-10-29终态补记：上述12槽主机CUDA、完整strict及MAIN独立数值复算已经完成。
root0的cp5/15模型→实际编码排序反转，cp5另见编码→任务反向；root1cp5精确平局保留，
终态编码/XY改善但角度损害。native callback v2另完成164分支完整重放与strict：固定80对
callback均不同，24对在整数图元前量化消失且512/224相同，56对没有该消失；不能把24对
推广为全部排序反转原因。历史callback/cache512未封存，仍不认证历史唯一原因。
完整来源、SHA、固定分母和测量数值见
`docs/D2P_LOCAL_SCALE_AND_DOWNSTREAM_MAIN_20261029_v1.md`的终态两节；保留旧进度段作为历史记录。

2026-10-29精确归因补记：新JEPA H1/H2的80对封存分析，与上述native新重放做逐身份连接，
而非仅比较数量。实际分析 `agent_p5_jepa_callback_zero_sets_20261029_v1_MAIN01.json` SHA
`35b4c01e4ce7ad846af740878b5399ef17247425f22cba44ceb330a55755c3be`：80对无缺项，
真实JEPA特征零响应24对与整数图元相同/512相同/224相同三集合精确一致，差集均空。
因此本批24对零响应有可定位的渲染量化测量链解释；另外56对并非图像完全不变。
H1非零响应中31方向cosine正、25负；H2为34正、22负。总体预测MSE优于persistence
不等于动作方向可靠，更不能打开P1或证明理论不可能。原历史callback membership仍未知，
不倒签原运行的唯一因果。后续按非零响应幅值、时间和坐标分解，所有因素使用固定全分母，
不按有利结果选根、阈值或方法。

非零因素实测补记：`agent_p5_nonzero_factors_20261029_v1_MAIN01.json` SHA
`8b936253fa6fe0e9e23f10620a35438a44b1a0e1dfc869ee2c27e87c1ae4f489`已MAIN运行并整份重算。
80个扰动前H2_15输入内容/预测/实际图像负控全部相同；56非零对的真实响应相对float32
ULP envelope最小约1.095×10^5，不能简单用末位舍入解释。19对H1/H2均负向，6负转正、
3正转负、28持续正，24零对保持null；不是单纯延长horizon便统一修正。
该分析为同已见根的探索性分解，ULP不是噪声方差，H1/H2也非纯horizon干预。
下一辨识任务固定全部分支做视觉历史/proprio历史2×2模型内部输入干预；不选有利根/h，
不把未必可达的混合输入说成物理反事实，更不据此认证P1或理论失败。

2026-10-29全量干预补记：完整历史2×2已实际HOST CUDA执行328次新predict并通过
完整strict及独立164/80复算。最新槽后续另锁定actual5/10原字节，只交叉15-slot，
AP/PA/PP均新增原生预测，共492次；实际53.9939秒、0encoder/physics、模型未变、errors=[]。
其完整strict SHA `c636f8d9c6624317d4da4f3ddb127b4f20710f9a5c02c212872d0d70c9cab642`；
独立全164四cell与80对算术summary及整份verify通过，SHA
`ff5958ef8a23ebd6a76b4748659e43008ebd932b8f4c39dbb817bb766b1b728f`。
AA/AP/PA/PP在原56非零对的方向正/负为31/25、31/25、35/21、34/22；
24真实零响应对仍null。各cell预测响应零集合均空，不能因cosine同为24个null
就说预测零集合等于真实零集合。四个cell共同使用同一H1真实20标签。
最新槽与整历史统计相同，目前正在比较全部原生输入/完整三帧输出及实际模型mask源码，
尚不认定前两槽无作用、架构屏蔽或统一偏移；统计相同不是因果等价证明。
这些是已见根探索性干预，不能替代fresh确认、物理收益或P1科学准入。

数组级终态补记：跨whole/latest全量比较已产出，SHA
`bc7665c448a89fee4b89bd3bc150cd2016d7772035cb0a0588bd8014b47003e0`，整份重算接续。
164/164分支的原actual与rolled前两槽视觉、本体均逐字节相同，动作全三槽也逐字节相同。
所以AA/AP/PA/PP各164个native输入和两个完整三槽输出（视觉、本体）均逐字节相同，
四cell×164分类全为`identical_inputs_and_outputs_no_distinct_contrast`，无执行数值差异。
据此纠正实验解释：先前“整history交叉”在此来源上实际也只改变最新槽，
最新槽重跑没有增加独立干预差异，不将两轮合并扩大样本量或当成两次机制确认。
绑定实际loader与mask源码允许末槽关注5/10/15，不存在由本结果证明的历史架构屏蔽。
它解释统计相同的协议原因，但没有解释剩余21/22方向反转，更不是P1通过或理论否定。
