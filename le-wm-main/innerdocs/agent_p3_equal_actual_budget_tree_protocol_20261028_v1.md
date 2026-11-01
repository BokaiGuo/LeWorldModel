# P3 下一步：真正等实际资源与有界完整树的待批准协议

本文仅为独占新协议文档，不是执行授权、新政策实现或科学准入。冻结 child compiler、signal v2、历史报告及 MAIN 文档不改；不读取或触碰 reserved roots，不运行神经、physics、网络。协议需 MAIN 审批后另建版本实施。当前 known-root 工程校准不能获得新鲜性或 C3 结论。

## 1. 当前结果能够和不能够说明什么

当前48方法槽固定：24 child×fixed/adaptive，其中23 observed、1 not reached。41可编译＝fixed23个H1×4＋adaptive18个H3×4；adaptive另5个H3×8因48>search cap24整节点拒绝；2方法未到达。不得只报告41成功槽或删掉5个拒绝。

付费 child signal 已实际完成，每方法每 observed child 为8predict、2J、384scalar VJP/grad/raw、2SVD、1schedule。当前 compiler 的207/450新增predict只是准备预算，必须等真实 terminal report＋严格consumer后才可称实际消耗。若全41按合同完成，局部signal＋compiler计账如下；不是全pipeline计账。

| scope | fixed | adaptive |
| --- | ---: | ---: |
| 实际已付signal predict | 184 | 184 |
| 预定compiler search predict（非已执行结论） | 184 | 432 |
| 预定selected raw5 replay | 23 | 18 |
| 预计局部合计predict | 391 | 634 |
| 已付J / scalar VJP | 46 / 8832 | 46 / 8832 |
| 已付signal SVD＋预定compiler SVD | 46＋23 | 46＋18 |
| 方法分母 / 预算拒绝 / 未到达 | 24 / 0 / 1 | 24 / 5 / 1 |

所以不能声称“等实际预算”“等总资源”“省算力”，也不能用共享上限24代替实际消耗。未知FLOPs、真实backward-engine数量、CUDA纳秒保留null，不能按predict/J/VJP的个数相加折算。P4历史RGB编码、物理采集和上游根信号另列；两根已知、8episode、24child不是24个独立root。jointXY+angle原生任务成功、native7 reward、latent Q、model replay equality必须分栏。

## 2. 等实际预算的可执行定义及不可同时满足的条件

预先固定比较单位为“同一已批准任务/observed起点上的fixed/adaptive两方法”，不是在结果出来后找资源碰巧相同的子集。每单位同时保存资源向量的attempted和completed：predict API/sample/context tokens、AE/predictor/pred_proj及shape、visual acquisition、J、scalar VJP/grad/raw hook、SVD、solve、compile、replay、metric、schedule，另记端到端wall、峰值显存和实际存储。inclusive nested wall不可相加。

三种声明不得混淆：

1. **equal-cap**：两方法同一预注册上限，实际可能不同；只能声明同上限。
2. **matched actual predict API/sample**：完整单位所有预先指定调用均返回，相关actual计数精确相同；其余J/VJP/SVD/encode/metric等仍逐项比较，不能升级为等总计算。
3. **matched actual resource vector**：预注册的全部计数维度attempted与completed均逐项相同，实际shape也相同；仍不代表FLOPs/wall相同，须独立实测。失败attempt不能退款或以completed数隐藏。

在“当前原policy、固定4/8候选、H1/H3、整节点原子队列、禁止无用padding、固定分母、禁止挑子集”同时成立时，**给同样cap不保证能够用满同样实际预算，更不保证全资源向量匹配**。离散节点成本、依赖信号决定的后续horizon/type、不同终止深度、验证长度均可造成不可填充余量。因此当前协议不能仅靠重命名预算闭合该条件。

当前不批准新增策略。下一阶段若MAIN要求真正等实际predict，须在运行前单独批准并封存“共同有用工作单元/预算分配wrapper”版本，明确4/8原队列如何保持而通过不同数量的完整节点分配成本，以及是否与原策略构成新的比较。不能暗改当前core.choose、换阈值、缩候选、随机补调用、复用模型缓存减少付费、事后把391补到634。wrapper若不能在原完整队列及停止规则下做到精确匹配，应预注册为equal-cap实验，实际不匹配时如实判matched-actual未满足；不得过滤不匹配任务后宣称整体匹配。

MAIN可先批准的具体最小交付是**不运行模型的预算可行性compiler**：输入封存policy、ledger维度、最大depth/queue、验证预留与分母；输出每合法节点成本集合和停止状态。对尚未知child决策只给可达集合/最坏上界，不借root J猜决策。若不存在保证精确匹配的有用调度，输出不可保证及离散余量，停在审批门；不制造新policy或有益性结论。

## 3. 从单局部extension到真正有界完整树

现接口只有一次extension，不能把requested_depth=3写成executed_depth=3。下一版必须实现现有runtime的**完整有界beam树执行**：max_depth=3、base_beam=4、adaptive expansion_factor=2、max_expansions=144、原chi=.5/gain=.05/SVD1e-6；这些是封存工程规则，不是本文件新选阈值。完整有界beam树不是穷举所有组合，pruned节点须明确记为pruned，不可声称全组合穷举。

从24个原observed起点分别设局部depth0，按episode/boundary/method固定顺序启动；每个后续模型node具备不可变node_id＝原child＋mode＋候选index链，保存parent/edge/context SHA及完整动作前缀。depth按extension层数递增，不按H包数或已执行物理步数递增。模型预测child不是新的真实RGB观察；与下一次真实物理observed child必须不同类型、不互换。

每个实际扩展parent：

1. 从该parent的完整latent3/action2和原task binding独立付H1/H3信号，不借根/兄弟J；每方法独立或same-parent公平共享必须运行前固定，不能事后选更便宜模式。shared时只在完整context/task/model/normalizer SHA相同且合同批准时可行，必须实际只付一次并以精确有理数分账；当前independent规则不自动改为shared。
2. 用原policy计算decision；depth/候选/原队列先durable封存，再检查整节点队列和预算。原max_expansions按候选扩展数计，绝不是predict预算；另加向量predispatch guard。signal已付款后才发现queue超限，signal保留，不退款。
3. 对全部候选进行合法生成、完整native IO封存与独立验证，形成parent→child edge；所有候选完成后按原 `(Q, depth, candidate index chain)` 稳定排序，保留原beam宽度。Q仅为预声明模型选择，不允许原生reward/物理success参与剪枝。其余完整记录为beam_pruned，未知或failed不能当低Q。
4. 下一层仅对该原beam frontier重新付本地信号；达到原depth_limit、max_depth或原expansion cap时给完整停止原因。不得把停止时的partial winner升级为完整model_plan。

**reconfiguration是硬前置**：当前frozen generator明确缺目标/方向/尺度/生成与组合规则。MAIN须先批准其数学目标、仅past/当前信号输入、原始raw映射、候选顺序、真实付费导数及验证标准，另版本实现＋RAM负向测试。不得把ridge progress、随机动作、旧spectrum结论或终端物理结果当替代。缺规则时保完整拒绝，完整树科学门不通过，不能通过只扩progress子树宣称完整P3。

## 4. 可预先推导的有限上界，不代表授权支出

在不改变原beam下：fixed最多扩展parent `1+4+4=9`、生成候选 `4+16+16=36`；adaptive parent最多 `1+8+8=17`、候选最多 `8+64+64=136`，小于144。少数parent因原depth/continue条件停止时无需虚构调用，实际scope照记。以上每个观察起点/方法独立计算，不能将48重复视为独立root证据。

每parent信号严格8predict/2J/384VJP。若下一版保留当前“每候选whole＋segmented”验证，候选额外 `2H` predict；最终从初始observed起点独立whole＋segmented整条selected composition，若最多3层且每层H≤3，则最多再18predict；独立selected raw5再1。不要把已有局部双replay误算为最终完整composition replay。

由此一个保守**几何上界**：fixed `9*8+2*36*1+2*3+1=151predict`，adaptive `17*8+2*136*3+2*9+1=971predict`。对应signal最多fixed18J/3456VJP、adaptive34J/6528VJP；generator额外费用及reconfiguration如需新模型调用必须另加，不能偷塞进这个上界。本协议没有批准151/971为新预算，也没有允许H3×8绕过当前cap24；当前局部cap不变时该节点仍拒绝。完整树如要接受该节点，必须MAIN事前批准另版本全树预算合同，绝不可在当前运行中放宽。

两个上界差异直接证明“相同树深/expansion上限”不等于同actual资源。任何新equal-cap数值、资源向量及wall/output预算必须由MAIN在新实现前批准，不根据本批Q/原生成功率倒选。

## 5. predipatch、最终验证与严格saved消费者验收

每实际paid入口先reserve/check向量与独占输出/真实空间/时间/显存，再落attempt，再调用，落return；失败attempt不退款，无retry补齐。开始树前预留最长合法selected composition的完整双replay及raw5 replay费用；reserved只是上界，结束时只按真实H支付，不做padding。不能让search耗光后跳过verification，也不能把未用reserve当实际计算。

正常停止、budget拒绝、unsupported、工程失败、beam pruning、native早停（若以后另行物理执行）分别编码。运行时任意工程失败则该单位无release，partial arrays/ledger/队列/分母仍封存。未知结果null；未dispatch调用计账为未dispatch，不混科学数值0。科学成功分析保全单位分母，并把缺失界限和配对可用性单列；不把早停padding为执行满步。

真正完整composition需要两个从同observed起点独立执行的整条raw路径（whole与按原edge切分segmented），逐包边界及每条edge端点exact；还须独立重放最终selected raw5并核其首边界。所有J只证明其own parent响应，不自动成为整条composition导数；若需要复合J另行定义、付费、验证，不能把local J拼接成已验证chain derivative。

saved consumer应完整重走原producer.consume→prepare闭包，检查24/48分母、nodes/frontier/候选完整性、每parent新信号及source/context精确身份、全部剪枝和稳定rank、原额度/原停止、每次实际ledger与native IO/数组/整条路径、actual资源匹配判定、output与whole-job wall。whole-job wall需从输入验证前到最终receipt完成另实测，不能借当前final-write-excluded wall声称端到端。nested wall/resource不可相加。

RAM负向测试必须包括：root/兄弟/同值异ancestry J替代；预测child冒充observed RGB；丢5拒绝/未到达；depth与H混淆；仅保有利子树；部分候选胜者；错误beam/tie顺序；跨edge raw/context拼接；漏final whole/segmented/prefix；共享信号双计或免计；sample/API混用；actual不同却标equal；buffer/CRC/hash链篡改；半途cap/磁盘/墙钟失败不退款。MAIN审读后仅HOST CUDA执行，独立saved回算和物理后果是后续门，tests绝非证据。

## 6. 下一步交付顺序与当前未授权项

先交上述预算可行性入口与RAM tests（只用原known-root及合成合同，不读reserved roots）；MAIN决定是否批准新的matched-actual wrapper与reconfiguration定义。随后另版全树adapter＋完整独立消费者＋失败保存/资源上界；先RAM全边界回归，再MAIN审查、实际HOST CUDA、有界strict消费。若预算精确不匹配，如实保留equal-cap而非matched-actual；不能为科学口径强造匹配。

最后由MAIN决定是否另批真实闭环物理合同、原生指标配对、fresh cohort/P1及适用P2/P3验证。本文不授权任何新增执行或接触reserved roots。所有scientific gates继续false；当前391对634的局部比较不成立等算力结论，即使Q下降、重放一致或代码测试全部通过也不改变这一点。
