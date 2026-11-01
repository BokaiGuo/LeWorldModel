# 新独立 panel 的整数配额分离：候选预注册协议

2026-10-27。状态：待 MAIN 批准的设计，不是执行授权。仅新增本文；不修改旧 scheduler、prereg、comparison 或结果。P1/P3=false。采用 experiment-design 的机制对照与前景冻结原则，不采用训练、调权或看 outcome 选方案。

## 1. 本次要补的不是旧结果，而是可识别的干预

原两根的 p=w0/(w0+w1)=0.5094744711674455；原 D/S 在 (.45,.55) 同为10/10。仅换成单候选粒度仍30/30；无偏随机取整也不能保证不同。这些解释详见已冻结 `moving_budget_scheduler_discretization_design_20261027_v1.md`。本文不是其隐式修改。

当前 RGB 已由 MAIN 告知实际完成，另有消费者负责；本文不读取 RGB、物理 outcome、候选 cost、winner 或未评分 bank tail 来设计分配。未来实际结果不得回流以下规则。原两根不进入新独立队列，不重新放大其权重获得“通过”。

## 2. 推荐保留原算法，用严格前置判据保证配额分离

保留原信号、20个三候选 block、每根5..15 block及全部候选/重放规则：

```text
bD0=floor(5+10*p+.5);       bD1=20-bD0
bS0=floor(5+10*(1-p)+.5);   bS1=20-bS0
bU=bN=[10,10]
```

数学上的精确干预判据是 D、S、U 三个向量两两不同。w非负且sumw>0时，对原 half-up 规则等价于 **p<.45 或 p>.55**。等号不可算通过：p=.45/.55有一信号臂仍与uniform相同。

建议新队列预声明更保守的 **G = (p<=.44 or p>=.56)**，边界距原碰撞区至少.01。此.01是本提案的工程分离裕量，不是已获验证的最优阈值、成功阈值或对原协议的修订；审批时一次确定后不得扫阈值。须同时检查实际整数向量两两不同；先以严格绑定数组重算w，再判G，拒绝非有限或负w。sumw=0登记不可分离，不除epsilon。所有舍入实现和float64输入字节先封存，边界失败不“容差放过”。

该G保证至少：D有一根<=9且另一根>=11 blocks，S方向相反，U/N仍10/10；D−U至少转移1 block=3个fullH2候选=6predict，D−S至少2 blocks。靠近half-up tie时D/S未必精确互补，因此仍各自按原公式计算，不偷换成强制互补算法。G不保证winner、commands、物理结果不同，更不保证D优于S/U。U与N搜索内容故意相同，用于披露谱开销，不能要求四臂向量全部不同。

## 3. 新独立 panel 的两种研究问题，不能混报

### A：总体前景队列（推荐作为不筛选的主问题）

先固定8个全新two-root panels共16根，保留所有panel。来源生成规则、完整seed/placement请求、goal规则、配对顺序、候选bank与anchor算法、模型/normalizer/Q/表示在新模型评分与真实结果前封存。不得按w重新配对，也不得用旧两根或相同根的新随机seed冒充独立。

各panel记录G，但不因G失败替换；全部按原规则运行。这回答总体上的调度价值；**不能保证8个panel都有不同配额**。若全部G=false，则无有效配额干预，预声明结论是该队列无法识别不同分配的效果，而不是谱理论负。不得为避免空结果临时切到B。

### B：固定分母的信号分离队列（若目标要求确实不同配额）

预先固定最多16个全新two-root panels共32根的有序清单，不按难度或信号排序、不交叉重配根。先仅运行所有panel的blind nominal/J/SVD；遮蔽物理/终点latent/候选评分。稳定按预封顺序取**最先8个G=true且完整通过输入验证的panel**，若不足8则用实际数量，绝不扩池/补根/降低G。数值或source失败计入16分母，禁止重试择优。

在任何候选评分和后续物理前，发布16项筛选账与选取manifest、所有原始信号数组/整数配额/失败原因及外部SHA，然后对固定入选panel运行全部U/D/S/N。若0项入选，停止搜索/物理，报告筛选不可行；不是P3失败。独立性还要求这32根及目标没有用于确定G、算法或参数，MAIN须认证来源；仅有新seed并不能证明此点。

这确实保证**每个入选panel**有D/U/S不同整数配额，且没有按真实outcome选择，但属于模型信号条件化的任务分布，不是原分布随机样本。它只能估计“G分离条件下”的效果，不能外推总体或把未入选任务从成本/筛选分母抹去。信号是被检验的方法量，因此G筛选仍有方法相关选择偏差；用S/U保留同任务对照，不能声称该偏差已消除。

MAIN应在看任何新信号前批准A或B之一为主问题；不能执行A后因碰撞改报B。B适合作为机制校准，A适合作为总体证据。现有P1/适用P2前提不足时两者都只开工程校准，不开科学gate。

## 4. 完整成本，不把筛选成本藏掉

原cached planning每panel：共享8predict/AE、2J、384方向、2SVD；四策略各128predict/AE，包括完整candidate与独立replay。因此A的8panel上限4160predict/AE、16J、3072方向、16SVD。

B所有16panel先付共享筛选：128predict/AE、32J、6144方向、32SVD；最多8项入选再付8*512=4096predict/AE，总上限 **4224predict/AE、32J、6144方向、32SVD、0solve**。筛选receipt需作为实际计算的可校验输入被新runner消费；若实现不能无额外模型调用复用该receipt，必须在执行前另封真实额外预算，不能宣称仍4224。不能为凑上限空跑，也不能把筛选AD等价成额外候选。

新source acquisition、history/goal encoding、GN/D2P anchor生成及其全部AD/solve、独立模型加载/限额与hash开销都另列实际账，不包含在上述cached上限；新任务上不能把anchors当免费旧资产。未入选panel的这些成本仍保留。各panel自己用16MiB/200MiBfloor约束产物；队列总磁盘/墙钟及后续物理/RGB额度必须另外批准，不从单panel预算推定获准。

共享信号不代表N实际读取它：N函数入口不可见w/J，D/S/U独立部署视图各136predict+完整谱轴，N128predict且无谱；raw共同执行总账与独立部署视图分别报告。不称equal-total-compute，沿用原forward/AD维度/SVD完整向量。

## 5. 最小可执行接口和预声明终点

```text
panel_manifest: panel_id, ordered stable root/task/blind identities,
                source/encoder/anchor/model/normalizer/Q SHA, bank rules/seeds
screen_receipt: input closure, nominal/J/SVD arrays + hashes,
                w[2], p, bD/bS/bU/bN, G, attempts/completed/costs/errors
selection_manifest: all screen receipts + external SHAs, mode=A|B,
                    fixed source order, selected_ids, rejected_ids/reasons
run: verified manifest -> fixed four-arm candidate prefixes/replays
     -> own frozen commands -> separately authorized physical/RGB
```

不存在上述新source/runner/manifest的实际SHA，本文不虚构CLI或prepare。实现必须保证screen与run相同blind context/goal/Q/chart，保存完整原精度数组且拒绝漂移，不复用旧根终点、不输入privileged state。所有选择只在模型结果的允许阶段完成；真实物理永远不参与screen。

固定主比较D−U，次比较D−S及D−N。报告全部根和panel的配额、候选索引、winner/commands碰撞、full10及native/wrapped/同Q真实结果；modelQ只能作机制指标。B同时报告16项筛入率及所有筛选资源；不把8个panel的16根当32个独立样本，推断单位为panel。失败保留，不能删共同失败后报完整配对平均。不同配额但同winner是有效零效应，不得事后改pool、排除anchor或再加候选。

不推荐“只要w不同就固定11/9”的强制rank调度作为原算法修复：虽可强制干预，却改变幅度响应，微小噪声也被放大；它若研究须独立命名/预注册，与上述保持原公式的B不能混报。也不推荐筛到满意物理结果为止、调gain使旧两根跨阈值，或选择随机取整seed来强制非均分。

## 6. 本次交付边界

仅提交新设计和纯标量算术检查；无新模型/物理、无筛选新根、无随机抽样、无CUDA、无结果回填。待MAIN批准研究问题/新独立来源/完整资源，才可实现新版本。旧协议与原两根结果保持冻结；当前科学P1/P3及独立确认仍false。
