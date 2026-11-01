# P2 三臂共同资源上限：前瞻、结果盲规格 v1

状态：供 MAIN 审定后实现的新运行规格，**不是现有结果重新贴标，不是已可运行的新 CLI**。本次仅读源码/合同，未打开新 GRASP/root 结果。P4 历史查找已结束，不再扫描。本文件不运行模型/物理、不改冻结源码；P1/P2 scientific admission=false、fresh=false、equal_total_compute=false、equal_FLOPs=false、total_FLOPs=null。

## 1. 已有入口的真实差别

以下文件均在 `experiments/spectral_reachability/`；SHA 为本次只读核验值。

| 文件 | SHA256 | 关键接点 |
|---|---|---|
| `agent_p2_wallcapped_cuda_20261029_v1.py` | `5e733e75f56b3dd2f5598d6e9c4ae7878c1e3dd0d613c5b4777c1ecd891b9776` | `drive`、`Control`、`HostBackend.meter_event`、`make_bank`、`consume` |
| `agent_p2_compute_matched_protocol_20261029_v2.py` | `f17101d1b4b1d8a2c32bcd0218bda9708a2c992c1747a5cfe863e89fb0fff2ee` | `caps/schedule/unit_cost`、`Ledger` |
| `agent_p2_history_joint30_grasp_cuda_20261029_v1.py` | `6c44630f75259ce107b5b001a8dfc7f50b60c4f7b6ab920371e2994cde1ca9a1` | `phases/budget/optimize`、`Native.step/soft/serial`、`validate_root` |
| `agent_p2_three_program_20261028_v1.py` | `3111312477ff3cd9a764dc7d063861c1becaaf58885dc0fb25c2f3ec73d15fd5` | `HostBackend.jacobian`、`optimize`、`contract` |
| `agent_p2_compute_matched_protocol_20261029_v1.py` | `5e49fddb6783b1349a9963cb03d0fb25540de57c0d6b105a3f3c1759d1168e40` | `local_joint` |

wallcapped 是每根每臂 450s 搜索窗口、1902 predict 上限，外层2400s安全界限；自然完成 joint1902、composition1062。GRASP v1 固定750 joint+150 sync，`3*(750+150)+6=2706` predict、900 scalar backward；3600s是包含初始验证/加载等的整阶段安全界限，**不是同口径450s搜索窗口**。因此现有2706 vs1902既非同调用预算，也非同wall上限/范围；不能称等预算，更不能称等FLOPs。

## 2. 固定最小设计：共同上限，不强造相同消耗

只取原 prepare 中 root_slot0/1、condition=off；三臂为 joint、composition、history-aware joint30 GRASP **variant**，共6槽。原 task/source/warm_raw10、root_context三状态/两动作历史、goal192、identity-Q mean192、normalizer、模型/源码闭包逐位/SHA join；不读取 native labels、物理难度或后验结果决定根/配置。任何身份不一致整次比较拒绝，不借旧执行换绑。

共同 raw15（3×raw5）与 joint30 logit L2球 `.75*sqrt(15)`；保留 float64 NumPy tanh→float32 raw chart及各自原导数语义。相同搜索预算：**每根每臂 predict_API_attempt≤1902 且 predictor_forward_samples_attempt≤1902；每次batch=1、context_length=3**。二者分别计数，不相加；本固定ABI数值相同不代表一般batch调用等价。每个predict关联AE/predictor/pred_proj各一次，pred_proj处理3行，不能把三个模块嵌套调用再加成额外predict。

共同搜索wall≤450s（同步CUDA后的monotonic；cooperative、非kernel抢占），显存allocated≤4GiB/reserved≤6GiB。包含本臂nominal、建bank、AD primal、优化、CPU求解、数组捕获、搜索journal/fsync、导出双replay；加载/完整saved输入验证/最终strict/hash另列实际耗时，不纳入搜索窗口。新六槽外层安全wall固定3600s，输出总cap1536MiB、final reserve96MiB、disk floor512MiB，仅为工程容限，不是额外搜索预算。MAIN独占GPU串行运行；固定顺序 root0:joint→composition→GRASP，root1:GRASP→composition→joint；不按结果重排，不用运行时间作FLOPs换算。

joint/composition保留原schedule、42 seed、原4 alpha、稳定tie及完整unit的严格改善incumbent，不为耗尽1902补候选/重放。GRASP保留原学习率/噪声/seed/同步间隔/优化器/stopgrad/投影/最后动作选择；仅把自然joint步数由共同cap**算出**：

```text
N = max{n <= 750 : 3*(n + 10*floor(n/50)) + 6 <= 1902} = 532
joint=532; sync=100; scalar_backward=632; final_replay=6
predict=3*(532+100)+6=1902; next n=533 requires1905 (拒绝)
```

这是新532步预算variant，不追认750步v1、也不称官方GRASP复现。同步仍每50 joint执行10步，至500次后共10轮；不得根据损失决定停止/改同步频率/挑best iterate。若wall先触发，只允许导出截止前最后完整优化unit已提交的动作快照；未完成unit的动作更新不得泄漏到导出。初始化零动作可作为未完成首unit时的已定义快照，仍必须付费完整replay才能发布，不虚构已经预测过。

## 3. 每根每臂账本（自然完成值；超时只能更少或partial）

| 资源单位 | joint | composition | 新GRASP532 | 共同允许上限/解释 |
|---|---:|---:|---:|---|
| predict API attempt；batch1 forward samples（各一列） | 1902 | 1062 | 1902 | 各1902，包含AD primal与全部replay |
| action_encoder / predictor / pred_proj 模块调用（各自） | 1902 | 1062 | 1902 | 每项1902，不与API相加 |
| predictor context tokens；pred_proj rows（各自） | 5706 | 3186 | 5706 | 每项5706，不是独立forward样本 |
| 显式完整 Jacobian 请求 | 85 | 85 | 0 | 85；非scalar backward |
| mathematical VJP requested output rows | 16320 | 16320 | 0 | 16320=85×192；非engine实际backward次数 |
| requested J elements | 489600 | 247680 | 0 | 489600；保存shape，非工作量通兑币 |
| AD primal packet steps（已含在predict中） | 255 | 129 | 1896 | 1896；GRASP为632×3带梯度forward，非J请求 |
| 用户显式 scalar loss.backward | 0 | 0 | 632 | 632；不把J内部AD记为零工作 |
| Adam step / virtual-noise生成 / joint optimizer reset | 0 / 0 / 0 | 0 / 0 / 0 | 632 / 532 / 10 | 各自上限632 / 532 / 10 |
| root SVD / ridge solve | 2 / 1 | 2 / 1 | 0 / 0 | 各自2 / 1；CPU实测时间另列 |
| 新encoder / projector / physics / training | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 均禁止；不抹去继承的编码费用 |
| engine backward次数、独立JVP次数、FLOPs | null | null | null | 未有实际instrumentation不得补0/推算 |

joint：`3 nominal + 3 bank AD + 42*3 scan + 84*21 local + 6 export=1902`。
composition：`3+3+126+42*(15+7)+6=1062`；J形状为一次192×30、42次192×20、42次192×10，故elements247680，AD primal `3+42*(2+1)=129`。
GRASP：1896个优化forward+6导出，无额外nominal/bank。scalar反传所产生向量梯度（含30动作及virtual384坐标）不是632次完整J，也不是632×192个独立VJP请求。没有显式JVP不等于已认证内部JVP=0。

所有共同上限组成**资源向量盒**；未使用导数额度不可兑换predict、wall或其它轴。上述不同算法AD并不要求同数；比较名称只能是“同predict API/sample上限与同搜索wall上限下的三臂响应”，不能称实际消耗完全相等。六槽predict最高11412，两根自然总数9732；两者不是已执行数。

继承原paid编码必须引用同一cohort来源一次：原three_program合同声明原四task共享encoder16/projector16；本次仅取其中两off task，若无逐task归因则分摊=null，不能每臂重复加16、也不能重复退款。source验证/加载、strict复算、日志与模型hash各自记录开销及重叠范围；新神经调用为0不代表这些免费。

## 4. admission、partial 与完整消费

每次unit先检查完整unit envelope并预留最终6 predict/sample及对应模块额度；每个真实callback前再检查并**先持久化attempt扣账**，返回后才记completed。GRASP unit为一次完整joint或sync更新，均3forward+1scalar backward+1Adam，joint另1noise；既有joint/composition沿用 `unit_cost`。失败不退款、无padding/额外replay/重试。wall到期不得派发新操作，已进入kernel可能overrun，记录实际时间及违反上限；不偷偷追加导出尾段。资源预留不是wall时间保证。

导出whole15和从原根独立三段重放共6predict，全部三边context/raw完全一致；未完成/越界/缺replay保留None及真实prefix，不补步/补零或换旧endpoint。固定6槽分母；completed/attempted/failed/pending分开，不由AE/module返回推断predict API已返回。部分sync/joint导致失败亦保留，不能用未完整Adam更新的参数生成“成功”计划。

strict必须从真实prepared task与完整源闭包重建计划/顺序/预算、全部native IO、loss/梯度保存链、Adam/投影/噪声及动作选择；保留原两类consumer的exact神经连线要求，不放宽原模型parity。新GRASP consumer验证新532步/允许partial的准确前缀，不能拿v1固定900/2706的验收换标签。完整源码、输入/输出NPY或blob、journal与模型before/after身份绑定，report/receipt SHA由MAIN实际给出，不预造。

后续物理比较须另审新6计划native handoff；可复用代码，不复用旧outcome。全部raw15、相同原prefix重建、cp5/10/15同动作时间的模型Q/真实编码Q/native XY+angle+success并栏；earlystop与工程missing均显式保留。native结果仅评估，不回传选配置；本文件不授权新物理或宣称机制/C2/P2通过。

## 5. 最小实施接点与当前可执行检查

**当前两套CLI均不能执行本规格。** wallcapped v1绑定固定四槽/旧协议；GRASP v1 `optimize`循环750、`Native.step`/`expected_census`/`validate_root`硬固定2706/900，且无单臂450s与scalar backward前拒绝接口。不能用不存在的 `--steps 532` / `--common-cap`，不能直接运行旧入口声称完成本规格。

下一最小实现是MAIN审定本文件后分配一个NEW三臂runner版本：复用 `drive`/原算法local接缝，接入统一scope的meter；将GRASP的固定phase上限与真实attempt hook/last-complete快照/partial消费作为显式新合同；新增三臂task/root SHA逐位join及上述六槽census。不改旧数学、旧sealed文件或旧结果。RAM先覆盖632/633次反传、1901/1902边界、533步拒绝、wall中断/不退款、partial Adam不发布、全六分母、batch2拒绝、源join篡改；然后MAIN prepare→verify→HOST CUDA→完整strict。新实现/命令尚不存在是确切工程缺项，不是缺新科学根的借口。

现有源码的轻量检查可直接执行（不是新comparison的prepare/run）：

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -c 'n=max(n for n in range(751) if 3*(n+10*(n//50))+6<=1902); assert (n,3*(n+10*(n//50))+6)==(532,1902); assert 3+3+126+84*21+6==1902; assert 3+3+126+42*22+6==1062; print("budget arithmetic verified; no model/physics imported")'
```

本规格仅用experiment-design技能的事前设计/预算分栏原则，不采用其泛用调参、扩数据或训练建议。预算算术/文档核验不是优化效果、真实成本或科学证据；单一已知两根校准不能作独立样本总体推断。
