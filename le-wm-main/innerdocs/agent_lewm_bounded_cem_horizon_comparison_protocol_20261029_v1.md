# bounded CEM H2 / P2 H3：后续比较的时距与实际任务合同

本轮唯一新文件；不是runner、生产授权或科学通过。冻结source/results不改。
采用experiment-design skill仅组织事前控制和费用，不训练、不调参、不按效果选根/预算。
本协议是在旧实验之后提出的前瞻设计，不冒称结果盲预注册。本文未读取本次bounded效果值；
只核source、prepare输入数组、文件SHA。MAIN报告实际bounded完成，本轮只绑定实际来源。

## 1. 首要结论与固定后续范围

现有bounded H2不能与P2 H3直接排强弱。最小真正等时距/同实际任务的新增工作，固定为：
**两个原off根，各运行一次新的H3 bounded-CEM，使用P2原task goal、相同完整history，
部署原生raw15；与全部既有P2 joint/composition/GRASP六槽分栏。**
不裁H3的prefix10冒充H2优化，不给H2末尾补zero5，不挑一个可部署根或一个有利对手。
原bounded H2保持独立历史栏，不能将新H3证据倒灌为旧H2的认证。

这只消除预测/部署时距和实际任务错配，**仍不是等预算/等support/等FLOPs比较**。
下述H3新增设计继承已批准N300/I30/topk30/MAIN seed0，不因本次结果重新选配额。
它仅为待MAIN批准的接口协议；本轮不实施新runner，也不新增科学门。

## 2. 已核来源与可复用边界

下列所有相对路径的绝对根固定为 `/home/gbk/code/LeWorldModel/le-wm-main/`；
例如B-prepare的完整路径为
`/home/gbk/code/LeWorldModel/le-wm-main/results/agent_lewm_bounded_native_cem_20261029_v1_prepare_MAIN01.json`。
没有通配符/按最新文件选authority；未来输入须使用表中精确路径和SHA。

| ID | 固定相对路径 | 实际SHA256 |
|---|---|---|
| B-prepare | results/agent_lewm_bounded_native_cem_20261029_v1_prepare_MAIN01.json | c91bf542d99471ef107946b1156688d0a3950995c129ba230632773eed5b013e |
| B-report | results/agent_lewm_bounded_native_cem_20261029_v1_cuda_MAIN01/report.json | f23014b2ee8a742ec097ec42f52ba5d326b73ea16cb27499af3399f5d7d0231a |
| B-strict | results/agent_lewm_bounded_native_cem_20261029_v1_consume_MAIN01.json | f4ebc1653a9bb69b70c280dc97dd1096447c2f98d30e749a0ecc40b987564c33 |
| P-prepare | results/agent_p2_common_resource_grasp_cuda_20261029_v1_prepare_MAIN01.json | 7facba500439172ced280ec0b893659715139b1e5e22073ae7313665a1c0faec |
| P-report | results/agent_p2_common_resource_grasp_cuda_20261029_v1_cuda_MAIN01/report.json | d781b493ecd624db846e3225ed0177bd3fe250d6ba9213d925b2a57956dd2242 |
| P-strict | results/agent_p2_common_resource_grasp_cuda_20261029_v1_consume_MAIN01.json | d3ee465f2fa6d31345f369f9514165c4817c51a29c3d00ffdd2d681e5829cc1f |
| P-physical-prepare | results/agent_p2_common_resource_physical_20261029_v1_prepare_MAIN01.json | 4b4f19146c352e00e6007b88030738264157b781ce34636120fbf2ad44321742 |
| P-physical-report | results/agent_p2_common_resource_physical_20261029_v1_MAIN01/report.json | d23bcc56d52bce969a75faeff776e92a95ad37d58ff7bb9df2879eb624a29627 |
| P-physical-strict | results/agent_p2_common_resource_physical_20261029_v1_consume_MAIN01.json | ae5a62dbc158ac1a7da217a413987f5a88bafde357326559c2c384c016ada87c |
| P-RGB-prepare | results/agent_p2_common_resource_rgb_20261029_v1_prepare_MAIN01.json | ed7b4ac2de94a6558adb12af587a66532c4ab54f60e56fd064cab158460f3d81 |
| P-RGB-report | results/agent_p2_common_resource_rgb_20261029_v1_cuda_MAIN01/report.json | 50f8944f1751c5e184e67b42b77aa4003015d7c9c98a122184911881c8f75af8 |
| P-RGB-strict | results/agent_p2_common_resource_rgb_20261029_v1_consume_MAIN01.json | ec38708895cefd85e9178936826a0903b427050849c97246674879e19f4a45c4 |
| Hi-tracker-prepare | results/agent_p2_history3_tracker_cuda_20261029_v1_prepare_MAIN01.json | 161af4ff5b70a656ad147f2775d169fb2087aa72a8e07e9b6e40aabce0ae5456 |
| Hi-tracker-report | results/agent_p2_history3_tracker_cuda_20261029_v1_cuda_MAIN01/report.json | ab89ea3c8c7198ce001280a93c73c312c5454f552b4452232e04a6776adf6884 |
| Hi-tracker-strict | results/agent_p2_history3_tracker_cuda_20261029_v1_consume_MAIN01.json | 080fe1173895621ea4c0565671509a267ba456135fedfac313ddac7894f98138 |
| targets-prepare | results/agent_p2_target_controller_cuda_20261029_v1_prepare_MAIN01.json | 1a021116784dfaa5b4ce976f694fbfde377e6b282ce3553cb57d49a5e033c0aa |
| targets-report | results/agent_p2_target_controller_cuda_20261029_v1_cuda_MAIN01/report.json | b1358eb3cec42cdddb30b0d138c2bc2e68773f195aa985977bf9c090527382cb |
| targets-strict | results/agent_p2_target_controller_cuda_20261029_v1_consume_MAIN01.json | 34e0491ca05a1b381db3541997c6215c44f9927761d6123a559e604160f0be4a |

B的13.41753181698732秒/errors[]来自MAIN实际交接，不据此改变H/N/I或宣称可比GPU时间。
本轮文件SHA核对不是重执行任何consume/fullstrict，也不假称已审全部NPY/PNG。
P已有六槽可整体复用其真实模型/物理/RGB证据，来源成本全部继承；不重跑known-root面板。
新CEM动作需要自己的物理/RGB，绝不能借P的PNG当作新动作执行结果。

## 3. 身份：相同输入已核，task导出并非相同文件

| 根 | 原H2 task_export（B.slots） | P2 H3 task_file（P.slots） |
|---|---|---|
| 0 | results/p1_constructive_moving_history_encode_cuda_20261027_v1_cuda_main01/task_00.json.gz，SHA a42bfbb1ea692782355e2572a5df2cd048b72cf3e6c41fee40eb12503674959b | results/p1_moving_overlay_recompile_cuda_20261027_v1_cuda_MAIN01/task_00.json.gz，SHA 80f0f4ed14498509e596322b830f0f8d1bedc4d38f9d3b7d64b57da375f9a132 |
| 1 | results/p1_constructive_moving_history_encode_cuda_20261027_v1_cuda_main01/task_01.json.gz，SHA 101add21a7a40591167a6ab8e8b22c3c62cb1fd700c741fd36cd2073e2d5df74 | results/p1_moving_overlay_recompile_cuda_20261027_v1_cuda_MAIN01/task_02.json.gz，SHA 009ed5769fb55513100c1223c5659bd5b34911b5cfe13e9b121117c14a0db2d7 |

原H2 task_id分别bf89d0d3…/6643e3cb…；P2为f5145aab…/69755789…，**不得重命名成同一个task_id**。
已从B/P两份实际prepare转成原float32，对全数组C bytes精确核得：

| 数据 | root0 SHA256 | root1 SHA256 | B/P关系 |
|---|---|---|---|
| context3×192 | a24f6ebf1755fc8280072aa845a967fd69e958178f8871b90f139d2bb76ffbf6 | ee4deee1cf18f55819634ceb8a541985cf772a22948bc3400c32400eda1f9988 | shape/bytes全相同 |
| warm2×10 | 591bf7d3e96c2eb0e9c110b66539703c63d61227c13eab358e5ce582bdb43405 | 591bf7d3e96c2eb0e9c110b66539703c63d61227c13eab358e5ce582bdb43405 | shape/bytes全相同 |
| goal192 | ab88362749928787117b47abd6d484f83238f565b4fa49684dc71cfdf7c6d6ca | 7d7b05f8dd53e8e48ad84ac448925bf360a9380cdb33b619228c30c2f2da2f69 | shape/bytes全相同 |
| blind NPZ | 3c749e224e075061353dee590aff5883b202854451ef95f9bbf951510e0f6f0c | 4e9aeb8504db653552c4a9924cbc0c3ddfc53f1e6f3a56e5b73fc6c8f1b3be39 | B/P/Hi tracker的文件SHA相同 |

原B盲输入在 `results/p1_constructive_moving_history_source_20261027_v1_main01/s0_blind_inputs.npz`
及`s1_blind_inputs.npz`；P2在`results/p1_moving_target_overlay_source_20261027_v1_main01/`
的`s0_off_blind_inputs.npz`及`s1_off_blind_inputs.npz`。不同路径保留，不据同SHA伪造来源相同。

新H3直接采用P-prepare两原slots/task identity，不修改H2 task或其horizon元数据。
物理必须用P-physical-prepare原constructor/reset recipe/root seed/goal/native状态，
warm10 float32前5[.1,0]后5[0,.1]真实执行，核RGB0/5/10后才部署raw15；
禁止从latent相同推断physics等价、reset到imagined poststate、改native goal为latent surrogate。

## 4. 当前协议差距：同名H3不是同一个轴

| 栏 | bounded native CEM现H2 | P2 joint/composition/GRASP H3 | Hi history3 / target-controller |
|---|---|---|---|
| 历史 / 未来 | 历史3，未来2 packets/raw10，native部署若有则到原frame20 | 历史3，未来3/raw15，到原frame25；P2相对cp15 | 历史3，未来3/raw15；history3不是3步历史都预测 |
| 模型 | 原LeWM installed类/原weights | 同原LeWM tensor hash；原packet full3 | 原Hi图/425 state与facade；已有parity是指定输入路由，不是任意动作全函数相等证书 |
| normalizer | P2工程mean/std，一次sub/div，不是Hi训练scaler | 同P2工程坐标；raw经原packet处理 | smoke直接接normalized history/候选；raw inverse及Hi训练scaler未因此认证 |
| action support | 全box tanh(float32无界Gaussian logits)，不是logit球 | ONE joint30 L2 ball，radius=.75sqrt15，float64 chart→raw32；不是三个独立packet球 | 无界normalized Gaussian；native raw可能越界，不能clip后沿用评分 |
| 规划目标 | 原task goal，native float32 sum192 | 同原goal192；终态float64 mean192；GRASP另有lifted loss/sync | smoke为原goal；四target槽则是prior FIRST/program postcondition，不是原task goal，来源物理时距也不同 |
| 返回 | logit mean，tanh后raw重载，另付whole/seg | joint/composition真实incumbent，GRASP LAST动作，均另付replay | native normalized mean需另付actualraw-token replay后才谈部署 |
| 资源 | N300 I30，batch300；见下一表 | 1902 API和1902 samples双cap/450s搜索；自然消耗不同，不padding | N8 I2 K2；12API/54samples每槽，更多proposal上游成本另列 |

虽然数学上sum=192×mean，float32与先转float64再规约可能不同排序，不能倒称优化完全同criterion。
跨模型feature MSE不能直接比大小；native任务指标与同LeWM RGB Q必须各列。
当前两known roots不是fresh；不同动作/方法的多个分支不增加独立根数。

## 5. 最小新增H3桥：预先固定接口，不实现

待MAIN批准的新stem建议 `agent_lewm_bounded_native_cem_h3_20261029_v1`；本文件不创建它。
固定P-prepare roots0/1、off、原goal、不带prior/program/arbitrary target，也不导入其future witness。

1. `prepare`接表中P输入authority、B实现source/protocol；逐字段绑定P的task/context/warm/goal/Q/
   normalizer/source/model以及原物理recipe。不同task_id保持原值。先seal输入，不读物理outcome。
2. 原安装CEM仍N300/I30/topk30/var_scale1/batch1，每root显式MAIN seed0；logit shape改为[1,3,10]，
   全H零初始化。每轮一次CUDA-f32 tanh→raw[1,300,3,5,2]→normalize一次。保留全box方法，
   **不**为追齐P2球临时投影、不改CEM候选分布或std、不额外nominal/best replay。
3. planning官方cached whole H3：真实context3、warm2+future3共5 action tokens；AE5一次，
   predict/proj各3次，保存完整6帧含3历史。每轮全300预测/native IO durable后才原criterion/update。
   保留native float32 sum192优化，以隔离本次最小时距迁移；最终统一另算float64 mean192只作评价，
   不伪称已消除优化规约差异。
4. actual final logit mean→一次tanh成为raw15，durable后重读raw bytes，原normalizer一次生成部署tokens；
   whole H3一遍，segmented1+1+1独立一遍。后者每步自己的pred移窗/真实action shift，无futureRGB。
   两根×两布局全4输出先seal才4个final criterion；全future576/frame15/20/25分栏bits/maxabs/ULP。
5. 新shape hooks必须自有AE5/planning batch300/7模块及final16模块/root合同（whole7+segmented9）；原v1 trace写死AE4，
   不能调用其旧guard后猴补globals。复用原数学/loader/codec不代表旧shape已认证H3。
6. saved readout固定全部新CEM两根与P既有6槽，逐root/arm分栏；旧P资源/时钟不能混入新job，
   不事后删GRASP或失败分支。允许同任务、同frame25描述性差值，不作公平方法强弱排名。

### 固定计划资源：API/sample/token不能相加

| 轴 | 新H3 CEM每root | 新H3 CEM两root | 现H2 CEM两root | P2每root/arm现合同 |
|---|---:|---:|---:|---|
| planning trajectories | 9000 | 18000 | 18000 | 算法各自schedule，不转换为9000 |
| cost oracle / criterion API | 30 / 32 | 60 / 64 | 60 / 64 | 原objective/AD/导出另列 |
| official rollout / segmented predict wrapper | 31 / 3 | 62 / 6 | 62 / 4 | 原单packet transition；wrapper不混比 |
| AE API | 34 | 68 | 66 | joint1902，composition自然1062，GRASP1902；均受wall |
| AE samples | 9004 | 18008 | 18006 | batch1，按实际完整ledger |
| AE tokens | 45014 | 90028 | 72020 | full3 packet通常3×实际AE samples；partial分轴保留 |
| predictor / pred_proj API（各） | 96 | 192 | 128 | 每轴1902上限，不是各臂保证实耗1902 |
| predictor samples | 27006 | 54012 | 36008 | 每轴1902上限；与CEM batch300明显不等 |
| predictor context tokens / projection rows | 81018 | 162036 | 108024 | 每样本3tokens，partial逐模块核 |
| AD / J / VJP / SVD | 0 | 0 | 0 | joint/composition含J/VJP/SVD；GRASP632backward/Adam上限 |
| 新encoder/image projector/physics/training | 0 | 0 | 0 | 原合同各自计账，不抵扣上游 |

H3公式：predictor API=2×3×(30+2)=192；samples=2×3×(300×30+2)=54012。
AE tokens=2×(30×300×5+5+3×3)=90028；不是每predict一个AE3的P2布局。
原P2联合方案各85J/16320数学VJP上限，不是实测engine backward；composition/full-H3
AD维度与J元素数不同，不能仅凭J计数宣称等算力。450s是搜索scope，B现900s是整个job含load，
即便实际时间接近也非wall-matched；batch300与batch1不直接比较API效率。

新H3工程提议：whole job900s、GPU allocated3GiB/reserved4GiB、floor200MiB、output2GiB含128MiB
reserve；这是待MAIN批准的存储保护，不是科学预算优势。H3全精度IO比H2大，不能沿旧1GiB
cap口头保证：实施前须完整shape/census上界RAM证明，超界停止报告，不缩N/I/精度/分母。

## 6. 如何补实际任务度量；不偷用现成终点

新模型replay strict认证不自动授权physics。若MAIN后续批准，新两分支复用
`experiments/spectral_reachability/agent_p2_common_resource_physical_20261029_v1.py`的原生helper，
但使用自有两分支schema，不伪造joint/composition/grasp计划。固定warm10+raw15、原nativegoal、
cp0/5/10/15，首plan action前history/RGB/goal guard，earlystop保缺cp/null不补帧。

按原模板完整执行上限：2constructor/close、4reset、50public、508physics、8set_state、
6goal_setter、4goal_render、2goal_bind、162explicit_render；这些是拟执行账，不是已付成本。
后续RGB固定新两根cp5/10/15共6 encoder/proj（batch1，无candidate选择），goal用原封存192，
checkpoint absent则null，不拿P的编码填新CEM。完整164或其它面板不能当新两根。
native success严格沿原agent+object及角度实现；native7/full distance、objectXY/angle/velocity、
same-model RGB goal-Q分栏，不能object-only失败代替完整任务成功门，也不能latent低Q当native成功。

## 7. 其它现成baseline能复用什么，不能当什么

- H2 matched CEM/nominal-MPPI：
  `results/moving_prediction_matched_sampling_cuda_20261027_v1_cuda_MAIN01/report.json`，SHA
  `60872ad114700731d82d293a616a4dede54a53b7be6514ad0cdf231d838d3ed8`。
  固定每方法/root144 API/samples，全四组576；24×3的全库sampled winner，不是returned nominal。
  双packet球、N24/I3/ddof0/种子生命周期均不同，不是本native CEM的廉价同预算版本。
- H2 beam：`results/agent_kinodynamic_matched_domain_20261028_v1_cuda_MAIN01/report.json`，SHA
  `d4976aeca357b5324c74be8b2b7e3bfc254e82b834b3acfe1b0ab90a1ef4614c`。
  20→6×20+whole/seg=144/root，固定bank/packet球，不是RRT；不能截P2末步当它的H3结果。
- 原native未bounded CEM及paid mean：
  `results/agent_lewm_native_cem_slice_20261029_v1_cuda_MAIN01/report.json`、
  `results/agent_lewm_native_cem_slice_20261029_v1_consume_MAIN01.json`；
  `results/agent_lewm_finalmean_paid_evaluator_20261029_v1_cuda_MAIN01/report.json`、
  `results/agent_lewm_finalmean_paid_evaluator_20261029_v1_consume_MAIN01.json`。
  无界normalized域/默认seed1234；原mean域外及roundtrip差异不修复，不能clip后沿用paid评分。
- Hi history3 smoke及四target-controller：复用表中完整authority/输入、真实mean付费replay接口，
  不能把same3-history说成sameH2，不能拿prior FIRST/program target结果对原goal作公平对照。
  两missing target slots照原分母保留；后续其它六目标版本须另绑定，不在本协议回写旧四槽。
- 当前GRASP是P表中common-resource history-aware joint30版本，不能引用旧H2 GRASP或官方论文值
  填此H3栏。PlanWAM驾驶协议/官方SAGE或其它作者值不进入本地PushT排名；作者原表独立
  `results/p2_reported_baseline_table_20261027_v3/export/report.json`，不是本地实测。

## 8. 公平性剩余项与停止线

本最小H3桥只声称same-time/same actual task接线。若要求真正方法公平排名，还缺：共同search support
（P2 joint球 vs CEM全box）、共同优化规约（f32sum vs f64mean）、共同预算政策与计量scope、
相同硬件/新鲜独立cohort。不能将本协议预定H3输出自动升级为公平结果。
要统一support或criterion必须再获MAIN对新方法定义的明确批准；不能在本runner静默投影/换cost。
要统一1902 samples cap，当前N300/I30原样不可能：每root27006>1902，必须另事前决定N/I或新预算，
不能padding、漏计batch、只计96API冒充已matched，亦不能依据本次效果选择变更。
不为满足cap将H3缩回H2。P2 AD/J/VJP的额外成本仍逐轴，FLOPs unknown保null。

实施/生产前依次需MAIN批准新H3设计及工程cap、新shape/source闭包与RAM、真实CUDA/完整strict、
独立原生adapter与授权、saved RGB，再讨论上述受限描述；若只做模型slice，真实任务效果全部unknown。
这不放宽P1/P2/freshness/gate，不宣布组合收益或新方法通过。

本轮实际核对：表中JSON字节SHA、限定prepare数组的f32 shape/bytes及原source固定预算公式。
未读bounded/其它新效果选参，未调用NN/physics/生产prepare/consume/fullstrict、未新增agent。
只新增本文，无可运行的新H3 CLI；现有B CLI继续按原冻结H2执行，不能传H3参数冒充新版本。
