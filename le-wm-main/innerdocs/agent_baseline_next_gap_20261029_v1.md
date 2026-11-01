# 强基线下一缺口：以 MAIN 实际产物为准

2026-10-29。本次只新增本文；只读定向文件、JSON、源码CLI及字节SHA，不执行producer/consumer/fullstrict、NN、物理、网络或子智能体，不代替MAIN代码审计。既有三份common-resource汇总已经完成，不另造stdlib聚合器/registry。以下路径除明确绝对路径外，均相对 **/home/gbk/code/LeWorldModel/le-wm-main**，不是外层同名results目录。

## 1. 当前结论与不应重复的工作

1. Hi/P2 pairing并非not-run：MAIN实际导出、consume、join齐全，五组tensor目录全等；随后normalized history3跨实现计算parity也已执行且exact。不能再建议重写loader/catalog/parity或仅补一次smoke。
2. common-resource GRASP六臂模型搜索、六臂物理、固定18槽RGB、独立模型/物理/RGB标量汇总均已存在。不是只有prepare，也不是仍缺GRASP物理/RGB。不能重复立项这条完整链。
3. 真正未补齐的不是再一份状态表，而是：**官方Hi raw scaler出处认证**，以及**新bounded H3 CEM实际raw15到原生/真实RGB的专用部署接线**和更强公平性设计。前者缺资产，后者已有模型/动作证据，但在本次定向检查的现成入口中没有能接受其own schema的部署命令。不得将P3 H3动作见证物理目录冒充CEM部署。
4. 原作者报告值已有完整reader表，继续与本地分栏；本次不重抄作者数值、不从文献值推本地胜负。known2不能包装成fresh确认。

读取canonical audit：外层 `/home/gbk/code/LeWorldModel/docs/D2P_LOCAL_SCALE_AND_DOWNSTREAM_MAIN_20261029_v1.md`，本次快照SHA `06aacbd364bd833598405d99ed822320efbc10c8b2b48c708d26ca60b6b9344f`（该文件由MAIN持续更新，不是本文冻结依赖）。重点对应原行83/127（tracker/parity）、165–169（pairing）、527及“共同cap完整strict已通过”段；更早/更晚段落保留历史not-run快照，不能按文件中某一段文字出现顺序代替实际receipt。本文不编辑该canonical记录。

## 2. 已存在且本次重算SHA的权威

读到完整MAIN receipt表示继承其认证；本轮没有重跑其完整strict，不能称本轮重新认证全部NN、梯度、物理或PNG。下表SHA均为实际文件字节重算，不是从旧doc复制待办状态。

|相对路径|SHA256|
|---|---|
|results/agent_p2_hilewm_pairing_catalog_20261029_v1_prepare_MAIN01.json|dc8ec5d7f31180c26c6e795911875086827e3cbeb8d2356cef98eb249f6bceec|
|results/agent_p2_hilewm_pairing_catalog_20261029_v1_cuda_MAIN01/report.json|5605d9d7966a38ac146a84f987a09099dae33da5a2864110a69da06fd1970650|
|results/agent_p2_hilewm_pairing_catalog_20261029_v1_consume_MAIN01.json|9a5406bd076ccfd8a931d41943a64a1f905fa07a6a42711d419db923bcd7abd7|
|results/agent_p2_hilewm_pairing_catalog_20261029_v1_join_MAIN01.json|1523183330336bd3e66884a15fa96e7a15161abd8a343d1f50dba7a0427494b8|
|results/agent_p2_hilewm_computation_parity_20261029_v1_prepare_MAIN01.json|796c80ca14574b29b9920b0ef53b02cc6c3b3b47842708951e5bc0b17aadf4e1|
|results/agent_p2_hilewm_computation_parity_20261029_v1_cuda_MAIN01/report.json|12fe43cc4a796fffc18b98fcc867283f8534c22110d794729652ac112aedead8|
|results/agent_p2_hilewm_computation_parity_20261029_v1_consume_MAIN01.json|3555696c02329062084d791c0c40958084d7352e6f7680938bade0068afe9480|
|results/main_hilewm_probe_saved_consumer_20261029_v1_MAIN01.json|ce69411a511820a0b3b11b05cb5bb0959e54ae65a0edc9e50cd0e790d721c0c9|
|results/agent_p2_common_resource_grasp_cuda_20261029_v1_prepare_MAIN01.json|7facba500439172ced280ec0b893659715139b1e5e22073ae7313665a1c0faec|
|results/agent_p2_common_resource_grasp_cuda_20261029_v1_cuda_MAIN01/report.json|d781b493ecd624db846e3225ed0177bd3fe250d6ba9213d925b2a57956dd2242|
|results/agent_p2_common_resource_grasp_cuda_20261029_v1_consume_MAIN01.json|d3ee465f2fa6d31345f369f9514165c4817c51a29c3d00ffdd2d681e5829cc1f|
|results/agent_p2_common_resource_physical_20261029_v1_prepare_MAIN01.json|4b4f19146c352e00e6007b88030738264157b781ce34636120fbf2ad44321742|
|results/agent_p2_common_resource_physical_20261029_v1_MAIN01/report.json|d23bcc56d52bce969a75faeff776e92a95ad37d58ff7bb9df2879eb624a29627|
|results/agent_p2_common_resource_physical_20261029_v1_consume_MAIN01.json|ae5a62dbc158ac1a7da217a413987f5a88bafde357326559c2c384c016ada87c|
|results/agent_p2_common_resource_rgb_20261029_v1_prepare_MAIN01.json|ed7b4ac2de94a6558adb12af587a66532c4ab54f60e56fd064cab158460f3d81|
|results/agent_p2_common_resource_rgb_20261029_v1_cuda_MAIN01/report.json|50f8944f1751c5e184e67b42b77aa4003015d7c9c98a122184911881c8f75af8|
|results/agent_p2_common_resource_rgb_20261029_v1_consume_MAIN01.json|ec38708895cefd85e9178936826a0903b427050849c97246674879e19f4a45c4|
|results/agent_p2_common_resource_summary_20261029_v1_MAIN01.json|61080aedce753b4a03bde8f8bf414d58e5a17f65ddce4b16e60efe603a89e093|
|results/agent_p2_common_resource_physical_summary_20261029_v1_MAIN01.json|052b1fe3b7d6000e1c8971a277414e1e183ab7b419a15e0abb044526de162959|
|results/agent_p2_common_resource_rgb_scalar_summary_20261029_v1_MAIN01.json|b3c1493d92816e0956f6e952a885505147879fdd243528d10f6098a82608b353|
|results/agent_p2_history3_tracker_cuda_20261029_v1_consume_MAIN01.json|080fe1173895621ea4c0565671509a267ba456135fedfac313ddac7894f98138|
|results/agent_lewm_bounded_native_cem_h3_20261029_v1_consume_MAIN01.json|d0e215618c367049c09eed7e93accb90615973f4f8cb322fd19359fef1f7e7a4|

## 3. Hi-LeWM：哪些gap已关闭、哪些仍未关闭

pairing report有333项P2 runtime目录，consume明确independent_tensor_rehash=false；join的encoder/projector/action_encoder/low_predictor/low_pred_proj总条目为198/9/6/111/9，后者predictor111含30 None、81非None。全部missing/unexpected/mismatch/layout differences为空、all_five_tensor_catalog_equal=true。不能把None计数当新权重或将目录配对当函数/训练数据配对。

后续computation parity receipt为all_known2_arrays_verified_not_scientific_admission、all_values_exact=true，已核known2真实history3跨实现保存数组。canonical记录每根104比较全部bitwise一致；资源为16encoder/16projector、18AE/18low_predictor/18low_pred_proj、6 P2 predict API/6 scalar VJP。normalizer_pairing/raw_deployment/independent_gradient_recompute仍false：后一个是saved消费没有独立反传，不等于MAIN没有运行梯度。

MAIN另有独立owned-low-probe receipt：forward_bitwise/gradient_bitwise/layouts_exact=true，directional_check.passed=true；official_parity/P2_pairing/scientific_admission=false。模块各15attempt/return、3 scalar VJP、9AD primal，0encoder/physics。该receipt绑定actual report ff4d92b73d5f11bb40953acb0b6e90c60bd97e6c991894066b51ab716e1e60c1，manifest 1d35207ac68979463012fb6c74a5190b344b142903869a55a9fc3054d16ccd91；loader_source_scope明确metadata_current_bytes_verified=true但metadata_before_after_manifest_verified=false，不将非执行doc/tests追认为原始运行前后闭包。

history3 tracker也已完成，不再当待接入口：known2 H3/N8/I2/K2/seed0 smoke，24API/108samples，returned mean whole/seg重放已付费；不是强搜索或官方raw部署。官方raw inverse仍缺与Hi checkpoint/data/eval配对的原始已保存process mean/scale、flags、字段顺序、精度、字节hash和provenance。现有P2工程mean/std不能升级为Hi原训练scaler。镜像v2实际metadata路线已完成（innerdocs/agent_hilewm_normalizer_asset_route_20261029_v2.md），没有找到该sidecar；此范围不证明全历史不存在。不得重fit/借常数、重复大包扫描或把raw_actions=null填成零。

## 4. common-resource GRASP：已有完整本地对照，而非官方复现

固定原两known off task，history3、H3/raw15、相同goal/identityQ/工程normalizer；六槽顺序root0 joint/composition/grasp，root1 grasp/composition/joint。每槽共同1902 predict API和1902 batch1 samples上限、450s搜索上限。实际所有槽natural_completion、均有export、无wall_overrun：

|每root方法|实际predict API/samples|完整J|请求数学VJP|scalar backward / Adam|其它独立轴|
|---|---:|---:|---:|---:|---|
|joint|1902 / 1902|85|16320|0 / 0|255 AD primal packet，2 SVD、1ridge|
|composition|1062 / 1062|85|16320|0 / 0|129 AD primal packet，2 SVD、1ridge；不padding|
|bounded history-aware GRASP|1902 / 1902|0|0|632 / 632|1896 AD primal packet，532 noise；532joint+100sync|

预测次数上限相同不等于实际次数相同、AD相同、等FLOPs或等总算力；算法本身不是要求相同的控制字段。GRASP不是官方750步DINO-WM复现。外层工程上限3600s/1536MiB输出含96MiB reserve/floor512MiB/GPU4GiB allocated、6GiB reserved；搜索scope不含loader/saved strict，不能把总wall误称450秒公平搜索。

physical receipt errors=[]，6constructor/close、149public/1514physics/483PNG。root0三臂与root1 grasp/composition各满15，root1 joint在14 native success提前停，cp15 null。all_six_physically_executed=false不能译成有分支未派发：所有六分支有执行记录，只有五条达到完整15步，具体布尔定义仍归原source审计。

RGB固定18槽/18对，实际17encoder+17projector；missing_slots=[17]，real_pairs_available=16。最后一格是root1 joint/cp15早停，不补终态、不删根、不强补图。旧GRASP750单独物理/RGB亦已完成，但不能与本六槽数据混作同一实验。模型Q、RGB实际Q、agent/object XY、速度/角度/native7和native success各自分栏；一次成功或低模型Q都不支持统一方法胜负。

三份独立summary均已有actual MAIN结果，故不再实现“missing scalar工具”。尤其RGB scalar summary已独立解原NPY、固定18槽/18对及误差恒等式，producer_math_imported=false；其存在不能代替MAIN代码审读，也不自动认证fair-ranking/freshness。

## 5. 下一实质实现缺口与预算前置

建议优先补 **已有bounded H3 CEM的部署见证**，而非再跑pairing/GRASP。不是在本次授权内启动生产或给旧job补公平标签：

- 可复用已封存agent_lewm_bounded_native_cem_h3_20261029_v1 own receipt及其prepare/report/source；固定原两root、相同task/history/goal和全部返回mean raw15。H3/N300/I30/K30/seed0已事前固定，seed0是MAIN选择，不是用户指定或安装默认1234。raw经一次CUDA tanh和一次P2 normalizer、落盘重载，whole/seg actual-token预测已付费，不需要为同一字节动作重做CEM。
- 已付模型成本：192predictor API/54012samples/162036context tokens；68AE/18008samples/90028tokens，0encoder/AD/physics。全box支持、f32 sum规划criterion、batch300和P2球支持/f64 mean/谱AD不同，因此即使同任务H3也不能直接公平排名。
- 真正欠缺的是新own-schema raw部署adapter与consumer、工程字节/事件预算证明及物理后RGB绑定：不得将common-resource physical CLI直接喂CEM receipt，不伪造旧plan/selection digest。部署只引用实际raw15、reset/warm10原recipe、原exact guards；不重新inverse/clip/tanh、换root或改mean。缺任何raw/recipe/guard即blocked。
- 若MAIN批准上述两条新部署：科学动作上限2×(warm10+raw15)=50 public controls，原同类recipe物理安全上限508 physics、162PNG可作为**待源码census核准的提议**（含reset/setup，不能称已证明）。原生早停保null，不补满。新增encoder上限6（2root×cp5/10/15）与6projector，0predict/AD；goal/history复用。该工作尚未实现/获执行授权，不能拿这组估算作为可运行合同或沿用别的adapter预算认证。
- 工程cap须在新源码完整shape/journal/metadata证明后由MAIN确认，不能在本文擅自增cap；NN仅MAIN HOSTCUDA，物理另MAIN主机执行。旧六臂与两个新CEM部署是跨job描述性扩展，不自动成为同job公平对照。

更严格的公平比较仍须事前固定共同动作support、时距、实际部署token、目标/归约/成功定义、资源向量和wall scope、原完整方法/root分母及fresh准入。不能根据已见root1成功或某模型Q挑对手/预算。当前known2不提供独立确认；作者checkpoint/process/训练权限配对也未因此补齐。

## 6. 现在可执行的saved入口，不重复fullstrict

最小下一命令直接查看**已经完成**的独立18槽/18对表，全量保留而不生成新wrapper或新结果：

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
jq '{schema,fixed_denominator,plan_denominator,group_denominator,pair_denominator,missing_slots,real_pairs_available,rows,pairs}' \
 /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_common_resource_rgb_scalar_summary_20261029_v1_MAIN01.json
jq '{all_five_tensor_catalog_equal,groups,config_pairing,computation_pairing,normalizer_pairing,original_class_parity,scientific_admission}' \
 /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_hilewm_pairing_catalog_20261029_v1_join_MAIN01.json
```

如MAIN需要重新校验现存RGB scalar算术（不是新增实验，也不是缺项），已有own CLI：

```bash
.venv/bin/python -m experiments.spectral_reachability.agent_p2_common_resource_rgb_scalar_summary_20261029_v1 verify \
 --summary /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_common_resource_rgb_scalar_summary_20261029_v1_MAIN01.json \
 --summary-sha b3c1493d92816e0956f6e952a885505147879fdd243528d10f6098a82608b353
```

该scalar实现没有producer-math import，校验绑定并重建saved算术；本轮不执行它。不要误用common_resource_physical_summary verify作为轻量入口：该源码会递归执行完整physical/planner strict，本轮禁止且无必要。上面read-view为0新NN/physics/梯度，只有现有小JSON读取；scalar复核会读取其绑定NPY/source，遵守原reader限额，不给新预算承诺。

**部署缺口目前没有真实可运行CLI**：不能虚构新adapter命令，也不能给已有common-resource CLI换入不兼容CEM schema。下一需要MAIN分派的是上述窄slice实现，而非再消费已完成结果。Hi官方raw缺原process，PlanWAM缺对应驾驶模型/4camera+ego/数据split/evaluator闭包，均没有可认证新生产命令。

## 7. 作者值与本地证据隔离、检查范围

作者表沿用innerdocs/agent_strong_baseline_reader_table_20261029_v1.md及results/p2_reported_baseline_table_20261027_v3/export/report.json的版本化HTML/PDF定位：GRASP学习模型成功率、Hi环境成功率、LeWM原solver表、PlanWAM驾驶分分别列。本文不重新运行解析器、不网络拉取、不把它们混进本地六槽native成功或latent MSE排名。SAGE/PlanWAM之缺本地完整闭包不填成本地0分。

本次验证仅：22个实际JSON SHA重算、定向schema/计数/flags/接线字段读取，canonical相关段与现有CLI源码/文档核对；文档SHA/path表另用stdlib只读检查。没有新source/tests、没有pytest或actual完整consumer执行，因此不冒称新增测试数/独立认证。source测试/审计继续MAIN负责。所有旧文件与dirty改动保留；P1/P2/scientific gates不变。
