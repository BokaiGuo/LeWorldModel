# 安装版 LeWM native CEM：known2 cached-H3 最小可运行切片

本交付只补安装版 `CEMSolver → LeWM.get_cost → rollout → criterion` 的运行接线；
不是论文完整复现、P1准入、fresh确认、控制收益或与已有采样器的公平排名。
worker只实现和RAM/临时文件fixtures，MAIN审读后主机CUDA执行。

## 选择依据与复用边界

已核现有可复用入口，避免再造同类工具：

- GRASP已有 `agent_p2_history_joint30_grasp_cuda_20261029_v1`、
  `agent_p2_common_resource_grasp_cuda_20261029_v1` 及physical/RGB链，本任务不重复。
- Hi-LeWM已有恢复的loader、low-CEM和
  `agent_p2_history3_tracker_cuda_20261029_v1.py`，不再复制outer tracker。
- LeWM已有bounded moving CEM/MPPI和固定动作的official rollout等价链；
  本切片实际新增安装版官方CEM直接调用官方cached get_cost，不调用旧optimize。
- PlanWAM的官方driving assets/evaluator未齐；PointMaze命名资产不冒充官方实现。
- SAGE已定位原仓库，但权重/config/normalizer/revision闭包未齐，不下载或新训练prior。

这些状态只是入口选择依据，不导出跨架构性能结论。
作者报告值单独隔离：本交付不抄作者数值，receipt的`published_reported_values=[]`；
`local_performance_claim=false`。尚无本切片实际生产结果。

## 事前固定协议

固定原moving off两个known roots，source_slot 0、1顺序，不看outcome筛选。
复用`p1_moving_sampling_baselines_cuda_20261027_v1.build(('CEM',))`的saved身份闭包，
不调用其旧sampler/owner consume/fullstrict。
原encoder report：
`results/p1_constructive_moving_history_encode_cuda_20261027_v1_cuda_main01/report.json`，
SHA `e7fb554327b3bb5a96190f6291f8450b2b0723cd981817715cfb7e50aa266bf6`。
原task、blind RGB/actions、批准manifest、sourcepack attestation和model config/weights
文件SHA均在prepare中继承并复核。没有新鲜性或训练lineage认证。

安装版`.venv/lib/python3.12/site-packages/stable_worldmodel/solver/cem.py`固定SHA
`a6662e7f1f2c6ecaf0a8527255d531aa05ed583339aabdf2798d0206382ef51b`；
LeWM源码SHA `ccc2ac2880880f85ea71a5baa2152ba97e3f6dc5cb546d27570795aa6020d3d2`。
默认batch1、300候选、30轮、topk30、var_scale1、seed1234，每root新建solver重置同seed。
H2来自原任务，不是根据结果调参。

初始future actions为完整normalized零均值`(1,2,10)`，无actor/额外nominal评估。
原raw历史前5×[.1,0]、后5×[0,.1]，保持原task normalizer一次处理后的两个warm packets；
`Cost.get_cost`只将它们prepend至实际CEM两future packets，传官方API四tokens。
cached emb为原actual0/5/10 `(1,3,192)`，goal_emb `(1,1,192)`。
仍传真实RGB3 `(1,3,3,224,224)`和真实goal RGB `(1,1,3,224,224)`，不使用dummy。
consumer独立按固定224/ToDtype×1/255、ImageNet normalize、identity Resize224复核像素。
运行时还与真实source transform逐位核对，若不一致fail closed，不引入容差。

官方CEM候选为**未裁剪normalized Gaussian**，不是原packet logit球/tanh/raw界限。
所以不能与原bounded CEM、MPPI、GN、beam作同动作域/等预算/等FLOPs比较，
也不发布逆归一化raw动作供physical执行。
返回action为末轮elite均值，不是已评候选winner；不做额外best/mean replay。
`outputs.costs`只是末轮elite costs平均，绝不是返回均值的cost。

## 真实计账与封存

每root：30 cost API、30 rollout API、30 AE、60 predictor、60 pred_proj；
每次AE batch300×4tokens，每次predictor batch300×3tokens，每次proj900rows。
因此每root predictor samples18000、AE samples9000、AE tokens36000、predict context tokens54000、
pred_proj rows54000。全job分别×2，predictor samples36000。
API与sample/token分别列出，绝不把API和内部模块相加。
原8encoder+8projector继承单列、只列一次；本切片新encoder/projector/AD/physics/training均0。
这不是equal144 runner。

记录每次CEM候选/cost/elites/前后mean/std，四packet动作、完整predicted_emb `(1,300,5,192)`，
以及每个AE/predictor/pred_proj的原输入和完整输出；保存最终mean/std，不缩精度。
encoder/projector prehook拒绝任何新encode；模块守卫要求实际CUDA/f32/inference/无autocast/TF32off、
完整原生shape和五模块顺序，durable intent/return独立保留失败。
请求SDP MATH不等于实际kernel证书，receipt明确`kernel_certificate=false`。
加载前绑定config/weights文件SHA，加载后/运行后比完整参数buffer tensor hash；
这不是权重文件**同buffer反序列化loaded bytes**证书，不冒充更强加载证据。

正式模型criterion为float32 terminal **sum192**，consumer另算float64 sum192/mean192。
CUDA reduction vs float64只用预声明`8*n*eps32*max(abs(reference),1)`的算术误差界，
不用于模型等价/任务成功或修改科学门。elite topk按实际indices/gather复核，不猜相同cost的tie顺序。

full IO主要payload每iteration不超过10MiB，60轮不超过600MiB；另有两个real RGB输入、
NPY头/JSONL/report余量，固定整个output cap1GiB，final reserve32MiB，磁盘floor200MiB。
本source实际数组shape的保守payload测试进一步核上界。
CUDA当前allocated3GiB/reserved4GiB；完整job cooperative wall900s（含verify/load），不重试。
不能抢占中断正在执行的单次CUDA op；超时/预算/异常保存partial而不发布成功receipt。
重建prepare对大型模型文件SHA计算可能占时间，算入wall，不把验证成本隐藏为免费推理。

## ABI / MAIN命令

从`/home/gbk/code/LeWorldModel/le-wm-main`执行。源码/tests/doc准备齐后才生成prepare；
MAIN使用每一步真实返回SHA替换示例变量，不预填未来值。

```bash
.venv/bin/python -m pytest -q tests/test_agent_lewm_native_cem_slice_20261029_v1.py
.venv/bin/python -m experiments.spectral_reachability.agent_lewm_native_cem_slice_20261029_v1 prepare --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_lewm_native_cem_slice_20261029_v1_prepare_MAIN01.json
.venv/bin/python -m experiments.spectral_reachability.agent_lewm_native_cem_slice_20261029_v1 verify --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_lewm_native_cem_slice_20261029_v1_prepare_MAIN01.json --prepare-sha "$PREPARE_SHA"
.venv/bin/python -m experiments.spectral_reachability.agent_lewm_native_cem_slice_20261029_v1 run --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_lewm_native_cem_slice_20261029_v1_prepare_MAIN01.json --prepare-sha "$PREPARE_SHA" --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_lewm_native_cem_slice_20261029_v1_cuda_MAIN01 --authorization MAIN_LEWM_NATIVE_CEM_NORMALIZED_HOSTCUDA_ONLY
.venv/bin/python -m experiments.spectral_reachability.agent_lewm_native_cem_slice_20261029_v1 consume --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_lewm_native_cem_slice_20261029_v1_cuda_MAIN01 --report-sha "$REPORT_SHA" --receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_lewm_native_cem_slice_20261029_v1_consume_MAIN01.json
```

所有输出是独占`results/agent_lewm_native_cem_slice_20261029_v1_*`直接child，拒绝覆盖和symlink。
report含`rows[0:2]`、所有artifact SHA/字节、prepare绑定、model前后hash、journal tail、
实际memory峰值、完整error分母；只complete可consume。
consumer重新saved身份闭包但不嵌套旧fullstrict，遍历全部NPY与journal、复算全部候选cost/elite更新，
检查全文件census，持久新独占receipt。源与模型文件改变后拒绝，不能用receipt回证其他job。

## 测试与未执行项

实际worker RAM/临时文件回归：`29 passed in 1.85s`。
fixtures使用新模块内部的小样本常量，只用于RAM/临时目录native-shaped数据，不替换旧模块globals。
生产`build`强制300/30/30/1234；没有CLI缩分母选项。
真实Store写入后完整consumer/hashchain回归，不依赖假codec；另有模块/候选/elite/历史/像素/
返回均值/计数/partial/文件census/SHA/内存/inherited costs破坏测试；默认参数通过AST读官方源码，
CLI冷启动验证不导入torch。测试不等于官方CUDA实现科学通过。
worker未运行production prepare/verify/run/consume、NN、physics、网络或下载。
MAIN尚需审读、实际prepare闭包验证、可用CUDA/资源并运行与consume；若上游文件已漂移则先拒绝，
不能改pin绕过。已有已声明源码默认不需新增科学超参选择；也不闭合论文公平基线任务。
