# Moving native rollout cached-embedding：官方API与既有packet路径比较

## 唯一问题、来源与限制

仅检验实际加载的官方`model.rollout` cached-embedding路径与已封存手写packet H1/H2是否得到相同数组。
不是再执行手写transition，也不训练/拟合/改变Q/normalizer/候选或控制；不是新物理样本、效能或P1准入。
本agent只实现三新文件、RAM/mock测试及纯saved build smoke，实际神经与物理由MAIN独占；不修改封存文件。
使用experiment-code技能的窄实现/日志/错误处理；不采用与用户禁止训练/扫参/新图要求冲突的模板。

实际安装源码：`.venv/lib/python3.12/site-packages/stable_worldmodel/wm/lewm/lewm.py`，
SHA `ccc2ac2880880f85ea71a5baa2152ba97e3f6dc5cb546d27570795aa6020d3d2`。
已完整只读核该类：rollout assert pixels并仅取`H=pixels.size(2)`；若`emb`已存在，则不调用encode，
将全部action_sequence一次AE，再对`T-H+1`个successor循环调用原predict。
cached emb是该版本官方显式支持分支，不是利用未定义行为。固定history_size=3、B=S=1。
运行时检验实际类module/name/源码路径/SHA、原rollout/predict/encode为未被实例覆盖的官方方法；
本实现随后只临时计量predict、禁止encode、挂模块hooks，**不替换或重写rollout循环**。
若安装版本、源码、模型类或cached接口不同，拒绝，不选另一个模型/fallback。

参考actual horizon报告：
`results/p1_moving_teacher_horizon_localization_cuda_20261027_v1_cuda_MAIN01/report.json`，
SHA `e319035d9e13a6ae2dc4cbfa5cefb9eb2e6ccd49dbcf35408fda1704f5f34253`；
其prepare SHA `c87b35732ebaa3a3f6c74ec2e65ec817d9b15f319e0b687cc3ffa453f03935ec`。
冻结horizon source SHA `8b8e4bdb708b914b3fc50bba9e0b6c3b624cc724396c05d660cc79432879e1ed`。
prepare经其严格verify重建原live/task/model/source闭包；参考report全artifact SHA纳入新bindings，
只hash参考NPY/封存receipt，不在prepare解码预测标签。参考模型tensor与原task一致。
旧horizon ownconsume及MAIN保存核对是MAIN已完成的工作，新build不冒称重做独立审计。

## 固定12自身轨迹与24官方rollout

cohort与原live完全相同：2roots×GN/D2P×O/I/R，逻辑12 episode，仍只有2个known任务。
每episode固定root_H2后suffix_H1，不筛选、不去重、不因预期误差改动作；无候选或代替控制器。
每次info.emb都来自实际原context，shape `[1,1,3,192]`；cached路径不执行encoder/projector。

| kind | info.emb的三真实过去latent | info.pixels三真实过去RGB | action_sequence | 官方调用数 |
| --- | --- | --- | --- | --- |
| root_H2 | 原source z0/z5/z10 | 本episode warmup0/5/10 | `[warm0:5,warm5:10,prefix5,suffix5]` 4token | 1rollout、1AE(T4)、2predict/predictor/pred_proj |
| suffix_H1 | source z5/z10/own actual prefix5 | 本episode warmup5、plan0、plan5 | `[warm5:10,prefix5,suffix5]` 3token | 1rollout、1AE(T3)、1predict/predictor/pred_proj |

每token严格5条实际raw动作、time-major2坐标flatten为10，原float32 subtract/divide恰一次；
不把每raw当packet、不zero padding、重复/clip/tanh历史，不把三输出token当三次自由rollout。
root→H2与suffix→H1仍是原合法packet时间尺度。
suffix使用actual prefix是本事后诊断权限，不倒写I的原控制权限。

plan0为warmup10之后attach原goal且with_target=False，build强制这两帧pixel SHA相同。
三个RGB均来自原journal fresh_observation唯一receipt，旧264PNG与raw history全量检查沿horizon source重建。
原goal/frame不改；所有query中的RGB receipt必须属于该episode正确phase/step，缺帧或future step10替入即拒绝。
不读native/witness/未来帧去选输入。只提供真实过去RGB，不造dummy，也不复制terminal图冒充过去。

## pixels与缓存的实际接线

运行读取原PNG，逐file/pixel SHA验证uint8[224,224,3]。
沿原`p0.image_transform`和`p0._image_tensor`转换：原ToDtype(scale=True)/ImageNet Normalize/Resize224；
stack后info.pixels为CUDA float32 `[1,1,3,3,224,224]`。
只做原输入预处理，没有CPU神经推理或新image encoder调用；cached路径仅用其H=3。
不把已存在emb视为可以传假pixels的许可。真实PNG及变换源码已封存。

保存每call的实际shape/dtype、变换后pixels array SHA、原RGB receipt、cached emb NPY和action_sequence NPY。
不重复保存24份约1.8MiB的pixels float tensor，原PNG不复制；因此不虚报可以从输出目录单独恢复像素，须读绑定的原PNG/变换。
consume核源PNG绑定与call shape/receipt/缓存数据；不以NumPy近似变换冒充逐位复算torchvision pixels tensor。
cached官方源码只读其shape，所有预测模块的实际输入输出另有完整NPY逐位证据。

直接调用`model.rollout(info, actions, history_size=3)`；记录其原样full predicted_emb：
root为`[1,1,5,192]`（原3帧+2successor），suffix为`[1,1,4,192]`（原3帧+1successor）。
同时保存final192和官方写回info.action的首3token，并核原emb/pixels对象保留及H3 prefix正确。
每次predict内部模块输入/输出全部保存，AE实际一次T4/T3，第二predict使用AE输出窗口1:4而非重新AE。
每个预测context必须是官方前3真实帧/随后shift加入实际前一次预测，保存数据与内部pred_proj逐位闭合。

## 预算、数值与保护

固定新增24rollout API、24AE、36predict API、36predictor、36pred_proj，共156条operation entries。
rollout/API/内部模块是嵌套计量，不相加宣称60次rollout或96次预测。
0encoder/图像projector、0AD/J/VJP/backward/SVD/solve、0physics/reset/render/publicsteps、0训练/拟合/选择。
原冻结LeWM一次加载，HOSTCUDA-only、eval/requires_grad=false、原tensor SHA前后不变、
MATH SDPA、TF32off、autocastoff、inference_mode；无CPU fallback、额外replay、nominal或retry。
encode API与image modules都predispatch禁止；AD/math禁止guard复用冻结core并在异常后恢复。
预测调用本身只用官方类rollout，不能调用旧Native forward来凑名义API。

完整输出cap8MiB、200MiB disk floor、256KiB final reserve；300s whole job含verify/load/evaluate/final。
CUDA allocated≤3GiB/reserved≤4GiB；NEW独占output，不删除/扩cap/降分母。
prepare≤256KiB，semantic≤1MiB，完整bindings无丢弃；单gzip/base64、CRC/无尾随/无第二member、
semantic SHA及完整build重建。若大小不满足则拒绝，不悄悄裁authority。

每query实际attempt/return和每AE/predict/predictor/proj module均durable记账，尝试不退款。
初始化24rows保固定12episode/24rollout分母；缺项保not_started/failed_partial/null。
首次错误保first_error，secondary cleanup/source/hash/report各尽力保存，不根据数值差异重试。
全部24个官方预测保存并封predictions_seal后，才解码旧horizon参考final/module标签并作对比。
差异不导致工程失败；工程失败也不当模型或理论负结论。

## 对比字段与criterion尺度

每row保存full_predicted_emb、final192、successor逐个比较；root分别对应旧prefix_H1/open_H2，
suffix对应旧teacher_H1，均按完整identity+kind连接，不按root借用另模式标签。
原horizon module receipt与每个官方step对比：AE窗口、predictor latent输入/action输入/output、完整3token pred_proj输出。
可能AE(T4)与AE(T3)或官方batch窗口产生数值差异，必须如实保留，不能事后改tolerance/重写官方调用使其一致。

每个比较固定报告shape、bitwise、numeric_equal、maxabs(float64差)、maxULP(float32有序整数距离)。
负零/正零：bitwise=false但numeric_equal=true、ULP=0；非有限或dtype/shape错为工程拒绝。
没有atol/rtol、自动accept阈值或“近似等价通过”字段。实验complete只表示合同完整，不代表等价或科学通过。

官方`criterion`对最后native192做float32 mse后**sum192**，本项目Q是**mean192**。
本实现不调用get_cost（可能encode goal），也不调用criterion；只在所有预测封存后用原goal、保存数组
重算`sum192_float64`与`mean192_float64`，新旧分别保存，不混尺度。
实数算术sum=192×mean；float32官方reduction可能有舍入差，不能将此float64复算冒充实际criterion逐位输出。
无新Qfit或阈值，不用criterion sum尺度改选旧动作或解释P1成功。

## prepare/run/consume与验证范围

schema本stem.prepare / .prepare_envelope / .report / .consume。
正常status=`native_rollout_comparison_recorded_not_admitted`；失败`partial_or_failed`。
rows固定24，kind与identity、status、predicted_emb/final receipts、comparison（含module比较与cost_native/reference）。
report外部SHA由MAIN实际完成后交接，不在源码预填新结果。
consume只saved/NumPy：外部SHA/source复建、全artifact集合/SHA、事件链、156attempt/return与ledger、
24预测全在seal前、比较在seal后、全cached数据/AE窗口/逐predict上下文/官方full output闭合、全部ULP/cost复算。
不调用新模型、encoder、simulator或旧producer.execute；数值差异仍正常保留。

cwd `/home/gbk/code/LeWorldModel/le-wm-main`，以下由MAIN完整审读后执行：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_moving_native_rollout_equivalence_cuda_20261027_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.p1_moving_native_rollout_equivalence_cuda_20261027_v1 prepare --output results/p1_moving_native_rollout_equivalence_cuda_20261027_v1_prepare_MAIN01.json
.venv/bin/python -B -m experiments.spectral_reachability.p1_moving_native_rollout_equivalence_cuda_20261027_v1 verify --prepared /ABS/NEW_PREPARE --prepare-sha ACTUAL_PREPARE_SHA
.venv/bin/python -B -m experiments.spectral_reachability.p1_moving_native_rollout_equivalence_cuda_20261027_v1 run --prepared /ABS/NEW_PREPARE --prepare-sha ACTUAL_PREPARE_SHA --output results/p1_moving_native_rollout_equivalence_cuda_20261027_v1_cuda_MAIN01
.venv/bin/python -B -m experiments.spectral_reachability.p1_moving_native_rollout_equivalence_cuda_20261027_v1 consume --output results/p1_moving_native_rollout_equivalence_cuda_20261027_v1_cuda_MAIN01 --report-sha ACTUAL_REPORT_SHA
```

RAM tests不调用Torch/神经/物理：提取实际源码rollout函数AST，使用人工tensor primitives与module hooks执行原循环，
而非另写一个假rollout验证自身wrapper。人工class身份仅在tests monkeypatch拒绝门，不在生产提供bypass参数。
覆盖cached无encode、24/24/36预算、full output/module shapes、数值差异非失败、ULP、非法过去帧拒绝、
partial无参考解码、全seal后比较、hooks恢复、CPU/encode/image拒绝、gzip边界和纯saved consume。
这不是实际加载神经模型的验证；MAIN后续HOSTCUDA结果不得由RAM测试或build smoke冒充。
