# Constructive selected RGB CUDA v1

独占新源码/tests/本文；不改已有sealed，不prepare，不运行神经/物理。使用experiment-code组织固定编码计划、RAM故障测试和durable receipts。实际执行仅MAIN HOST CUDA。

## 固定范围与输入

显式完整physical report路径+外部SHA、planner report路径+外部SHA。两根×GN/D2P四臂必须完整own10，严格原顺序；只编码report.checkpoints枚举的plan5和plan10各一次，共8images。缺帧、提前terminal、partial、guard失败、重复checkpoint均在CUDA前拒绝，不能用last observation、相邻帧、预测帧、旧终态或构造证书补齐。拒绝并不意味着partial运行没有科学价值，只表示不满足本consumer固定8帧合同。

prepare只读：核physical execution_binding→prepare→journal首条prepare相等、journal外部SHA/逐事件hashchain/末条终结；复算完整physical调用数（8reset/80public/816space.step/88explicit render/8goal render/16set_state/4goal_bind）。四branch_closed receipt必须与report逐字段相同。原root+goal state7 hash匹配physical prepared.source_slots；goal_attached及每个returned transition的goal state/RGB SHA须匹配该批准task的goal来源。选择命令逐10步核对。仅重hash physical prepared.bindings中的.py/.md源码，不打开其certificate/sourcepack数据bindings。

checkpoint必须与fresh_observation、通过的guard_receipt、对应returned_transition state一致，state/snapshot/observation_state精确一致。PNG仅从checkpoint枚举，flat本目录，原RGB模式224×224，不convert、不resize补救；文件SHA与原始pixel SHA都验证，encoding前再查一次。prepare绑定全部8PNG、physical report/journal/prepare/binding、handoff文件、planner prepare/export/已选plan，以及模型/source/transform依赖闭包。final重新构造全输入，不接受漂移。

physical handoff必须指向显式planner report SHA/output，并在source_slot/arm/task_id/commands/planner_selection SHA上相等。只读取任务export和四个已选plan，不解析result候选库，不读取sourcepack/certificate/见证/非选中未来物理。opaque planner order须对应source slots0,1；新root family/不同排序须NEW协议，不隐式重映射。

## 评分

复用冻结encoder的原transform/model/checkpoint/native projected CLS192，不学习readout/adapter、不改Q。每个批准task的identity192 Q与goal latent不重编码，root取原task context最后latent。H2末端三帧context包含[root,prefix,terminal]，因此prefix预测取已选plan.prediction.latent_history[-2]，terminal取[-1]；完整context/action chronology、float32 tanh、每packet球及原终态Q通过原core.check_prediction校验。这是saved consistency，独立replay证据继承planner，而非本consumer新做replay。

每帧保存lossless float32 NPY192及来源，报告predicted_Q/actual_Q/root_Q、pred-real MSE、pred/root与real/root MSE、预测和真实Q变化及符号反转。没有新阈值/选择；不从这些结果改任务/算法/预算。不将全图Q改善当物体pose改善；原physical state仅做离线provenance，不进model.encode或planner。

## CUDA与资源

8encoder+8projector，逐张batch1，0predict/actionencoder/AD/physics/training，继承root/goal编码和planner/physical成本非零且另列来源，不声称整pipeline免费。沿用冻结enc.Ledger(n=2)的8图序列和enc.guard；虽旧接口称4n，此处只借8image module序列，不调用旧encode_all/任务构造。

guard禁止model.predict/predictor/action_encoder/pred_proj及autograd/backward/functional/torch.func AD；只有inference_mode内原model.encode。HOST cuda:0，MATH SDP、TF32 off，无CPU fallback/下载。模型eval frozen、参数/缓冲CUDA与before/after tensor SHA、模型文件SHA均核对。

attempt在module.forward前fsync，return在forward返回且CUDA synchronize后记录；这是继承encoder的forward边界，不是一般__call__/return-hook/transform故障的全资源精确计账。已完成encoder后projector或encode输出校验失败，保留实际已完成计数，不伪造image成功。无重试，后续not_started；完整8 NPY和16module完成才成功。源码/模型finalization异常分别保留，主错优先；final写失败仅best effort stderr，不声称产物完整。无断电后自动resume。

总输出cap2MiB（含partial/final），final预留128KiB，启动需202MiB free，每写200MiB floor；NPY总仅7168bytes，report/journal保留完整小向量。wall由MAIN外部timeout。prepare<256KiB且200MiB floor，不复制旧protocol大闭包或PNG。测试全部RAM/mock，不写大tmp。

## MAIN执行命令

cwd `/home/gbk/code/LeWorldModel/le-wm-main`。实际physical终结并审阅后由MAIN提供路径/SHA，本交付暂不prepare。

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_constructive_selected_rgb_cuda_20261027_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_selected_rgb_cuda_20261027_v1 prepare --physical-report /home/gbk/code/LeWorldModel/le-wm-main/results/p1_constructive_plan_physical_20261027_v1_main01/report.json --physical-sha d205850e95f639b649fba9b5c6ee8059f2507431753a538b1ce30eccbd234300 --planner-report /home/gbk/code/LeWorldModel/le-wm-main/results/p1_constructive_blind_planners_cuda_20261027_v1_cuda_main01/report.json --planner-sha 4dd5823e9a273b820d6b7b3f6a36a4a5a19b193383eaa9cbf94442082e79d98d --output results/p1_constructive_selected_rgb_cuda_20261027_v1_prepare01.json
.venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_selected_rgb_cuda_20261027_v1 verify --prepared results/p1_constructive_selected_rgb_cuda_20261027_v1_prepare01.json --prepare-sha PREPARE_SHA
timeout 300s .venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_selected_rgb_cuda_20261027_v1 run --prepared results/p1_constructive_selected_rgb_cuda_20261027_v1_prepare01.json --prepare-sha PREPARE_SHA --output results/p1_constructive_selected_rgb_cuda_20261027_v1_cuda_main01
```

模型原权重不变；GN78与D2P66/root不等预算不因本8帧编码改变。P1_admission=false、非freshconfirmation、非SOTA/公平优胜结论。当前仅工程实现/mock；实际8images及物理完整前置均须MAIN另执行验证。

## 给Huygens的冻结下游接口

输入本consumer实际完整report绝对路径+MAIN外部SHA。要求schema=`p1_constructive_selected_rgb_cuda_20261027_v1.report`、status=`selected_rgb_encoded_not_admitted`、errors=[]、source_verified/model_unchanged=true，原模型hash一致，resources.encoder/projector均attempted=completed=8；binding.json中的prepare路径/SHA另核。report.artifact_sha256绑定8个NPY及events/binding。

report.rows固定顺序root0 GN prefix5/terminal10、root0 D2P prefix5/terminal10、root1 GN prefix5/terminal10、root1 D2P prefix5/terminal10。每row有index/branch/root_slot/arm/role/task_id/task_export/task_sha256/status/latent/metrics/provenance。latent为`{file: image_XX.native192.npy, sha256, dtype: float32, shape: [192]}`，file相对report目录；先hash，allow_pickle=False，必须float32(192,)有限且不转换。task_export是原批准旧root task的绝对路径与外部SHA，task_id须精确相等。

observed-prefix suffix compiler只从四个role=prefix5的row取native192，按root_slot+arm+task_id配对批准task历史和已选第一包commands；raw action history必须来自对应已选动作，不置零、不按另一arm借用。terminal10仅用于本次预测—真实诊断，不作为suffix编译输入或候选选优标签；root/goal沿用task。无需打开provenance中的物理state、physical journal或certificate。本consumer不自动编译任何suffix，不改目标/权重，不为下游配置预算。当前会话无跨线程消息工具，此段作为可直接转交的接口说明。
