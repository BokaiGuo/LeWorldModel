# Teacher library40：实际 terminal RGB → 冻结 native192 adapter

独占交付：`experiments/spectral_reachability/agent_p1_teacher_library40_encode_20261028_v1.py`、`tests/test_agent_p1_teacher_library40_encode_20261028_v1.py`及本文。不改旧producer/consumer、P4 schema、MAIN结果或共享总账。不派生代理、不访问预留四fresh根。此交付只有实现与RAM回归，未执行实际prepare、神经、物理、网络或完整saved consume。MAIN审读后决定是否另行授权partial39 CUDA；测试不是科学证据。现有个人goal工具仍显示旧任务blocked，不覆盖/完成该旧goal；本次有界交付条件为这三件套、RAM验证、命令与限制。

## 真实接线与固定分母

已有可复用入口是冻结teacher的`rgb/load_array`、`enc.frozen/tensor_hash`、原`preflight_pusht_d2p_p0._image_tensor/image_transform`，以及overlay的`NativeTrace.save/traced_scope/visual_bridge`。原trace的16/8次上限不能承载39张；本文件仅新增自己的39次hook，不改旧globals或旧算法。实际运行调用原`load_compatible_pretrained`和`model.encode({'pixels':x})['emb']`，不是callback-only框架。encoder与projector的真实forward entry/return hook持久化完整输入、输出、scope、原精度数组。

固定40顺序：`(root0,off),(root0,on),(root1,off),(root1,on)` × `Z,S` × `zero,uZ,-uZ,uS,-uS`。全部身份由原job、sealed plan、physical row、actual journal及checkpoint join，不按Q、native结果、图片是否相同筛选或去重。39张可用图片每张独立batch1一次encoder+projector；不重新编码goal，不predict/AE/pred_proj/AD，不执行环境或render。

源内固定以下实际SHA，prepare重建并核所有原artifact、journal、物理/编译prepare中的source/输入闭包，以及原encoder模型config/weights；不能只核日志hash：

| 材料 | SHA256 |
| --- | --- |
| physical MAIN02 report | `464656755bb83cec14b7acdd457641592b11be6d335f285166687df15368bc01` |
| physical partial fullstrict consume MAIN02 | `44acd71c60c8edc7fbfa196c2f46f7d8692ef487ec0c3a01b6b8f7e5071e9212` |
| compiler CUDA MAIN01 report | `d5ab99d55bacd73c8d9cf0fb774819bb46af63100053198858e639d8fa746738` |
| compiler strict consume MAIN01 | `d82109eb8de582e59a4f0485307c01c9b91f9f84551c218c61cca4da24335213` |
| native saved audit MAIN02 | `3dfc301c40471e0620caf8660baf4d8ad6534f25055e546aadc7466f61e79b78` |
| RGB protocol v1 | `0b9148dba5211723e6fb670658a71acd9de73487c2140affa78fbc28f2ace8aa` |
| RGB correction v2 | `06bba2e5af4992003549a6ea3817d40162fc8179617753d69cf67622024049f9` |

这些是已有材料身份，不是本adapter已完成实际编码的证明。物理partial严格回执是显式接受的partial证据，不伪称complete；原physical全部40 suffix5真实返回及800public/8160physics/2638render费用继承单列，原编译费用单列。新任务不得宣称等总算力或等FLOPs。

## 缺失与不放行

index12=`root0/on,Z,-uZ`，原canvas guard因agent_y=`514.7195833901377`越512拒绝；其off incident尚有8个canonical blue pixels。该像素计数不放宽canvas guard。final-on/checkpoint10/end_state缺失；本adapter只保留incident证据，不编码off冒on，不补图、不重渲染、不用旧帧或预测代替。该槽actual latent/Q/model-error为null。native success是原全部40真实最后返回的独立字段，不能拿其值补缺失latent，也不能将object-only投影解释为完整PushT成功或失败。

默认`full40`的manifest/prepare及任何full40准备输入在加载模型、创建执行output之前拒绝。当前只有显式`partial_observed_only`可prepare；真实run还须`--authorize-main-cuda-partial39`。partial成功状态严格为`partial39_terminals_encoded_full40_blocked_not_admitted`；40槽、39eligible、missing=[12]和actual attempted/completed并列。若再缺任何项则拒绝，不自动缩分母。known roots=2、fresh=0、P1=false，不生成库winner/rank、不科学放行。

## 数值、预算和durability

固定checkpoint224/patch14，encoder `[1,257,192]`，CLS/projector `[1,192]`，latent192；197反向拒绝。原ImageNet mean/std和Resize224，真实CUDA预处理逐元素与独立NumPy算术核对，保完整preprocessed与模块输入/输出。CUDA-only、MATH attention、TF32=false、autocast=false，模型冻结前后tensor SHA相同；CPU仅搬运保存数组和saved算术，不作为神经fallback。

目标和predicted三帧latent history从原compiler数组三层SHA读入，和原同condition anchor/q/goal/context绑定。原Q严格为`mean((identity192 @ (native.astype(float64)-original_goal192))**2)`，192维mean，不重拟合、不更改归一化；adapter在identity合同下直接逐元素平方mean，并核原predicted_Q与原receipt完全相等。另列predicted-vs-actual latent MSE，不称物理收益。

预算：prepare64MiB，output64MiB，final reserve16MiB，report8MiB，floor200MiB，wall600s。journal实施更严格8MiB（低于协议32MiB上限）。prepare提前逐图计算lossless pixels/preprocessed/encoder-input的实际gzip大小，未知native tensor用保守gzip上界，加8MiB+64KiB元数据，须整体低于48MiB pre-final空间；不丢trace、不降精度换通过。实际每写入再检查bytes/floor；部分写入在finally按fstat真实占用计费，fsync/排他创建/终态inode+size+bytes+census。39逐项intent→encoder entry/return→projector entry/return→image return严格对账，无重试。失败保40槽与已用账；strict consume不收partial/pending/失败目录，不按中途文件推断完成。

数组无pickle，gzip NPY保持uint8/float32，压缩file、原NPY、原array三个SHA。consume重建prepare全闭包、原PNG像素、预处理、257/CLS/projector/latent接线、精确Q/MSE、全部78 module与234 journal事件的chronology和费用，校验全目录artifact census及输入前后hash。报告包含来源计量，不以RAM mock receipt冒充真实执行。

## MAIN命令（在inner repo执行；本worker未运行）

将下列`PREP_SHA`/`REPORT_SHA`替换为实际终态文件的外部SHA，不预造未来SHA。output父目录须已存在，名字必须属本stem，新文件/目录不得覆盖。

```bash
.venv/bin/python -B -m experiments.spectral_reachability.agent_p1_teacher_library40_encode_20261028_v1 prepare --mode partial_observed_only --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p1_teacher_library40_encode_20261028_v1_prepare_MAIN01.json
.venv/bin/python -B -m experiments.spectral_reachability.agent_p1_teacher_library40_encode_20261028_v1 verify --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p1_teacher_library40_encode_20261028_v1_prepare_MAIN01.json --prepare-sha PREP_SHA
.venv/bin/python -B -m experiments.spectral_reachability.agent_p1_teacher_library40_encode_20261028_v1 run --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p1_teacher_library40_encode_20261028_v1_prepare_MAIN01.json --prepare-sha PREP_SHA --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p1_teacher_library40_encode_20261028_v1_cuda_MAIN01 --authorize-main-cuda-partial39
.venv/bin/python -B -m experiments.spectral_reachability.agent_p1_teacher_library40_encode_20261028_v1 consume --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p1_teacher_library40_encode_20261028_v1_cuda_MAIN01 --report-sha REPORT_SHA
```

manifest/prepare/verify/consume只做saved读取、hash及算术；prepare可能因真实容量或来源绑定不满足而拒绝，不能先创建成功receipt。run唯一涉及神经的入口，仅MAIN主机CUDA审读授权后执行。stdout输出JSON，consume本身不写旧目录；需保存回执时由MAIN另给新输出位置。

RAM回归命令：

```bash
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p1_teacher_library40_encode_20261028_v1.py
```

测试用纯NumPy、BytesIO及fake modules在trace.save边界模拟张量，不导入torch、不调用CUDA/physics、不读取预留根。覆盖40身份/跨condition/目标/guard/原费用/plan与job seal、缺失null、重复图片39真实dispatch逻辑、首错不重试、257正向/197反向、禁止predictor/第40次encode、全精度三SHA、journal链和entry/return顺序、部分write计费、容量拒绝、原Q与MSE。

实际本worker运行：本文件单测72 passed（5.75s）；此前与原overlay、teacher共同库两组RAM测试联合154 passed（20.42s，当时本文件68项，随后新增4项pre-model failure/finally测试再单独全跑72项）。CLI `--help`退出0。未运行真实prepare/verify/run/consume；没有生成adapter实验结果或假receipt。以上测试次数不代表图像编码或科学样本数。

未完成项明确：MAIN最终源码审读、实际saved prepare/verify的闭包/容量回归、显式partial39 CUDA授权与执行、真实module trace及严格consume。没有完整40可用RGB，无法交完整40科学比较；本次实现不能修复这项缺失，也不授予P1或fresh确认资格。冻结后任何修复必须新版本，不改这三件套或MAIN prepare闭包。
