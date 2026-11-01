# P4 prefix-specific CUDA AD response producer

本 NEW stem 独占源码、tests、本文；保持冻结 runtime/sourcepack/旧结果不变。prepare、verify、inspect 仅 saved-data；只有 MAIN 审核后独立进程运行 run。没有物理调用、下载、训练、拟合、重新编码或初始J复用。

## 输入与算法

默认输入是已封存 root0 sourcepack `results/p4_feedback_root_sourcepack_20261027_v1_root0_prepare01`，prepare SHA `4de418b03fb7e4a664d9e5d04d8c78b7f7ac467e89f04677329b463fb6489fe2`。完整 sourcepack.verify 重新核对原P3/NEW17/teacherforcing/首批排除来源；actual context 必须是已保存的真实 prefix192、两个较早latent及原动作历史。imagined使用相同因果历史的原预测末latent；两臂各自独立重新AD编译，禁止共享J或以teacherforcing输出代替候选预测。

每臂固定H1、全3帧latent上下文、两个between-state action tokens：从已封存本地原生模型的 action_encoder→predict，调用原 `normalize_live_packet(raw, mean=sourcepack.mean, std=sourcepack.std)`，不多除100。`pred_proj` 属于原生predict，合法；视觉encoder和existing视觉projector严格禁止。全部神经前向/AD为cuda:0，float32输入、MATH SDPA、TF32关闭；权重加载/字节核验在CPU不是CPU神经推理。模型eval且参数requires_grad=False，仅动作logits求导。

在logits=0先独立nominal，再运行 `torch.autograd.functional.jacobian(...strict=True, vectorize=False, create_graph=False, strategy='reverse-mode')`，获得真实native192×10 J。它的fresh forward完整context必须等于nominal；两者、J、bank均持久化。原sealed `spectral_bank` 用原Q与goal构造十候选：zero、bounded ridge progress、两半径×两模式×正负；统一logit球半径和damping，不作实际效果过滤或补样。所有十候选都做真实非线性predict并按原Q goal MSE评分，稳定index打破平局；优胜者从原请求重新建tensor独立重放，完整滚动context及动作history精确相等才输出response。

这是NEW H1响应定义，不是旧P3 H2/H3资源等价比较；spectral数值退化或重复候选仍保留十槽，不据此宣称十个独立效果。

## 资源与失败保存

固定工作量（每臂/双臂）：13/26 predict samples，13/26 action_encoder calls；39/78 processed context tokens；1/2 Jacobian API；192/384数学标量VJP方向。13次forward=1nominal+1AD forward+10候选+1独立replay，没有额外诊断/FD/重试。serial reverse实现预期每臂192 grad API与192 Python `_engine_run_backward`入口；后者不是GPU kernel数量或C++引擎内部步骤数。

prepare **必须显式提供** grad API与engine API每臂cap（至少192）、CUDA peak allocated/reserved bytes cap、墙钟秒cap。没有猜测默认显存/时限或自动扩容；即使MAIN给更宽API cap，代码也不允许超过192次有用reverse入口。成功必须核验真实attempted=completed和精确工作量，不能只检查未超cap。

每个模块/API attempt前同步fsync journal，返回后CUDA synchronize并核对时间/allocated及reserved峰值，然后才记completed；异常attempt不退款，后续两臂/候选停止。模块仅允许native batch1/time3形状、固定phase顺序；encode/视觉encoder/projector/autograd.backward入口禁止。`functional.jacobian`、`autograd.grad`及当前PyTorch的`_engine_run_backward`两处Python别名封存并拦截，版本不符/工作数不符fail-closed。该有界driver不声称能防御任意恶意C++扩展绕过Python计量。

显存使用allocator fraction补充限制reserved cap；峰值在同步边界核对（allocated不是逐allocation硬拦截）。POSIX alarm与边界检查限制执行时长；CUDA kernel不可由Python可靠抢占，MAIN仍应给独立进程设外部watchdog。超时/OOM/计量不符均不得写成功。最终模型/源核验及错误保存不受alarm中断，不计作额外神经工作；receipt同时报告执行时长与含保存的总时长。

每阶段意图、native nominal/AD-forward/J/候选/重放即时独占落盘。receipt保存partial rows、attempt/error账、blocked、峰值、model前后hash、源复核与所有输出文件SHA。若持久化本身失败，保留已有journal/文件且进程失败，不承诺磁盘/硬件失效时一定能生成最终receipt。输出总cap12MiB，预留1MiB最终报告，磁盘保留200MiB；源只封存小文件，旧大protocol/checkpoint只引用。输出目录禁止覆盖或自动重试。

## MAIN命令

在 `/home/gbk/code/LeWorldModel/le-wm-main`，先纯mock测试及只读实源核验：

```bash
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p4_feedback_root_cuda_response_20261027_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.p4_feedback_root_cuda_response_20261027_v1 inspect
```

MAIN已审查并固定本次工程cap：wall600s、allocated3GiB（3221225472bytes）、reserved4GiB（4294967296bytes）、grad/engine各192次每臂。这是授权上限，不是已测成功结果；serial实现及资源仍须MAIN实际HOST验证：

```bash
.venv/bin/python -B -m experiments.spectral_reachability.p4_feedback_root_cuda_response_20261027_v1 prepare \
  --output results/p4_feedback_root_cuda_response_20261027_v1_prepare_main01 \
  --wall-seconds 600 \
  --allocated-bytes 3221225472 \
  --reserved-bytes 4294967296 \
  --grad-api-per-arm 192 --engine-api-per-arm 192
```

保留上述输出的真实prepare SHA到 `P4_RESPONSE_PREPARE_SHA`，再执行：

```bash
.venv/bin/python -B -m experiments.spectral_reachability.p4_feedback_root_cuda_response_20261027_v1 verify \
  --prepared results/p4_feedback_root_cuda_response_20261027_v1_prepare_main01 \
  --prepare-sha "${P4_RESPONSE_PREPARE_SHA:?retain printed prepare SHA}"

# 仅MAIN审核后运行；本任务不执行此命令。
.venv/bin/python -B -m experiments.spectral_reachability.p4_feedback_root_cuda_response_20261027_v1 run \
  --prepared results/p4_feedback_root_cuda_response_20261027_v1_prepare_main01 \
  --prepare-sha "${P4_RESPONSE_PREPARE_SHA:?retain printed prepare SHA}" \
  --output results/p4_feedback_root_cuda_response_20261027_v1_cuda_main01
```

本次只交付代码、mock和saved源验证；MAIN给定cap后可按上方命令prepare。本任务不代MAIN生成实际prepare、不运行神经/physics，无真实CUDA成功宣称。新增43项纯mock测试，联合sourcepack35项、runtime54项、second-batch preconditions32项，共164项；默认root0输入inspect通过。显式小型source closure34文件/小于1MiB，单文件均小于1MiB；这不是实际模型推理或显存/时限验证。

MAIN在首次prepare/source seal之前发现normalization含source/source_sha256/training_provenance_verified元数据，不能整体kwargs传给只收mean/std的函数。未封存v1已修复Native.transition/replay两条路径为显式mean/std；新增两项numpy tensor mock直接执行Native方法并传入完整真实pack字段布局，严格keyword-only normalizer验证不泄漏元数据。旧源码SHA已作废，以此次修复后交付SHA为准；未修改任何既有sealed来源或产物。此为工程接口缺陷修复，不是理论或任务负结果。

## 下游边界

成功response也只是known-root saved-prefix条件化编译，不是第二独立确认、live反馈或任务收益；不能直接喂给冻结runtime v1。未来NEW闭环driver仍须真实reset/prefix身份绑定、逐步guard、每轮实际step5/10/15反馈后重新编译、选择先封存再执行，step20只能评价。P1/P4、independent_confirmation、runtime_admissible始终false，保留teacherforcing改善MSE但旧Q排序仍反的历史。
