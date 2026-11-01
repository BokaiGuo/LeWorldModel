# P2 factorial：固定32缺失forward的实际执行接线

新增同stem source/tests/innerdoc；旧factorial producer、recompile/FD源码与MAIN结果完全不改。worker不运行CUDA/神经CPU/physics/network。此处只补原预声明zero-anchor四alpha候选，不再生成/筛选根、改goal/Q或训练。

## 接受的固定authority及边界

只接受MAIN已完整重建的 `results/agent_p2_safeguard_factorial_20261028_v1_prepare_MAIN01.json`，SHA **`9ad5980d7943f11dc5e6d53c285f37dc4a86e69620d136036bcdd656d3e9d469`**；其它SHA在读文件前拒绝。原producer源SHA `0af90ec7383c52f5061a395159dfeecd562a2c5a45be788abb97878d7bb97fc3`、FD源SHA `4329b05f6c8d5d11b5eb89d6a22cf7d019120f589fdba3bf7edc46de7cb4dc7f`固定核验。

按MAIN允许的authority复用：读取原prepare、核其实际SHA，重新hash全部 `p.bindings`（原report、数组、模型/normalizer及consumer等源码）；调用本模块 `material(p)` 重算literal ABI，再绑定本新三件套。**不调用原source.verify/consume_compile重复6分钟闭包**。新prepare、report、receipt明确 `full_upstream_rebuilt_here=false` 和 `authority=pinned MAIN actual complete factorial prepare and every upstream binding SHA`。这不是声称本runner重新独立consume了所有上游科学证据；其合法authority是MAIN刚验收的不可变实际prepare。

新prepare是可执行wrapper闭包，原status=blocked_missing_model_candidates只表示缺32模型预测，不是科学负结果。新run/consume前后均重hash全部bindings与实际准备文件；源码/绑定漂移必须失败。

## 实际forward与完整分母

只执行原requests：branch-major原8 × alpha `[1,.5,.25,.125]`，共32，每条恰好1次原 `F.d2p.TorchBackend.step`。无去重、额外replay、AD/J/SVD、图像或物理；不存在CPU fallback。原rawfloat32[5,2]逐bit相同，不重新tanh、clip、inverse或强制cast修复。原prepare已证明的logit/fulljoint/remaining半径合法性保持，material还核chart/raw/context/normalizer/goal与full3输入。

原FD `Modules`、`PredictAPI`、`NoAD`直接复用，不改globals。CUDA-only，frozen模型实际before/after hash一致；inference mode、TF32/autocast off、MATH SDP、batch1。完整action_encoder→predictor→pred_proj输入/输出NPY持久化，AE输入(1,3,10)，native latent输入/输出(1,3,192)，投影(3,192)，完整returned shifted latent(3,192)/tokens(2,10)，最后一行必须是同一实际native endpoint。predict API为inclusive单独轴，不能与内模块相加。

新Forbidden scope阻止NumPy/torch SVD及model encoder/projector调用；NoAD阻止grad/Jacobian/backward/JVP/Hessian等。禁止操作在执行前写拒绝事件，任何partial报告不能strict通过。原生模型无新visual调用，原context来自已编码actual-mid，而非state输入。

## Durable journal / strict

外层run_intent、每request_intent/request_return以及native evaluation intent/input-save/module/API-return完整写入fsync journal。链index只作运输序号，payload使用独立 `request_index`。serializer必须是 **`kernel.raw`**，digest必须是 **`kernel.source.digest`**，与原 `physical.journal` 完全一致；不能混用本模块compact metadata serializer或physical.saved.digest。

consume先以原physical.journal验证全部canonical bytes/chain，再独立重建完整预期事件（包括index/previous_sha256/event_sha256）逐字段相等；不删空记录、不跳未知事件、不忽略未知payload字段。所有原full数组SHA/形状/dtype/因果native输入/AE连接/投影连接/返回history/Q逐一重算；固定32 rows、实际资源与API attempts/returns必须完整对账，report/selection/output全census闭合。仅hash正确但事件非法也拒绝。

所有32新增预测都通过数组和事件校验后，才调用冻结 `source.choose` 封存32cells：每branch zero/unguarded、zero/guarded、incumbent/unguarded、incumbent/guarded。所有64提案与8incumbent保留，旧32incumbent-anchor候选复用原预测；每个cost使用原goal/identity192重新算。任一missing不得封存子集。只传模型候选ABI，不传物理标签、success、coverage或native metrics。每cell新digest绑定原prefixbits+选中suffixbits、模型成本/context/goal身份，scope明确为actual-mid suffix5，不冒充root→terminal全图Q。

输出cells是模型选择见证，不是新物理执行。MAIN已确认incumbent侧selectedindices `[2,1,3,2,3,1,1,1]`，均Q<old，因此该侧有/无guard动作相同；不能把已有7/8 XY改善归因于保旧机制。这个观察不参与本新zero候选生成或选择，也不是模型排名可靠性/效能保证。

## 预算与失败

新实际最大predict/AE/pred_proj各32，inclusive predict API32，新AD/SVD/visual/physics均0；继承两版资源原账单列。每anchor/branch同4候选pool由guard off/on共享，不重复计费；zero-anchor新增32，incumbent-anchor新增0、继承费用非0；guard额外forward0。总FLOPs=null，不声称旧算法或两个anchor等FLOPs。planned和actual资源分栏，unknown未来物理费用不填0。

whole-stage900s从verify前计时，64MiB output、2MiB final reserve、200MiB disk floor。时间为操作边界检查，不是抢占native call的硬watchdog。失败不重试、不补根，保32 rows以及已用module/API attempts/returns，未开始保not_started。超时后最终row/report与run_finished使用final reserve路径，不再触发normal wall拒写；但仍遵守cap/floor，严重磁盘I/O失败不能保证落盘。任何失败status=partial_or_failed、selection不获strict放行。验证前失败没有神经计算，不制造完整receipt。

## MAIN准确命令

cwd `/home/gbk/code/LeWorldModel/le-wm-main`，所有输出为不存在的新路径；新prepare/report SHA只填实际返回值。

```bash
.venv/bin/python -m pytest -q tests/test_agent_p2_safeguard_factorial_cuda_20261028_v1.py tests/test_agent_p2_safeguard_factorial_20261028_v1.py
.venv/bin/python -m experiments.spectral_reachability.agent_p2_safeguard_factorial_cuda_20261028_v1 prepare --factorial-prepared results/agent_p2_safeguard_factorial_20261028_v1_prepare_MAIN01.json --factorial-sha 9ad5980d7943f11dc5e6d53c285f37dc4a86e69620d136036bcdd656d3e9d469 --output results/agent_p2_safeguard_factorial_cuda_20261028_v1_prepare_MAIN01.json
.venv/bin/python -m experiments.spectral_reachability.agent_p2_safeguard_factorial_cuda_20261028_v1 verify --prepared results/agent_p2_safeguard_factorial_cuda_20261028_v1_prepare_MAIN01.json --prepare-sha <ACTUAL_RUNTIME_PREPARE_SHA>
# 仅MAIN审读后主机CUDA：
.venv/bin/python -m experiments.spectral_reachability.agent_p2_safeguard_factorial_cuda_20261028_v1 run --prepared results/agent_p2_safeguard_factorial_cuda_20261028_v1_prepare_MAIN01.json --prepare-sha <ACTUAL_RUNTIME_PREPARE_SHA> --output results/agent_p2_safeguard_factorial_cuda_20261028_v1_cuda_MAIN01 --authorize-host-cuda
.venv/bin/python -m experiments.spectral_reachability.agent_p2_safeguard_factorial_cuda_20261028_v1 consume --output results/agent_p2_safeguard_factorial_cuda_20261028_v1_cuda_MAIN01 --report-sha <ACTUAL_RUNTIME_REPORT_SHA> --receipt results/agent_p2_safeguard_factorial_cuda_20261028_v1_consume_MAIN01.json
```

CLI只stdout小路径/SHA，完整32cell写独占receipt，不倾倒计划。没有physics CLI，也没有自动放行。

## RAM验证与未运行项

首轮持久化测试实际捕获payload index撞链index、随后捕获compact serializer与原native journal ABI不匹配；这是本新runner开发错误，不是历史P1原因，失败记录保留。修复后原44项通过；追加真实run/32callback/全hooks/consume的NumPy-double接线时，fixture漏支持always_call参数，修fixture而不改原kernel。最终worker新tests **46 passed in6.58s**；MAIN独立runner+factorial联合 **57 passed in6.98s**（MAIN回报），未放宽旧kernel或hash验证。三件套至此冻结，SHA由交付消息给出。

覆盖全32持久化NPY/row/journal/cell验收、合法原生hashchain、重hash后的未知payload拒绝、篡改数组/成本/输出census/selection/分母/资源拒绝、绑定和源SHA漂移拒绝、NoAD/SVD/visual hook恢复、超时仍写partial分母和终态日志。实际MAIN metadata接缝只读固定prepare并运行material；没有重复上游consumer闭包。所有计算是NumPy RAM mock/metadata，不是CPU神经实验、不是科学证据。worker未生产prepare/verify、未运行GPU/physics；实际CUDA数值、计量、strict结果仍待MAIN。

全部P1/P2 admission、fresh和physical_release=false，仅2known roots calibration，不报告科学通过。交付三件套冻结后修复必须用新版本。
