# actual cp5 原 incumbent：十个独立 raw 坐标 FD164

独占 source/tests/本文；不改任何冻结源码或 MAIN 文档。本工作位仅实现、RAM 与只读 saved metadata，MAIN 审读后 HOST CUDA。旧 scalar_fd 是另一锚点/三h/无baseline的实验；旧 observed packet2dim FD也不是本轮十个独立 raw 坐标。

## 已固定输入与算法

输入是 `agent_nonlinear_suffix_observed_recompile_20261028_v2` 的实际 prepare/report/完整 strict receipt，三份实际SHA都从CLI提供并重建；只取四槽各自 `numeric.iterates[0]` / `request.incumbent`，不是优化后 winner。原known2 roots×off/on全保留，无fresh。actual cp5 context3、incoming raw10、goal192/identityQ/model/normalizer/prefix5不变，baseline必须逐位复现原 observed iterate0 全context与float64Q，不一致在第一个probe后failclosed，不运行该槽后续probe。

每槽 baseline1 + h=.01/.001 × 10 time-major raw5×2坐标 × sign(-1,+1) =41，共164predict/80central pairs。扰动唯一字面公式 `float32(float64(anchor_coordinate)+sign*h)`；其余9坐标bit不变，保留signed zero。没有tanh、inverse、logits扰动、clip或动作替换。执行前必须全部164 raw suffix与原prefix在[-1,1]合法且实际跨度不坍缩；否则prepare保留全manifest、标blocked，run在创建output/加载模型之前拒绝全批。不运行合法子集。

这是ambient actuator-box诊断，可能超原logit L2 sphere；不因sphere拒绝，不投影、不宣称原planner可部署候选。原solver/cast/selected_domain见证继承供解释，不修改历史81/168合法cast边界。原模型与参数全冻结，绝无训练、Qfit或选择动作。

## 纯 manifest ABI：Pasteur/Bohr

```python
from experiments.spectral_reachability import agent_observed_raw_coordinate_fd_20261028_v1 as fd
jobs = fd.plan_jobs(prepared_dict, report_dict, strict_receipt_dict, authority)
# authority: prepared, prepare_sha256, output, report_sha256,
#            receipt, receipt_sha256；路径均absolute、SHA均实际64hex
```

纯函数不读盘、不运行模型、不签磁盘authority；生产build先完整`source.consume()`并逐字节比较saved receipt，再调用它。固定list164，slot-major，每槽baseline→h→coord→sign(-1,+1)。每job含 `slot/index/local_index/id/coord/h/sign/raw_suffix5/raw_bits/prefix5/prefix_bits/full10/task/task_id/observed_context/observed_raw10/root_raw10/provenance/job_sha256`。baseline的coord/h为null、sign=0，id例如`s0_baseline`；probe例如`s0_h0_c0_minus`。`job_sha256`使用原producer canonical/digest，不混用物理codec。`actual_delta/actual_pair_spacing`以float64保存两个literal float32 raw差；`nominal_delta/nominal_pair_spacing`另栏保留。每项`legal/missing_reason/deployment_allowed=False/physical_release=False`，无物理标签。

全task（含192×192 identity projection）仍逐job直给。为不扩32MiB metadata cap，`original_AD_context`与`original_iterate0_context`仅每槽baseline job非null，其余job为null；所有job的`baseline_reference_id`指向该槽baseline。其余完整task/observed-context/raw/goal/provenance仍同槽相同。原实际`original_AD_raw_gradient`、`original_AD_native_loss`继承（gradient index0，即原gradient_1），不重算新AD。下游须验证baseline-reference规则，不能要求这两个null字段等于baseline非null context。prepare的`jobs`就是此manifest，`manifest_sha256=digest(jobs)`，全4 `points`为紧凑模型执行视图。

物理侧必须另核root/warmup/prefix/cp5 state+snapshot+原51图像/公开返回，不可仅靠模型manifest视作同位；frame/state/label永不进入本模型selector。本模块无物理执行权限。Bohr的终态encoder应按原`slot/id/job_sha256/full10/provenance`和实际physical authority一一配对全部164，不能按模型/物理响应挑图；不得用旧packet2dim或优化后suffix冒充本manifest。

## Q、vector与AD比较

原scalar_fd的policy-free Backend提供同一native float32 H1 forward，保存所有module IO/完整native3与shifted history；仅末行是future5 endpoint。原check_objective复核endpoint/goal/residual/square/native float32 scalar buffers及保守reduction roundoff bound。每probe保存192 endpoint、native float32 mean goalQ、原selector float64 endpoint meanQ。原生scalar buffer运输shape[1]来自原codec，未修改codec。

主差分：`(Qplus_native32-Qminus_native32)/actual_float32_span`；向量192每轴同分母。float64 endpointQ差分单列，nominal2h分母结果仅diagnostic；两h分别保留、不挑best h。全部40signed/20pair每槽保存；每h列10维FD与继承AD raw_gradient0的L2/Linf/relativeL2/cosine、每轴signed/abs error。zero向量cosine、AD norm0的relative error为null，不加epsilon、不以阈值宣称导数“科学通过”。baseline也保存原AD forward endpoint/native loss对照，不能把原梯度冒称本轮新计算。原iterate0 exact gate与AD forward baseline诊断分栏。

## Authority、实际账与限制

固定source pin：observed-v2 `e66bac861fce455ab6bb7a0cf12ba1644258c6c2e9cb6f2905973ac146e33567`；旧scalar_fd `a56e403c0d5b7a904b502c8044aaa4c575b667631fb98992716f1e13acd79639`。build绑定全部上游source/model/prepare/artifacts/receipt与本三文件，并将OUTER `/home/gbk/code/LeWorldModel/docs/D2P_RAW_COORDINATE_CAUSAL_PROTOCOL_MAIN_20261028_v1.md`真实SHA收入bindings；不改Main协议。每次verify/consume完整重建上游，不是status-only或暗中复用receipt。

固定164predict/action_encoder/predictor/pred_proj、492 temporal rows；新AD/encoder/physics/训练为0。原72predict/32reverse及更早费用分栏继承，FLOPs未知null。no_grad+冻结参数，MATH SDP，TF32/autocast off，CUDA-only，无CPU推理fallback。复用原NativeJournal/资源Ledger和F.audit_native+scalar_fd.audit_native_bits，逐probe durable intent→全部native start/resource/数组/return/end→objective/return→下一intent。全事件/字段/跨phase时序验证，未知事件或新AD不被过滤后通过。完整census/hashchain只是一层，必须同时重算数组、动作、目标、Q与资源。

沿原caps：wall1800s（从verify前起，边界watchdog非抢占中止）、output128MiB、final reserve16MiB、disk floor200MiB、CUDA allocated3GiB/reserved4GiB。prepare/receipt metadata32MiB上限，不因重复projection扩容。失败保留4槽/164请求、已付账、failed_partial/未到达null，不重试补槽；partial consumer保留已验panel但complete=false，不造科学通过。baseline不一致也保存实际已用1forward。输出报告、strict receipt均不授权物理/P1/fresh/C2。

## MAIN命令

在 `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
.venv/bin/python -m pytest -q tests/test_agent_observed_raw_coordinate_fd_20261028_v1.py tests/test_agent_nonlinear_suffix_scalar_fd_20261028_v1.py tests/test_agent_nonlinear_suffix_observed_recompile_20261028_v2.py

.venv/bin/python -m experiments.spectral_reachability.agent_observed_raw_coordinate_fd_20261028_v1 prepare \
  --observed-output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_nonlinear_suffix_observed_recompile_20261028_v2_cuda_MAIN01 \
  --observed-report-sha 3de58e380acbef66269e69150cab7bca312604b7df88a0f1ae5663764d691b45 \
  --observed-prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_nonlinear_suffix_observed_recompile_20261028_v2_prepare_MAIN01.json \
  --observed-prepare-sha 9aa3d63db85857b80c8a614a00a5a7b22c77c26e688e149970a83825c0d74411 \
  --observed-receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_nonlinear_suffix_observed_recompile_20261028_v2_consume_MAIN01.json \
  --observed-receipt-sha 09d617dbece12ce3ff2ca7246390e433454b1d8ac29042ab07eadd2be75238ec \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_observed_raw_coordinate_fd_20261028_v1_prepare_MAIN01.json

.venv/bin/python -m experiments.spectral_reachability.agent_observed_raw_coordinate_fd_20261028_v1 verify --prepared ABS_PREPARED --prepare-sha ACTUAL_PREPARE_SHA
.venv/bin/python -m experiments.spectral_reachability.agent_observed_raw_coordinate_fd_20261028_v1 run --prepared ABS_PREPARED --prepare-sha ACTUAL_PREPARE_SHA --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_observed_raw_coordinate_fd_20261028_v1_cuda_MAIN01 --authorization MAIN_OBSERVED_RAW164_FD_HOST_CUDA_ONLY
.venv/bin/python -m experiments.spectral_reachability.agent_observed_raw_coordinate_fd_20261028_v1 consume --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_observed_raw_coordinate_fd_20261028_v1_cuda_MAIN01 --report-sha ACTUAL_REPORT_SHA --receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_observed_raw_coordinate_fd_20261028_v1_consume_MAIN01.json
```

测试为RAM/native-ABI fixture与只读实际metadata接缝，不是神经/物理证据。实际CUDA、物理与终态编码尚待MAIN授权/执行；无新科学结论。任何真实下游ABI修复在封存后另版本，不改旧结果。
