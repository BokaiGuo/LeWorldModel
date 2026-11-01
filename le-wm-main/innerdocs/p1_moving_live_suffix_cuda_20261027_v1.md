# Moving live suffix CUDA core v1

实现交付；仅RAM/mock验证，未prepare、未运行CUDA/神经/物理。MAIN独占审读、实际执行与准入。固定旧moving两根，GN/D2P prefix不构成独立任务；`P1_admission=false`、`P4_admission=false`。只新增本stem三文件，不改sealed GRASP/gradient/旧suffix/共享文档。

## 接口与driver合同

```python
compile_suffix(request, artifact_sink, *, model, device, check_limits) -> response
verify_suffix_seal(request, response) -> raw5
```

显式传driver已加载原冻结eval CUDA model与明确device（如`torch.device('cuda:0')`）；无loader/CLI/物理driver，绝不重复加载权重。core不编码RGB，不接`encode_observation`；driver先完成自己的新RGB编码。`artifact_sink.output`为已存在、本次编译独占的输出目录；`_write(name,bytes)`须排他写入+文件/目录fsync，`event(dict)`须durable flush+fsync并在破损后拒绝重试。可由driver适配已有store，禁止两个编译共用同名文件namespace。core不close driver journal。`check_limits()`无参数，超全局wall/allocated/reserved/累计API/输出cap时抛错；本core每次模块/AD/math/文件/journal事件前后走sink检查，不能用空callback进行实际准入。core额外用真实`shutil.disk_usage(output).free`检查200MiB floor，单编译8MiB cap（NPY、packet、receipt、partial及journal按未压缩JSON保守计费），普通写保留256KiB final预算。建议沿用driver独立600s/allocated3GiB/reserved4GiB cap，不将其伪称本core测得的上限。

进程级nonblocking锁防止core并发patch torch AD；driver须保证同进程同期没有其他神经/AD调用（锁不控制外部代码）。冻结model前后tensor hash；MATH SDP、TF32off、autocastoff、enable_grad作用域，退出恢复caller设置。禁止外层inference_mode，允许外层no_grad但core显式enable_grad。无CPU neural fallback。preflight/device错误可直接抛ValueError；进入执行后错误抛`SuffixFailure`，`.receipt`保留first_error及secondary/已发生attempt。broken sink不重试、无动作发布。caller必须捕获并持久记录partial；禁止从失败目录中提取任何动作替代成功返回。

## 最小request（全部字段白名单）

```python
{
 'schema': 'p1_moving_live_suffix_cuda_20261027_v1.packet',
 'mode': 'I' or 'R',
 'context3': {'layout':'states_with_between_state_actions_v1',
   'context_length':3,'latent_dim':192,'action_dim':10,
   'latent_history': float32_literal_3x192,
   'action_history': float32_literal_2x10},
 'actualrawhistory10': float32_literal_10x2,
 'oldrawsuffix5': float32_literal_5x2,
 'goal': float32_literal_192,
 'identityQ': {'kind':'identity','dimension':192},
 'modelbinding': modelbinding(),
 'episode_observation_seal': seal,
}
```

JSON用Python list/原float32提升的float字面值，不能先四舍五入。`actualrawhistory10=原warmup末5+本次真正执行的prefix5`，前5固定float32 `[0,.1]`；不zero填充，不tanh/clip/除100。两历史token严格复算float32 public mean `[-.0087,.0068]`、std `[.2019,.2002]`，training provenance仍unknown。`oldrawsuffix5`须在原atanh球中，仅诊断。模型binding恰为`tensor_sha256/model_files/action_mean/action_std/training_provenance/source_closure`；helper `modelbinding()`只读源码及常量，供driver在开始episode前封存。source_closure含静态本地imports、runtime backend/torch AD源码和三文件，执行前后完整比较，不读task/sourcepack/results/witness，不重新hash大checkpoint文件；原model_files SHA为原封存身份，实际加载model以tensor SHA验证。

seal恰含以下字段（SHA均64位小写）：

```text
episode_id source_slot prefix_arm mode parent_task_id task_export_sha256
parent_selection_sha256 step remaining_public_steps context_kind
context_sha256 rawhistory_sha256 oldsuffix_sha256 goal_sha256 Q_sha256
modelbinding_sha256 rgb_file_sha256 rgb_pixels_sha256 encoded_npy_sha256
encoded_latent_sha256 encoder_receipt_sha256 imagined_context_sha256
observed_event_index encoded_event_index seal_sha256
```

slot固定0/1，prefix_arm固定`line_search_GN/full10_conditional_D2P`；task_export SHA为原moving两task（源码常量）。step=5、remaining=5。episode_id为1..128字母数字/`_.-`。context_kind与mode一一对应：R observed、I imagined。`*_sha256`中context/rawhistory/oldsuffix/goal/Q/modelbinding及encoded_latent使用本module `digest(JSON值)`（排序紧凑JSON+换行）；PNG file/pixel、encoded NPY、encoder receipt为各自原字节SHA，不可混淆。seal_sha256用`digest(seal中除seal_sha256之外全部字段)`。

R：context末latent为本episode新observed latent，encoded_latent_sha256严格绑定该值；imagined_context_sha256=None。I：context只含原selected imagined prefix，imagined_context_sha256=digest(context3)；actual latent值不能进入packet，encoded_latent_sha256仅保留fresh guard编码的SHA身份，不与想象值冒称相同。两模式的两旧latent和goal仍须由driver核原task/plan接口。observed_event_index<encoded_event_index只验证请求的声明顺序，**不证明实际physics时间因果**。

允许输入仅上述字段；context/modelbinding/seal任意未知字段拒绝。禁止hidden state/pose/witness/outcome/terminal label、未来RGB、任意完整task/sourcepack。source路径由helper产生的完整闭包比较而非任意读请求路径。driver必须 independently admit原goal/parent selection/原两个历史latent/原imagined prefix、encode原PNG->NPY SHA及本episode时序；自签seal只是绑定，不能凭core成功admit旧runtime。core不读state做选择。

## 数学、API与固定预算

直接复用sealed `moving_history_observed_suffix_cuda_20261027_v1`的evaluate、native hooks、AD及bounded ridge；只新增窄bridge，不调用其build/run/verify/旧RGB结果loader。context保持3状态/2between-actions。每I/R固定以下五阶段：

1. zero raw nominal：1 forward。
2. 相同context零logit点fresh192×10 J：1 AD-forward。
3. old suffix非线性诊断：1 forward。
4. new ridge suffix非线性评价：1 forward。
5. new suffix独立完整context replay：1 forward，要求exact相同。

每compile：model.predict API5、predictor模块5、action_encoder5、pred_proj5；API和模块为同调用不同入口，不加计成10。J1、SVD1；functional.jacobian1、其内batched autograd.grad1，数学VJP方向192，非192次API，AD-forward已含在5内。新增encoder/projector/physics全0。ridge=.002，cap=.75√5，raw=tanh(float32 params)。先封存new raw再做old/new非线性评分，**无条件派发new ridge**，旧分更好也不fallback、不重试、不调参、不复用原J。O不进入core。完整主协议8个I/R编译合计40forward、8J/SVD、1536数学VJP；driver全实验physics/编码/原计划成本另账，不能称全pipeline等算力。

逐模块保留全部actual inputs/AE embeddings/predictor outputs/pred_proj完整三token数组，pred float32原值、J和ridge原精度、原literal输入、动作、stage/API/math attempt-return durable存储。输出receipt含source/model前后、artifact SHA、release_sha256；`verify_suffix_seal`只在完整成功、请求身份及同episode绑定一致时返回raw5。它是交接核验，不是独立磁盘artifact审计；driver负责验证实际保存字节与seal并在step6前durable握手。收到成功返回也不能省略driver全链`observe5<encode5<compile<seal<step6`、无reset/set_state、原goal不变与fresh逐步RGB guard。

## MAIN调用与验证

```python
# 仅MAIN实际同进程driver内；此文交付未运行：
try:
    response = suffix.compile_suffix(request, sink, model=model,
        device=device, check_limits=driver_limits.check)
    raw5 = suffix.verify_suffix_seal(request, response)
    # MAIN另核并封存episode握手，随后方可执行step6..10。
except suffix.SuffixFailure as exc:
    driver_record_partial(exc.receipt)  # 不dispatch，不fallback。
```

RAM/mock测试命令（Python根，禁cache写）：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_moving_live_suffix_cuda_20261027_v1.py
```

本文件58项RAM/mock通过；与原sealed suffix的53项联合为111 passed。测试仅NumPy及device-labelled hook/API stand-ins，不执行任何CPU/GPU神经。覆盖I/R×两个slot×两prefix、五forward/freshJ/AD账、真实非零history、未知privileged字段拒绝、绑定篡改、每阶段partial、磁盘/global cap、durable失败/first-error、API恢复及CPU拒绝；专门验证old模型评分更优时仍派发new ridge。mock不证明真实CUDA、物理时序、观测freshness或反馈收益。尚需MAIN审读+driver联调后真实运行；未生成prepare/result，不标P1通过。
