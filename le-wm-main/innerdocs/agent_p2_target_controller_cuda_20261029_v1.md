# 固定四 target 的共同 controller HOST 接通协议

本协议先于本runner实现固定。仅新增同stem source/tests/doc，不改已封存tracker/delivery/core。本slice不是三proposal生产、同时间比较、公平基线或科学P2；P1/scientific gate仍false，known2不是fresh。

输入唯一实际delivery prepare：`results/agent_p2_target_only_delivery_20261029_v1_prepare_MAIN01.json`，SHA `82b36fa3030f8ebb2da37bdc91ba3e8d582f129d6bc9e5c9f8e774b363ef3f62`。重建完整saved-only delivery核验，不调用旧full strict。顺序固定：root0 prior FIRST、program首postcondition、arbitrary missing；root1同三槽，共六槽。四assembled槽全部运行，不挑target，不补arbitrary；缺失与失败保留原分母。model函数仅接严格白名单 `controller_payload={z,a,target}`，未来witness actions/metadata不进入controller；过去action2是合法causal history。

每槽复用原smoke spec：H3/N8/I2/topk2/var1/seed0/segments=[1,2]，full_horizon_zero_no_witness，unclipped_normalized_gaussian，identity192_terminal_meanMSE_float64；predict API cap12、samples cap54、数组cap32MiB、每槽wall300s。四槽总48API/216samples，搜索与actual mean的whole/segmented paid replay全保留。encoder/high/AD/physics均0，FLOPs=null，原proposal成本继承未知且另列，不伪称同生成预算。

固定原model425及原source闭包，单次受信CUDA构造，完整construction receipt、模型前后/每槽前后shape/dtype/值SHA冻结守卫。所有四槽输入须在模型构造前排他持久化完成；所有full traces/callbacks/candidates/mean/std/context/checkpoints/paid scores完整封存，不dedup。

资源固定全job1024MiB（含128MiB失败footer reserve），每root256MiB（同root两槽合计，预留32MiB footer），floor200MiB，全job wall1800s，每槽原300s；CUDA allocated3GiB/reserved4GiB，HOST CUDA only，MAIN授权。失败attempt不退款，未返回trace的扣账unknown，partial不伪装完成。不得运行NNCPU或自动重试/调参。whole wall包括verify/load/hash/run/persist/local_check，到最终report写入前采样；最终排他fsync另列为非硬实时收尾限制，不能抢占正在运行的CUDA内核。

MAIN source-first审阅后执行prepare→verify→HOST run→完整saved consume；worker仅RAM/fixture/小保存ABI。后续实现/CLI与测试记录在此同一未冻结文档补齐后，三件套一起冻结。

## 实现接口与持久化

source为 `experiments/spectral_reachability/agent_p2_target_controller_cuda_20261029_v1.py`；tests为 `tests/test_agent_p2_target_controller_cuda_20261029_v1.py`。采用experiment-code的固定协议、日志和RAM回归；不适用训练、图表、调参或以正结果为完成门槛。

`build/prepare`读取上述固定delivery并完整重建 `delivery.build()`（仅saved数组/metadata join）。引入原tracker prepare的完整expected_state/export/construction/sources；merge时拒绝冲突，封存旧425 NPY与所有依赖/所选输入source SHA。新delivery三件套也固定原SHA。只调用原read/hash/closure/NPY/helpers，不调用原tracker `verify/consume/run`，不重建旧full strict。prepare里的425源文件哈希由MAIN执行；worker不读weights、不做实际prepare。

新 `Store` 仅继承旧Store的排他写/NPY序列化，覆盖本stem输出路径与资源检查；没有修改旧module globals。root0的两target共用256MiB root budget，root1同理。文件名为 `slot0/1/3/4_input_{z,a,target}.npy`、`slotX_array_N.npy`、`slotX_trace.json`，不做跨槽复用。最终report不计入自身SHA inventory，以外部实际report SHA认证；完整作业文件总数/字节、root合计、每次写尝试/返回、排他fsync都严格消费。

`persist_inputs`在导入torch/构造模型前，先写完四槽12份输入NPY与dispatch_manifest。`execute_slots`只向invoke交新拷贝的三键数组；生产invoke只把三数组搬CUDA后调用冻结 `core.track_cuda(graph,z,a,target,spec=原smoke,expected_state=原425)`。不把root、arm、origin、witness或原生成器的动作交给model。不增加encoder/high调用，也不把原输入编码成本记成新工作。

`track_cuda`实际返回的core trace/全部数组不改backend、不修补、不去重。whole与segments=[1,2]的actual mean两次paid replay完整保存。每槽实际计数、所有candidate/callback/mean/std/module/checkpoint输入输出，及独立重建的完整 `core.validate_saved`结果均join。这里的CUDA mean/std/RNG检查仍是原core运行时witness，不是worker重新运行GPU或独立梯度证明。

模型只构造一次，construction字典与旧已运行parity/tracker中的原字典完全相等；完整425 shape/dtype/value SHA/eval/frozen guards在全局前后及每槽前后复查。CUDA op前后调用已冻结GPUResourceGuard+新store，实际peak allocated/reserved必须不超过3/4GiB；无CPU fallback。deterministic/TF32off/autocastoff与CUBLAS workspace固定，退出恢复原precision与alarm，不patch全局model/solver函数。

非返回调用的debit标unknown不填0；普通partial返回的实际debit先记下再写文件，失败不退款。后续槽保not_started、arbitrary两槽始终missing。任何partial均不能消费成功；不重试、不padding。core trace内存返回后才落盘，无法承诺强杀/进程崩溃前未返回事件已持久化；失败report保留现有完整文件与写失败记录。footer可用reserve但仍受floor/总容量限制；极端disk I/O失败无法保证footer成功，不隐藏此局限。

`consume`核外部reportSHA、重建本prepare（仍不运行上游full strict），逐槽验证同delivery payload字节、完整source/model beforeafter、core backend/spec/数组/paid mean replay/local_check、48API/216samples、六槽原序、两missing零dispatch、所有NPY header/EOF/finite/file/dataSHA、root/slot归属、完整census/写账/资源。输出status仅 `all4_target_controller_saved_verified_2_arbitrary_missing`，不称公平比较、时间匹配或科学收益。

## MAIN完整CLI

在 `/home/gbk/code/LeWorldModel/le-wm-main` 执行。以下实际prepare/verify/run/consume交MAIN；worker只跑RAM测试。输出名已存在时换新MAIN编号，禁止覆盖失败。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q tests/test_agent_p2_target_controller_cuda_20261029_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_target_controller_cuda_20261029_v1 protocol
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_target_controller_cuda_20261029_v1 prepare \
  --delivery /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_target_only_delivery_20261029_v1_prepare_MAIN01.json \
  --delivery-sha256 82b36fa3030f8ebb2da37bdc91ba3e8d582f129d6bc9e5c9f8e774b363ef3f62 \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_target_controller_cuda_20261029_v1_prepare_MAIN01.json
# CONTROLLER_MH 填上一步实际返回SHA，未预造未来hash。
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_target_controller_cuda_20261029_v1 verify \
  --manifest /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_target_controller_cuda_20261029_v1_prepare_MAIN01.json \
  --manifest-sha256 "$CONTROLLER_MH"
CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p2_target_controller_cuda_20261029_v1 run \
  --manifest /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_target_controller_cuda_20261029_v1_prepare_MAIN01.json \
  --manifest-sha256 "$CONTROLLER_MH" \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_target_controller_cuda_20261029_v1_cuda_MAIN01 --main-authorized
# CONTROLLER_RH 填实际run返回report SHA。
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_target_controller_cuda_20261029_v1 consume \
  --report /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_target_controller_cuda_20261029_v1_cuda_MAIN01/report.json \
  --report-sha256 "$CONTROLLER_RH" \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_target_controller_cuda_20261029_v1_consume_MAIN01.json
```

RAM覆盖真实delivery metadata固定SHA/三件套pins、四槽白名单和六分母、12输入先落盘/失败unknown账、RAM返回不冒认HOST且partial保留、NPY dtype/shape/EOF/hash/symlink、root/job caps与footer、文件census/账/来源冲突、fixture prepare→verify和cold CLI无torch。真实HOST与全量consume尚待MAIN执行；未运行四target不能报真实完成/性能。原scaler与prior物理时间未知不阻塞此normalized接通，但仍阻塞相应raw或time-matched结论。
