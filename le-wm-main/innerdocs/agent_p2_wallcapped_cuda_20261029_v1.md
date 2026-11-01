# P2 wall-capped HOST CUDA runtime v1

本三件套落实冻结 v2 协议的真实 prepare/run/consume，不再只是协议。worker仅执行RAM/NumPy测试，没有生产prepare、神经、CUDA或物理运行，没有读取新物理outcome。原v1/v2协议、旧runner、模型、Q、normalizer与结果不改。

## 实际 authority 与固定合同

MAIN 的 v2 prepare：`results/agent_p2_compute_matched_protocol_20261029_v2_prepare_MAIN01.json`，SHA `fbc151b975409eff95bffa084707531f20d1659cbcf08ea23e5da1b490db4bd2`。本版prepare要求外部实际SHA，调用真实v2 build完整重建，包括其原三程序report `56e8b4f2df522f7a0c4cb9bc37ad74bcbc5d4f89e2c5cd601d42cd0eb071d8ef` 和strict receipt `c491555fda5bb543af69f70987541edb0abb99fa32c559f3aa6190901393017f` 的完整saved消费，不编造future SHA。原完整slots/task/root context/goal192/Q/模型身份都在source closure内。

固定顺序 `(root0,joint),(root0,composition),(root1,composition),(root1,joint)`，全off，2 known roots，非fresh或独立确认。共同每root×策略450s同步wall cap；累计caps为1902 predict及三个模型模块各1902、85 J、16320 mathematical VJP requested rows、489600 requested J elements、255 AD primal steps、2 root SVD/1 ridge，显存4GiB allocated/6GiB reserved。实际消耗不要求相等，墙时为响应变量；equalFLOPs/equal-total-compute始终false，未知FLOPs/engine backward为null。

两臂都付费执行zero nominal3、root J primal3和bank、全部42固定seed各3 forward。随后joint round-major两轮full30（每unit21 forward），composition seed-major H2（prefix1+optimize14）与H1（7）；不按outcome更换根/候选/参数，不读真实状态，不训练/拟合。full15合法域、float64 NumPy tanh→float32 chart一次、原public action normalization、full context3/action history、原mean192 Q不变。自然完成每根joint1902、composition1062 predict，均85 J。严格改善才更新完整候选incumbent；平局保留较早者。

root bank使用显式两个NumPy SVD与冻结ridge相同的乘子求解，使CPU调用也能真实intent/return，而非宣称计数。RAM测试逐字段证明和原 `old.bank` 数值一致。`old.optimize`、冻结v1 `local_joint` 作为原算法局部接缝复用。原model transition、模块/预测器、autograd.functional.jacobian(vectorize=True)不替换为scalar AD。

## 生产与持久化

真实loader复用原single-read immutable config/weights buffer、原legacy remap、missing/unexpected检查、installed source closure、CUDA frozen/eval张量哈希；只加载一次模型，原生encoder/projector被forward hook禁止。TF32 off、autocast off、MATH SDP，模型参数不求梯度，J只对原动作坐标求导。调用前后source和模型tensor hash核验，global不monkeypatch。模型实例的临时predict观察wrapper/hooks在原context manager finally恢复。

本版继承原HostBackend的观察hook/Jacobian底层，但局部覆写forward loop和packet validator：旧1068上限不能用来限制新1902；J数仍85。每个packet保存原raw位、完整incoming context、AE/predictor/pred_proj所有input/output float32 blob和shifted context，完整J保存原NPY字节/SHA/shape。每臂独立index，文件 `root{arm_index}_J{ordinal}.npy`，不冒充旧root schema。测试实际调用新 `_rollout` 的index1901，模块ordinal5703/5704/5705；下一1902在dispatch前拒绝。

v2 Ledger通过同步monotonic clock接入；每次command先durably emit/fsync `ledger_intent`，执行后emit return或failure，包含实际clock、调用args及返回digest。native hook intent/return与操作ID严格配对；CPU callback也有成对原生事件。只计实际进入的模块/forward，失败intent不退款。J requested工作和实际primal/module计数分列，不能把前者说成engine实际backward。encoder/physics/training新增均0。

Store真实写入gzip chained journal（每次Z_SYNC_FLUSH+fsync）与完整row/NPY。输出cap512MiB，final reserve32MiB，floor512MiB；journal解压cap1GiB，metadata prepare4MiB、receipt128MiB。outer2400s边界包括加载/初始saved verify/四臂/写盘，是安全界限，不是共同450s搜索计时的替代品。加载和最终strict不计入单臂搜索窗口；CPU优化/搜索IO/捕获/导出在窗口内。MAIN须串行占GPU，不和其它job并行。

## deadline、partial 与导出

只在完整unit闭合后提交完整full15 witness；deadline之后的改善保留late记录但不进入capped incumbent。资源admission预留最终6次forward/模块双replay；资源拒绝且仍有墙时可以导出。wall deadline不准追加隐形尾部；缺replay则export missing，固定四分母仍保留。自然完成不padding、不重试、不换根。

这是cooperative cap，不是CUDA kernel硬抢占：in-flight可能overrun，如实记录并停止新dispatch。异常逐层unwind，未返回的module/forward/J以非成功闭合，requested/attempted不退款；模型/输入raw字节、所有实际前缀和错误信息保留。最后seal冻结搜索终点，后续其它臂和strict时间不污染elapsed。fatal启动/加载错误保留failed report，但consumer不替此类不完整全四记录签完整receipt。

unfinished native消费按 **实际事件前缀** 而非模块数合成API返回：AE返回之后未必进入predict；predictor必须先有真实API intent；第三模块返回之后可能在hook的check抛超时，故不推出model.predict已返回。缺失J矩阵不能制造或补零。partial可以获得“完整保留/可核验失败前缀”的工程验证，但不能提交不完整候选或伪作full export。

## 严格saved consumer

1. 外部report SHA、实际own prepare、v2与原三程序strict全量重建；loader consumed buffers/model hashes/source closure核验；精确文件集合/SHA、gzip EOF、ordinal/previous/event hash/canonical bytes全检，未知或terminal extra事件拒绝。
2. 按每条保存的clock/args独立重建v2 Ledger，验证deadline/resource/memory/attempted/completed/停止/错误/最终seal，不信report flags。所有charged intent必须有真实native开始，所有成功return必须有对应native结束；失败pending必须非成功闭合，不退款。
3. 由固定root/arm/unit顺序独立重走同一 `drive` 算法。SavedBackend逐call核完整expected context/raw/tag/normalizer/full IO/精确forward ordinal/J坐标与192×d NPY/所有算法选择事件；CPU bank从保存J重新计算。完整unit及所有控制请求/commit都必须相同，中断单元消费恰好已执行前缀。
4. 完整导出必须whole15和独立3 segments全部context精确相同，并等于capped anytime winner。receipt按固定顺序返回own `.plan` 或None、4个account summary、2个配对response panel以及完整bindings。不将未完成计划丢出分母；None不进入物理runner。
5. 验证后再hash输出和source，拒绝漂移。receipt不重复巨大ledger事件字符串；保留ledger事件digest，原row/journal由bindings固定。工程receipt不是P2科学准入或等FLOPs证据。

## MAIN完整CLI

工作目录 `/home/gbk/code/LeWorldModel/le-wm-main`；所有新输出必须不存在。worker没有执行以下生产命令。

```bash
.venv/bin/python -m experiments.spectral_reachability.agent_p2_wallcapped_cuda_20261029_v1 prepare \
  --protocol-prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_compute_matched_protocol_20261029_v2_prepare_MAIN01.json \
  --protocol-sha fbc151b975409eff95bffa084707531f20d1659cbcf08ea23e5da1b490db4bd2 \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_wallcapped_cuda_20261029_v1_prepare_MAIN01.json

.venv/bin/python -m experiments.spectral_reachability.agent_p2_wallcapped_cuda_20261029_v1 verify \
  --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_wallcapped_cuda_20261029_v1_prepare_MAIN01.json \
  --prepare-sha ACTUAL_OWN_PREPARE_SHA

.venv/bin/python -m experiments.spectral_reachability.agent_p2_wallcapped_cuda_20261029_v1 run \
  --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_wallcapped_cuda_20261029_v1_prepare_MAIN01.json \
  --prepare-sha ACTUAL_OWN_PREPARE_SHA \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_wallcapped_cuda_20261029_v1_cuda_MAIN01 \
  --authorization MAIN_P2_WALLCAPPED_KNOWN2_HOST_CUDA_ONLY

.venv/bin/python -m experiments.spectral_reachability.agent_p2_wallcapped_cuda_20261029_v1 consume \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_wallcapped_cuda_20261029_v1_cuda_MAIN01 \
  --report-sha ACTUAL_OWN_REPORT_SHA \
  --receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_wallcapped_cuda_20261029_v1_consume_MAIN01.json
```

ABI `consume(output, expected_report_sha)` 返回本版 `.consume`。plans固定4项（可None），每个非None含原task/root context/goal/warmraw10、15命令/raw bits、3预测context、模型Q与selection_digest；完整原任务路径/配置从其真实prepared slots和bindings取，不把本plan伪装旧三程序物理schema。accounts给实际资源/elapsed/stop/error/export；comparisons明确wall-capped、不是equalFLOPs。

## 最终验证记录与边界

测试命令 `.venv/bin/python -m pytest -q tests/test_agent_p2_wallcapped_cuda_20261029_v1.py -x`，**28 passed in 12.54s**。初始TDD RED为缺新模块；后续小测、真实writer、strict、partial前缀逐步扩展到当前版本。没有把测试当真实NN证据。

覆盖真实Store→磁盘→gzip/hashchain reader、四行完整saved consumer（模型加载authority为RAM fixture，不加载权重）、原生full3 IO/1901边界、完整H2/H1与joint单元、完整composition1062/85自然停止与双replay、raw/goal-context/clock/J/ledger/native chronology篡改、unknown terminal重封hash仍拒绝。mid-J primal已返回但矩阵缺失，以及mid-forward module inputs已保存但没有module返回的fixture，均真实落盘重读并保留非成功unwind；另有AE/第二模块/第三模块返回后及predictor inputs中断，验证不合成不存在的API返回、缺API intent时拒绝。

RAM模型为显式NumPy加法/零张量或fake tensor，仅验证schema/dispatch/算术/消费；不import torch做worker NN、不运行原生物理、不声称CUDA性能/数学J正确性已实测。实际installed model forward/J/显存/时长及wallcap下收益，仍待MAIN审读冻结版本后HOST CUDA实测。P1/P2 admission、fresh、physical release与equalFLOPs均false。
