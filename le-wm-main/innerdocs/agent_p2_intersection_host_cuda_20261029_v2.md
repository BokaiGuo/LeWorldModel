# P2 六槽原27轴交集：HOST 接线与 saved 重建

## v2 故障修复

MAIN01 v1 的 index603/1204 是失败报告而非未落盘的过大事件；事后按原事件形状和日志中三个已编码 context 重建 `AD_return` 为 2009 bytes，超过原 1400 上限。其实际执行停于 J 后、该事件写入前；不是模型/科学负结果。v2 只把最多三个 context 的 `AD_return` 上限改为4096，保持记录和完整 reader 验证，其他事件上限不变。`AD_intent` 的 float 列表在事件写入前转为有界数组描述符；真实形状 RAM/tempdisk 回归检查其1400上限。其他已观察事件最大值不得视为未观察未来事件上界；所有事件超限仍拒绝并保留失败。

原整数证明每个 J 的四条额外事件各按1400记账，v2 将其中一条改为4096：2根×2谱臂×85次×(4096−1400)=916640 bytes，纳入原40MiB extra_metadata；重新计算总 journal 上界仍小于原784MiB 分区，不扩大1536MiB总体或科学资源。若实际事件、空间或形状超界仍拒绝，不重试或补结果。

原 v1 内部 operation_failure 可能被 driver 捕获而顶层 recorded/errors=[]；v2 完成既有六槽记录后把内部失败计入顶层 errors/status failed_partial，consumer 明确拒绝完整回执，且独立检查每槽；正常 wall partial 仍按原合同。固定分母、失败记录和原六槽顺序保留。旧 MAIN01 不换绑、不覆盖。

仅本 stem source/tests/doc；不改冻结 intersection、原common runner或原结果。MAIN负责审读、prepare、CUDA及实际consume。本轮测试仅RAM、假数组及临时盘，不训练、不跑NN/physics/fullstrict，不按已见outcome改预算、选根或删GRASP。

## 比较对象与准入

原两known off根，固定顺序 root0 joint/composition/grasp，root1 grasp/composition/joint；相同原task、model、history3、goal、identity192 Q、原P2工程normalizer和raw15 support/criterion。自然上限1902/1062/1902 predict，wall每槽450s，完整双replay6次均实扣；不padding、不共享结果或免费梯度。J数学192行、标量backward、AD primal、CPU SVD/ridge、Adam/noise分别计数，不称等FLOPs、等实际算力、官方750步GRASP或新fresh确认。P1/P2科学机制、物理部署许可均false。

prepare复用冻结common `build`/`join_tasks`全task/model闭包；本件再绑定intersection源码、tests/doc、安装版 `stable_worldmodel/wm/lewm/lewm.py` 和本三件套SHA。实际checkpoint config的 `_target_` 为 `stable_worldmodel.wm.lewm.LeWM`，其具体方法在安装版 `lewm.py` 的`LeWM.predict`，不是仓库 `jepa.py` 的类。实际receipt/model/权重不得用fixture伪造。明确授权字符串只打开MAIN HOST CUDA，无CPU neural fallback。`contract.budget.production_runner_implemented=false`是嵌入的**原内核自身描述**，不是本新接线的运行结果。

## 接口与真实阶段

`StagedLedger`继承冻结box的unit准入、逐轴cap、墙钟和commit；对实际模块拆为 attempt → input capture → dispatch → 原模块回调 → returned → output capture → captured。shape由真实参数的 `shape`读取；输入捕获若失败，dispatch保持0；kernel返回后capture失败，returned保留、captured保持0。上限、durable写、wall、模块形状失败均不退款/重试/补结果。捕获CPU bytes/秒含输入和输出两个阶段，不把它们称纯GPU kernel耗时；已进入kernel不能抢占。

冻结box的通用`predict`把packet入口记为`predict_API`；本件显式拆开该轴：packet建立回调栈但不记API，AE成功后、调用原`LeWM.predict`之前才扣`predict_API`及相应AD primal。AE失败时真实API为0，不把packet尝试谎报为API；完整日程的27轴上限与自然总数不变。这是新子类解释，不改旧box。`AD_primal`在实际predict API入口计，不能等价于所有内部AD engine工作。

`ModelFacade`是新的局部对象：三个方法调用原实例的AE/predictor/pred_proj；调用原安装版`LeWM.predict.__func__`执行原预测函数体，并检查module、qualname与源码路径。源码SHA `ccc2ac2880880f85ea71a5baa2152ba97e3f6dc5cb546d27570795aa6020d3d2`；实际文件位于 `.venv/lib/python3.12/site-packages/stable_worldmodel/wm/lewm/lewm.py`。不写旧节点、原model方法、全局或`_modules`，不复制/训练参数。`HostBackend`原transition/history/normalizer/J API不变；`GraspBackend`保留原AD/Adam/noise。GRASP lifted三项损失的**加法顺序、原项和dtype不变**，为实现score前计账，在三次预测完成后按原顺序求和；这改变了与预测调用的执行交错，尚未由本轮CUDA认证为逐位旧实现等值，不应称官方复现。

`Tape`把同步clock与每次typed命令和底层实际ledger event耐久化；own saved侧用原输入和保全的完整IO/J/梯度逐项重走相同控制算法，重建27轴/side费用与journal顺序。谱方法复算原bank、proposal、同context/动作、原Q和完整whole/segmented；GRASP核loss、chart导数、原已封存Adam算术、noise和全模块接线。保存的神经输出/J/梯度不因此变成独立重跑神经或独立反传。cooperative wall在unit内的可重放前缀独立核已捕获的输入、返回阶段与账本；缺失输出仅用不允许解码的内部sentinel到达已记wall边界，绝不作为模型输出、分数或plan。其他执行异常或transport损坏保留partial/report但拒绝完整receipt；wall边界及missing replay保持原分母和null。

计分scope显式区分：joint完整路径为`root_full_path`，H2/H1自身context的后缀终点评分为`serial_terminal`，GRASP joint为`lifted_loss`，sync为`serial_terminal`；完成的整条witness另在原root goal上评分。CPU capture单独字节/时间覆盖模块输入与输出，不包含raw/context/J/optimizer其他CPU拷贝，这些操作仍包含在450秒搜索wall中；不能据此称CPU总费用已对齐。

## 整数形状/字节与资源边界

`spec`提供可执行的整数census。六槽全部自然日程合计9732 batch1 packet：每packet AE `1×3×10→1×3×192`、predictor `2×(1×3×192)→1×3×192`、lowproj `3×192→3×192`；另保存完整前后history与raw。模块IO按f32、history列表传输按f64计算共229285920字节；谱J上界5898240字节，GRASP optimizer上界35917824字节。存储采用有SHA/offset/dtype/shape的raw archive，输出字节去重只针对已捕获的完全相同字节，**不复用模型调用或省费用**。不依赖压缩比。

原总上限1536MiB不增加：archive368MiB、journal784MiB、row192MiB、final reserve96MiB，合1440MiB；磁盘floor512MiB。budget event各类别有严格逐事件字节上限；真实packet native事件最多21条，额外score、GRASP optimizer、J和失败metadata各自以固定整数额计入40MiB，`byte_proof()`检查总和。每row 32MiB，forward/AD/neural/ledger各项单独限制；所有写入前检查分区、总量、磁盘、排他路径并fsync。shape/census外值、过长事件、空间不足或写失败直接停止，不能把部分信息当成功；这些上界不保证墙钟、OS或非法模型ABI下全日程完成。

搜索scope包含clock同步、预算门、模型、CPU、捕获和逐事件持久化；loader和saved strict另计。report单列initial verify、host setup、loader、task rebind、搜索后row持久化、最终identity与artifact hash；`total_seconds`从initial verify开始到artifact hash结束，明确不含最后report本身的写入/fsync。逐事件fsync可能使本新job在450s提前停止，保留实际停止，不把wall当性能结论或从旧job结果挑配置。stage限allocated4GiB/reserved6GiB，外层3600s。

## 现在可核验的命令

从 `/home/gbk/code/LeWorldModel/le-wm-main`。仅MAIN审读接受本件后使用生产命令；生产SHA由实际返回，不猜。原protocol固定入口：

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p2_intersection_host_cuda_20261029_v2.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_intersection_host_cuda_20261029_v2 verify-sources
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_intersection_host_cuda_20261029_v2 spec
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_intersection_host_cuda_20261029_v2 prepare --protocol-prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_compute_matched_protocol_20261029_v2_prepare_MAIN01.json --protocol-sha fbc151b975409eff95bffa084707531f20d1659cbcf08ea23e5da1b490db4bd2 --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_intersection_host_cuda_20261029_v2_prepare_MAIN01.json
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_intersection_host_cuda_20261029_v2 verify --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_intersection_host_cuda_20261029_v2_prepare_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_intersection_host_cuda_20261029_v2 run --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_intersection_host_cuda_20261029_v2_prepare_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_intersection_host_cuda_20261029_v2_cuda_MAIN01 --authorization MAIN_P2_INTERSECTION_HOST_V2_CUDA_ONLY
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_intersection_host_cuda_20261029_v2 consume --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_intersection_host_cuda_20261029_v2_cuda_MAIN01 --report-sha ACTUAL_REPORT_SHA --receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_intersection_host_cuda_20261029_v2_consume_MAIN01.json
```

Worker当前证据是RAM/fake-array/tempdisk，不能预先确认真实host图类型、旧lifted交错数值一致或全slot产出。独立验证须查实际source/input/model前后SHA、所有row/journal/archive完整性和分区、每槽完整可重建状态；任何真实缺项/不符拒绝receipt，不通过扩大cap、松容差或补missing修复。原始common结果不被换绑为本件证据。
