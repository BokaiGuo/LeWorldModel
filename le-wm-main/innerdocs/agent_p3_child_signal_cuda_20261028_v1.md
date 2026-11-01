# P3 paid observed-child native H1/H3 adapter

独占source/tests/doc；不改冻结child contract/原signal/MAIN文件。worker仅RAM和已保存metadata测试，不执行模型、CUDA、CPU神经、物理或网络，不创建后代。不训练/改模型/Q/normalizer、不调阈值或按结果挑根。

## 范围与计费

prepare严格verify实际child合同，pin其三SHA及旧native signal源码SHA。实际合同路径`/home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_child_scheduler_contract_20261028_v1_prepare_MAIN01.json`，SHA `8669fb219c6a17e05d5c1b8dc557b401651a2bf3fc759529d64231e5d317f7fd`。全部24 child/48方法保留，23 observed、1not_reached；不能按chi/预算/outcome筛槽。signal acquisition数量按事前independent/shared_equal确定；不制造未来signal report SHA。

NativeChildSignal直接校验root task binding和精确observed child context；不把child冒充root、不改旧globals、不借旧root J。模型实际原生`action_encoder→predict→predictor→pred_proj`，latent[1,3,192]、原history[1,2,10]。每个H分别独立nominal和AD，H1+H3共8predict API/8samples、2J、384 scalar VJP/grad/raw hooks，2array SVD；零raw仅是固定nominal，真实非零history绝不清零。完整origin/raw/normalization/每层输入输出/parent-child context/192 onehot/梯度/J原float32 buffer逐条fsync绑定。

复用原native frozen检查、instance-local forward observer、全state_dict/所有buffers hash与原Store；新signal请求校验不调用旧root-only validator。权重**文件**SHA为`48938400...`，native state_dict hash为`4b12c01f...`，明确不同owner，不互换。P4 modelbinding精确绑定weights/config/source_closure；prepare合并原合同完整source/安装包闭包及本三文件；run核实际模块类源，最终再次核闭包。无新encoder/projector调用，P4编码获取仍属历史账，不能称免费。

independent为每方法各8API/2J/384VJP：当前46 acquisition，368predict/92J/17664 scalar VJP，46schedule。shared_equal为23 acquisition，184predict/46J/8832VJP，46schedule；每方法只分摊同一实际acquisition的1/2，不伪造独立信号。所有资源保持独立单位，FLOPs/CUDA ns/backward engine未知null，nested wall不求和。

run实际调用冻结contract.plan_child，保留原4/8候选、H1/H3与depth规则，包含真实paid schedule；高候选/H3若超过24search cap整节点拒收，不缩配额。此adapter只获取信号和生成请求，**不调用generate/replay/物理**。所有候选请求、拒收、未到达槽完整保留；signal工程失败停止本panel，余槽blocked、paid attempts不退款、不重试、不从部分成功释放。

HOST CUDA-only，TF32off、autocastoff、MATH SDP，原参数eval/frozen，前后独立hash包括buffers；加载CPU反序列化不做CPU神经forward，无CPU fallback、离线禁下载。run必须精确`MAIN_APPROVED_HOST_CHILD_CUDA`，授权核验先于prepare消费/torch导入。900s whole-job wall含initial verify至final source检查及journal close，final report写入另列；512MiB整个output、256MiB final reserve、200MiB disk floor、allocated3/reserved4GiB固定工程cap，不据mock放宽。cap/进程异常可能只有durable journal无完整report，strict consume拒绝部分结果。

## 独立saved consumer

无torch/模型/物理导入。外部实际report SHA、真实prepare/source闭包、完整三artifact/哈希链、原48分母、actual attempt/return嵌套、IO支付与scope关联均重核；逐H验证child原点及非零history、raw坐标、AE/predict/predictor/pred_proj完整wiring、终点滚动及384个onehot raw梯度/J行。独立saved NumPy SVD重算同短H参考残差chi/有符号gain，再调用原choose核全部queue/配额/拒收/公平分摊。saved验证自身SVD数量单列，不混入CUDA作业成本。

consume输出actual measured_search_budget_refusals、fixed/adaptive各24分母下observed/not_reached/refused分栏及是否所有observed方法拒收；实际signal没运行之前不能给该数值或预测拒收分布。RAM fixture即使借用HOST字段测试ABI也不是GPU证据；外部manifest/源码审读、主机执行认证由MAIN负责，数组内部一致性不独立证明导数数学正确性/无hidden神经调用。

仍缺child progress/reconfiguration compiler、whole/segmented与selected-prefix独立回放、实际新物理桥/因果在线控制和新后果、fresh/P1/适用P2/P3前置。known-root工程校准，不称C3/P3落地或compute优势；所有scientific/freshness/physical_release门False。

## MAIN绝对路径命令

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
OPENBLAS_NUM_THREADS=1 /home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -B -m pytest -q -p no:cacheprovider /home/gbk/code/LeWorldModel/le-wm-main/tests/agent_p3_child_signal_cuda_20261028_v1_test.py
/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -B -m experiments.spectral_reachability.agent_p3_child_signal_cuda_20261028_v1 prepare --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_child_scheduler_contract_20261028_v1_prepare_MAIN01.json --sha256 8669fb219c6a17e05d5c1b8dc557b401651a2bf3fc759529d64231e5d317f7fd --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_child_signal_cuda_20261028_v1_prepare_MAIN01.json
# CHILD_SIGNAL_PREPARE_SHA取上条真实输出，不能填未来SHA。
/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -B -m experiments.spectral_reachability.agent_p3_child_signal_cuda_20261028_v1 verify --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_child_signal_cuda_20261028_v1_prepare_MAIN01.json --sha256 "$CHILD_SIGNAL_PREPARE_SHA"
# 以下仅MAIN审读后在HOST运行；worker没有执行。
/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -B -m experiments.spectral_reachability.agent_p3_child_signal_cuda_20261028_v1 run --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_child_signal_cuda_20261028_v1_prepare_MAIN01.json --sha256 "$CHILD_SIGNAL_PREPARE_SHA" --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_child_signal_cuda_20261028_v1_cuda_MAIN01 --authorization MAIN_APPROVED_HOST_CHILD_CUDA
# CHILD_SIGNAL_REPORT_SHA取真实terminal report，不能伪造。--receipt排他持久化，只打印compact写入回执。
/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -B -m experiments.spectral_reachability.agent_p3_child_signal_cuda_20261028_v1 consume --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_child_signal_cuda_20261028_v1_cuda_MAIN01 --sha256 "$CHILD_SIGNAL_REPORT_SHA" --receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_child_signal_cuda_20261028_v1_consume_MAIN01.json
```

本任务只提供代码/RAM回归；实际prepare/模型run/consumer均由MAIN执行。测试数与冻结三SHA在交付消息报告；冻结后后续改动必须新版本。
