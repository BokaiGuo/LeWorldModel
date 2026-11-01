# P3 续接：封存 rank-reserve 的可执行队列前端

仅新增 agent_p3_resume_20261027_ 前缀文件，保留全部旧源码、结果及 dirty 改动；不派生后代 agent，不改共享总账，不训练/拟合/调 Q/normalizer，不筛根、不改阈值、不启动 GPU 或物理。MAIN 负责审读与真实主机调度。

个人有界完成条件：接续现有规则而非重复设计；提供 prepare/verify/queue/consume，绑定完整同 root/goal/task/bank 的 raw10 请求与预算；RAM 变异测试通过，并实际只读验证 model-only saved 资产、保存真实碰撞/零收益和完整成本，交付 SHA。此完成不是研究 goal、P1 或 P3 完成。

## 已存在的设计与实现

已读 `docs/parallel_chats_20261027/{README,04_p3_scheduling}.md`、指定五份 innerdocs 与现有 chat_p3 scheduler/CUDA/saved-validation。采用 experiment-code 技能的固定输入/日志/失败测试；不采用训练、强制非零效果或结果择优模板。新前端复用现有 rank-reserve allocation/queue 与 CUDA saved consumer，不覆写任何 globals。
旧比例 half-up 的当前两根均 10/10；新 rank-reserve 已另行预声明、封存并实际运行，不是仅有旧文档所述的“缺新尾部评分”。不再重新评分或重复旧 worker 实验。

固定工程算法：总20 blocks、每根至少5、每 block3 个完整 H2 candidate。U/N10/10且 allocation 函数不接 weights；D_rank 额外10给严格较大 w，S_rank 反向；精确平权均10/10。不新增温度、gain、阈值或随机 tie seed。保底轮询后继续原 slot0/1 顺序，稳定(cost,index)选择，whole/segmented 独立回放。rank-only 会放大微小信号差，不能冒称原比例规则修复或理论优势。

绑定的真实 model-only 输入：

| 对象 | 实际 SHA256 |
| --- | --- |
| results/chat_p3_20261027_queue_seal.json | 2f39483bceb87e638f74b2da170db8cf7ddbf6a1050df4137bbbcf9dde37ed78 |
| results/chat_p3_20261027_cuda_prepare01.json | 72a02d66c99c9037cd55828bf67342514ec110a3310ced194477ef6e34c432f1 |
| results/chat_p3_20261027_cuda_run01/report.json | 6df856e36b145162e6a8d5ce5f141880a49afe88f43d3cc88f8b98ad790b6d08 |

## 新接口

source：`experiments/spectral_reachability/agent_p3_resume_20261027_frontend.py`。

- `material/prepare`：调用既有 CUDA 的 saved verify 重建封存输入；只把 frozen weights 交 allocation，验证 seal 的 blocks/jobs/全部240候选摘要。生成每 job 的身份、literal raw float32[10,2]、raw bits hash、两包完整 H2、API预算与回放依赖；新 prepare 绑定自身 source/tests/innerdoc 和完整 predecessor 闭包。
- `verify`：外部 prepare SHA + 完整重建精确相等，拒绝漂移。
- `queue --arm`：返回实际可给现有执行器的60项 literal raw10 请求和四个依赖 winner 的 replay job，不调用 backend。分配已经冻结，不允许消费者按收益重新分配。
- `consume`：先验证新 prepare，再调用既有 CUDA 完整 saved consumer；该 consumer 重建信号/J、全部候选/argmin、520 个完整 context、实际模块数组/资源和两个独立 replay。本前端再将实际 candidate_request/queue_frozen 与每个请求 seal 配对，保留八计划、配额/ winner / literal command 三层碰撞及 model-only 零负差。

没有新 GPU runner：MAIN 如要再执行必须沿现有受审 CUDA 执行器另行授权；本前端没有 run/物理开关，已完成产物的 resume 状态明确 `completed_saved_execution_do_not_rerun_not_admitted`。不是需要补跑新 tail 的状态。

## 预算与科学边界

每策略120 search predict API +8 replay =128；共享8实际只计一次，panel520。sample 轴从既有实际 module input shape 逐 group/模块重数，不能以 API 数代替。predictor 的 batch rows 与 pred_proj flatten rows 分开。2J、384 VJP/grad API/raw callbacks、2 model SVD 与所有 historical encoder/anchor receipt 单列。nested inclusive API/module wall 不相加冒充总wall或FLOPs；backward engine calls未知保持 null。saved 检验会重算 numpy SVD，其次数未在本层计量保持 null，而不是0；新模型 SVD、神经和物理调用为0。
独立部署视图 U/D/S136 predict +各自谱成本，N128；不是相同总资源。历史8 encoder/8 projector、288 anchor predict与完整 AD/J/SVD/solve receipts 保留，不退款。默认固定预算比较只指本次各 arm 的128 predict，不宣称端到端/FLOPs/壁钟优势。

P1/freshness/P3/independent confirmation/compute/wall advantage 全 false。不读取 downstream physical/RGB outcome 决定资源，也不把 model-Q 零收益当作真实物理收益。新 fresh 队列仍缺 MAIN 认证的事前有序独立 root/goal/source 排重、source/encoder/anchor 完整闭包与全部 acquisition/执行预算，以及原 P1 机制准入；不得把当前 known roots 追认为 fresh。已有 prospective queue 的 population/signal-separated 设计不被此 rank 工程前端改写或自动授权。

## CLI（cwd=/home/gbk/code/LeWorldModel/le-wm-main）

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/agent_p3_resume_20261027_test_frontend.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p3_resume_20261027_frontend prepare --output results/agent_p3_resume_20261027_prepare01.json
# PREPARE_SHA 取上一条真实输出；不虚构未来 SHA。
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p3_resume_20261027_frontend verify --prepared results/agent_p3_resume_20261027_prepare01.json --prepare-sha "$PREPARE_SHA"
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p3_resume_20261027_frontend queue --prepared results/agent_p3_resume_20261027_prepare01.json --prepare-sha "$PREPARE_SHA" --arm D_rank
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p3_resume_20261027_frontend consume --prepared results/agent_p3_resume_20261027_prepare01.json --prepare-sha "$PREPARE_SHA" --cuda-output results/chat_p3_20261027_cuda_run01 --cuda-report-sha 6df856e36b145162e6a8d5ce5f141880a49afe88f43d3cc88f8b98ad790b6d08 --output results/agent_p3_resume_20261027_saved01.json
```

输出只允许新 results/agent_p3_resume_20261027_ 文件，exclusive/fsync，2MiB cap/200MiB floor；不覆盖旧文件。prepare/verify/queue/consume 均仅 saved metadata/数组工作，不启动模型、CUDA、模拟器或训练。RAM mock 不是科学证据。实际验证记录在最终交付时另列，不以未来状态充数。

## 本工作位实际验证（不是新神经实验）

RAM 回归25 passed in 0.37s；CLI --help exit0。包括完整240 candidate请求、5-block保底、总预算、精确平权/双零碰撞、N信号隔离、原始动作/hash/order/source身份、实际sample轴、八分母、bool计数拒绝、API时间与科学flag变异拒绝，零/负模型差均保留。

实际只读 `material` 重建成功：172项闭包，prepare序列化334,723 bytes，未加载 torch/gymnasium/pymunk/stable_worldmodel。随后实际调用既有完整 CUDA saved consumer并运行新前端 summarize，重建请求顺序与全部原始候选 seal，saved 检验耗时8.716520439949818s（含全部保存数组验证，不是新CUDA墙钟）。

| 模型信号/分配 | slot0 | slot1 |
| --- | ---: | ---: |
| w | 162.32405169996156 | 156.2867146609139 |
| U / N blocks | 10 | 10 |
| D_rank blocks | 15 | 5 |
| S_rank blocks | 5 | 15 |
| 四策略共同 winner index | 1 | 2 |

配额确已分开，但同根各策略 literal commands 全部碰撞；每根六个成对碰撞全部保留。实际 D−U、D−S、D−N model-Q 差均 `[0,0]`。不以增加尾候选未赢作为删根理由，不宣称真实任务收益为0，不放大信号或再调整预算。
从实际 module shape 重数：各策略128 predictor API/128 predictor rows/384 context tokens，shared8 API/8 rows/24 tokens；总520 API/520 predictor rows/1560 tokens。pred_proj 各策略384 flatten rows、shared24、总1560，不能把它当1560个predictor samples。
原已完成 CUDA 全panel墙钟28.22814436489716s，2J/384 VJP/384 grad API/384 raw callbacks/2SVD，全部attempt=return；新验证不重复支付这些神经调用。历史anchor receipt仍保留30J、32SVD等各臂原账，不将本轮shared2J/2SVD冒充全部历史开销。
本工作位不读取已存在的 chat_p3 physical/RGB outcome。只封存一个新的可重建 saved metadata 交接，不重跑其神经或物理；P1/freshness仍关闭。
