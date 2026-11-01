# P3 next：碰撞诊断与冷启动等增量资源对照

仅写 agent_p3_resume_20261027_next 独占新文件；旧前端、源码、封存结果、MAIN文档与dirty改动不动，不派生下级。采用 experiment-code 的固定输入、成本日志、失败保全和变异回归；训练、强制非零效果、择优留存模板不适用。本位不执行模型/CUDA/physics，不下载。MAIN负责审读、主机调度及科学准入。

个人有界完成条件：实际saved核旧碰撞链，固定新候选/调度/预算，交付可运行 prepare/verify/queue/run/consume、RAM回归、诊断/prepare和SHA。不把个人完成、RAM测试或工程执行当作P1/P3研究完成。

## 已实际读到的碰撞原因

先复用上次完整前端consumer核验 prepare `f6b827cc50f1daf4070a67058fbddb548e24a223c62645f6788654352602723b` 与原CUDA report `6df856e36b145162e6a8d5ce5f141880a49afe88f43d3cc88f8b98ad790b6d08`，后分析所有240次已评分候选和八分支，不读物理/RGB outcome。

配额 U/N=[10,10]、D=[15,5]、S=[5,15] 真被消费；实际模块输入输出与尾候选均核验通过。因此新rank结果不是旧比例取整退化，也不是只改metadata没消费。两根赢家恒为旧bank的1/GN、2/D2P，均在前三候选，paid=3时已达到最终最优；paid=15与全配额未改变。随机proposal命令/预测endpoint分别有12/27/42个精确不同值，非“所有候选复制同一个命令或预测”接线。重复同root/index的cost和完整context逐项一致。

最佳proposal减锚点winner的model-Q：root0 U/D/N为+0.1482603400826339、S为+0.1871053973227157；root1各臂+0.4571237692434916。旧D−U/S/N仍[0,0]，动作全部碰撞，旧负结果不删除、不重标签。

这是当前有限bank的**模型目标下锚点饱和**；不能由此证明真实物理不可达、理论错误或因果泛化。真实可达性保持null。原PushT成功位置包括agent+object，object-only方向不能代替完整任务成功；本模块不读取或重算该物理标签。

## 固定的新对照（运行前封存，不保证分离或收益）

这是诊断后的known-root工程冷启动ablation，不是原warm-start基准的修复成功，也不与旧数值直接宣称优势。四策略同root/goal/history/model/normalizer/Q/action domain，共享同一proposal顺序。只移除原bank中的zero/GN/D2P三个特权incumbent，所有付费候选均可赢，不事后过滤winner。

每root固定45个PCG64_ball候选：完全沿用原seed=20261027+101*slot、两包10维球均匀生成和BALL=.75*sqrt(5)、float32 tanh规则。前42项逐位核对原bank3..44，后3项为同流紧接draw45..47；不换seed、重采样、改半径或按分数排列。不选择根、不使用真实outcome、未先探测后3项收益。新index0..44，old_stream_index3..47，保留旧bank_sha和单独search_bank_sha，身份不混淆。

复用封存rank/minimum/dispatch：20blocks/min5/max15、每block3个候选；U/N10/10且allocation不接信号，D将剩余10给严格较大原冻结model-only w，S反向；精确平权包括双零均10/10，允许碰撞。顺序仍原slot0/1 block轮询，stable(cost,index) tie-break。新算法不接candidate score或physical outcome决定配额；不修改权重、温度、阈值或门槛。

每臂60项独立fullH2候选=120 predict API，加每root whole/segmented两种独立回放=8，共128；完整四臂512。所有候选均重算，不用历史score抵扣，不跨臂缓存/dedup/early-stop。whole与segmented必须全context精确等于所选search路径。失败不补候选、不换动作、不自动重试，八分母与已支付账保留。

## 资源口径

新增各臂 predict/action_encoder/predictor/pred_proj 各128；AD/J/VJP/raw-backward/model-SVD/solve/encoder/projector/physics均0，输入batch=1、context=3，consumer从每个实际shape重数sample。predictor rows与pred_proj flatten rows分列，不用API数假装samples。

原模型谱signal是已支付不可变saved复用：8 predict、2J、384VJP/grad/raw callbacks、2model-SVD，仅共享一次。新CUDA不重做shared谱计算。为等资源对照，独立部署比较U/N也承担同一已付谱acquisition（计算后不使用），四臂统一136 predict+同谱向量；**N在此是signal-blind资源匹配消融，不是零谱成本的部署基线**。旧低成本N口径不被覆盖。此前完整520panel、encoder与anchor的完整历史账另外保留，不能把历史总成本都叫shared8或说全管线免费。run中verify的numpy重算SVD次数未计量为null，非0；不调用神经模型。墙钟真实记录，nested inclusive API/module时间不相加冒充total/FLOPs。仅声称固定新增向量对照，不声称等FLOPs、等wall或优势。

## 接口与封存

源 `experiments/spectral_reachability/agent_p3_resume_20261027_next.py`；测试 `tests/agent_p3_resume_20261027_next_test.py`。

- diagnose：完整旧consumer成功后输出逐臂paid=3/15/final、所有cost范围、proposal/winner gap、精确不同commands/endpoints和重叠候选一致性。不能把distinct latent当物理见证。
- prepare/verify：重建全部前序闭包，绑定自身source/tests/innerdoc、外部prepare/report及原实际artifact hashes，固定90候选与240请求，外部SHA+完整canonical相等。缺失/漂移即拒绝。
- queue：返回每臂60个literal float32[10,2]、root/task/bank/source身份、hash、H2包预算及四个winner回放依赖。
- run：仅MAIN审读后以显式ack在host CUDA启动；加载本地冻结模型，实际CUDA allocation、MATH/TF32off，复用原Native低层transition，不调用旧execute或改globals；本位不执行此分支。ack不是P1证书或安全权限机制。16MiB/200MiB floor/600s/3GiB allocated/4GiB reserved继承原限制；完整attempt-return/模块数组/轨迹/selection保存，异常保全不重试。
- consume：新report外部SHA、exact artifact census/hash、prepare完整重建；全240候选cost/context、literal dispatch、512个三帧context、1536模块输入输出链、实际shape、全部回放、argmin/selection及8分母独立复算。输出全部碰撞和零负modeldelta。未知backward_engine次数null。没有physical_release入口；所有P1/P3/freshness/advantage flag永false。

## 命令（cwd=/home/gbk/code/LeWorldModel/le-wm-main）

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/agent_p3_resume_20261027_next_test.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p3_resume_20261027_next diagnose --output results/agent_p3_resume_20261027_next_diagnose02.json
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p3_resume_20261027_next prepare --output results/agent_p3_resume_20261027_next_prepare01.json
# PREPARE_SHA使用上条实际输出；所有output均exclusive，不覆盖已存在文件。
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p3_resume_20261027_next verify --prepared results/agent_p3_resume_20261027_next_prepare01.json --sha256 "$PREPARE_SHA"
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p3_resume_20261027_next queue --prepared results/agent_p3_resume_20261027_next_prepare01.json --sha256 "$PREPARE_SHA" --arm D_rank
# 下面仅MAIN审读批准后在HOST CUDA执行，本位未执行：
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p3_resume_20261027_next run --prepared results/agent_p3_resume_20261027_next_prepare01.json --sha256 "$PREPARE_SHA" --output results/agent_p3_resume_20261027_next_cuda_MAIN01 --authorization MAIN_APPROVED_HOST_CUDA
# REPORT_SHA必须等真实封存后从实际输出取得，不填写未来哈希。
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p3_resume_20261027_next consume --cuda-output results/agent_p3_resume_20261027_next_cuda_MAIN01 --sha256 "$REPORT_SHA" --output results/agent_p3_resume_20261027_next_consume_MAIN01.json
```

## 精确未决与交付边界

本位已实际完成旧saved完整核验及diagnose01/02（诊断不调用模型），最终33项RAM通过15.54s，CLI --help exit0。diagnose02 SHA为 `07dabb686b41d455a0c8de6b49e6768a690ea487b8c72b14e0e0870f3d7b4136`，含自身source/前序来源绑定，diagnose01不覆盖，均为saved诊断不是研究报告。prepare/verify与最终冻结SHA另列，不在此虚填。测试含240请求、固定预算/minimum/tie、原42项逐位保持、3项续流、模块全桥接、八分母、失败无补步、bool身份/支付次数、cost/命令/模块/回放/包时序篡改拒绝，零负差保留。RAM backend不是预测证据。

尚缺：MAIN代码审读与显式host CUDA排期；新cold45bank的实际完整CUDA结果及ownconsume，尤其draw45..47从未评分，缺失不填0；P1独立机制准入；事前认证独立fresh有序root/goal及source去重、source/encoder/anchor全闭包与完整acquisition预算。当前两个known roots不能追认fresh，本对照不可充作独立显著性或因果泛化确认。新consumer未获实际CUDA运行验证，只RAM；没有新物理/控制收益、P1准入或泛化结果。
