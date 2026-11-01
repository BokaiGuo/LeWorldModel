# JEPA native action-word：程序生成与两步预测接线

仅新增本stem source/tests/本文；未改旧P5、sealed结果或共享docs。按MAIN最新要求不继续新增JVP重复验证：本轮最初未封版的pullback源草稿已在自身未交付写集中重命名/改造成本文对应action_words源；没有执行该草稿，没有遗留待运行的pullback交付。使用experiment-code技能组织固定程序、实际日志和失败测试；不采用通用训练/按结果调优/绘图要求。没有新架构、teacher、adapter、repair、拟合、训练、CUDA/CPU神经、物理或网络执行。worker只做RAM接口测试。

## 补的具体缺口，不是假装已有task witness

已有PushT JEPA native v12是三根一包四臂factual fidelity；v8及prospective C1已有JVP/FD与单根ridge编译。本模块都不重跑。新增的是**原生两包action-word生成、预测视觉与预测proprio共同反馈、完整历史移位、全部候选原始返回保存与独立saved复算**。每根四个固定10步program，不优化、排名或选winner：XY、YX、X−X、零零。X是连续5步native relative u=[.1,0]，Y是5步[0,.1]。XY/YX具有相同边际控制与不同次序；取消与零分支是描述性对照。输出只是**model-only action-witness候选**，没有真实接触/物体进展的witness认证。非零预测差异也不能自动称C1任务前提、非贪心收益或P5。

三根仍为已打开的2026102702/03/04，不称fresh，不换根。JEPA-WM AdaLN patch384+proprio相对LeWM是独立动力学候选族；当前仍是原PushT checkpoint，不是新checkpoint或第三架构；换到JEPA PointMaze也只增加该族环境/权重，冻结normalizer的先前阻塞不消失。

## 合法接口与精确ABI

使用真实v12 manifest文件SHA `41ef9bf8f5c5ae5c4ff8890a25e6a83137b7e637058a8033281a5eabc9cea07f` 与实际CUDA报告SHA `d8667a4e8537a5e005d13507d4d05b8d7907996549889decc8a00e1db873eda6`；canonical manifest摘要另按原json.dumps(sort_keys=True)口径验，不与文件SHA混淆。沿原988项bindings核原源/数据/权重；补绑定v12保存的patch数组及本三文件，不扫描全库。载入原v8的load_cuda，源SHA `fc915ddee2e53ed58366994684fe2190da5eaaeb8323ad608ffe3292534c936a`，**不调用v8.prepare**、不改其全局路径。实际严格load及模块receipt必须与v12一致，失败就阻断。

非神经prepare只导入stdlib/NumPy，不import torch/native dataset，不解pickle。精确读取已保存float32 patch [4,256,384] 的前3帧0/5/10为视觉context（文件含frame15，读取后立即排除；frame15不进模型或指标），原normalized proprio [3,4]，原normalized action [3,10] 中前两包作真实between-state历史。每个plan包含raw64 [2,5,2]和normalized32 [2,10]；投影严格为 `(100*u64).astype(float32)/float32(100)` 后减原source mean32、除std32，time-major，不套tanh、不裁剪、不重算统计。物理零不是normalized零。[-1,1]仅是已有bridge采用的保守native-u策略域，不谎称native advertised absolute-position Box。

直接原predictor调用的输入为visual [1,3,1,16,16,384]、actions [1,3,10]、proprio_features [1,3,256,16]；原返回tuple应为visual [1,3,256,384]、None、proprio [1,3,256,16]。原生`VideoWM.forward_pred`的AdaLN路径只将patch reshape回空间网格；原config normalize_reps=false。其source SHA `657bd3e9eb2ec35442cc06e208b4632a53503a35a83a96abbbf82608d2fe9154`；`EncPredWM.unroll`的predict_proprio分支SHA `105a6768050e4b34c95de8d7d0cc0c7ba07b69d364acf0649ba2625bacce907b`。本模块对应其取last visual/prop、追加并滑窗的递推，但**不调用unroll里会补normalized-zero历史的padding分支**：第一步actions=[原packet0,原packet1,program0]；第二步=[原packet1,program0,program1]，visual=[原frame5,原frame10,pred15]，proprio=[原feature5,原feature10,pred_feature15]。禁止拿真实未来proprio/RGB、初始proprio或别的program终点替换预测反馈。

每根只一次原proprio encoder，得到[1,3,1,16]后expand256，跨四program共享且**只计一次**；每个program独立从同一context开始，不能共享已预测prefix“退款”。官方encoder在严格load闭包内但0forward。逐次predict输出float32原值同步拷到CPU持久化，再原值回CUDA作下一步输入；CPU仅做concat/reshape及saved float64均方差算术，不跑CPU神经。SDPA MATH、TF32 off、原eval/frozen状态由原loader设置，前后module state receipt必须一致。无feature正则化、输出修补或readout。

## 预算、保存与consumer

每根predictor=8 API /8forward samples、proprio encoder=1 API/1forward sample；三根总predictor24/24、proprio3/3；batch固定1，predictor每次3context frames，合计72 predictor context frames、9 proprio frames。视觉encoder/AD/backward/JVP/VJP/SVD/physics均0；构造/加载状态校验不是神经forward，也不伪装为免费总算力。参数构造、权重加载、GPU拷贝、同步、hash及saved写入墙钟实际单列；FLOPs=null，不能把轴相加或称等总算力。forward prehook持久化attempt，posthook记录Python API return；返回后同步并验finite；失败不退款、无额外replay或补步。Python return不等于独立kernel计数。

固定96MiB output cap、200MiB free floor、600s wall、4GiB CUDA free scratch工程前置（不是峰值内存上界）；启动和每次dispatch/数组写前检查，SIGALRM只在HOST同步Python执行可响应时中断，不是不可抢占CUDA kernel的硬实时保证。失败保留12program分母所在3root行，当前failed、剩余not_run；保留已写输入/输出/attempt-return。异常清理forward hooks和alarm，保留原错误与finalization错误。失败后小terminal receipt仍尝试写入，不许删除partial重跑覆盖。MAIN需另用外部HOST进程超时保障硬墙钟和OS失败；磁盘耗尽/进程kill可能仅余partial事件，不可consume为成功。

每步保存visual/proprio/actions实际输入和visual/proprio完整原返回；41数组/root，123数组总计，原生维度约60MiB，96MiB含事件有余量。事件独占创建、SHA链、每个root terminal；consumer重放计数、文件序号、sample/frame轴，按逐位concat重建第二步视觉/proprio/actions，检查完整12program数组与SHA、全部对比算术。consumer不再运行模型，不证明重新推理正确或远端官方真实性；它只验证MAIN受审代码产出的保存记录自洽。不存在命令release或physical-witness签名。source root/goal绑定带原factual目录、root_sha256及frame10原full_public_state文件SHA，但不拿其中privileged state做模型输入、goal优化或挑候选。

## MAIN准确入口

cwd `/home/gbk/code/LeWorldModel/le-wm-main`。先审读三文件、测试；输出目录由MAIN选择新的同stem直属results子目录。先显式建立空的新prepare目录，例如（存在时mkdir失败，不覆盖）：

```bash
mkdir results/agent_p5_native_action_words_20261028_v1_prepare_MAIN01
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p5_native_action_words_20261028_v1 prepare \
 --manifest results/agent_p5_native_action_words_20261028_v1_prepare_MAIN01/manifest.json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p5_native_action_words_20261028_v1 verify \
 --manifest results/agent_p5_native_action_words_20261028_v1_prepare_MAIN01/manifest.json
```

仅MAIN批准后HOST CUDA（worker未执行）：

```bash
env -u PYTHONPATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
 .venv/bin/python -B -m experiments.spectral_reachability.agent_p5_native_action_words_20261028_v1 cuda \
 --manifest results/agent_p5_native_action_words_20261028_v1_prepare_MAIN01/manifest.json \
 --output results/agent_p5_native_action_words_20261028_v1_cuda_MAIN01
sha256sum results/agent_p5_native_action_words_20261028_v1_cuda_MAIN01/report.json
```

成功返回exit0/status native_action_word_predictions_recorded；失败exit1/status failed，前置异常非零并保留已有partial。MAIN记录上条命令的**实际report文件SHA**后，填入下面变量，不预造未来hash：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p5_native_action_words_20261028_v1 consume \
 --manifest results/agent_p5_native_action_words_20261028_v1_prepare_MAIN01/manifest.json \
 --report results/agent_p5_native_action_words_20261028_v1_cuda_MAIN01/report.json \
 --report-sha256 "$MAIN_ACTUAL_REPORT_SHA256"
```

consume成功为all12_saved_program_predictions_verified，仍physical_release=false/scientific_admission=false/action_witness_established=false。消费后不追加文件到原CUDA输出目录，以免改变其outputcap census。MAIN若要保consumer stdout，应另用独占新路径，不覆盖输入。

RAMtests（tiny NumPy轴/解析callback，绝非真实模型证据）：

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p5_native_action_words_20261028_v1.py
```

实际结果：**35 passed（1.75s）**；CLI --help通过。包含完整tiny producer-layout→consume、重写SHA链后仍拒绝第二步proprio substitution、source/manifest漂移、坏数组/外部reportSHA/缺word/计数/事件链/伪release、API超cap/batch、free-floor/wall/output、native完整123数组尺寸回归及模拟CUDA不可用时保留12分母且无CPU回退。首轮fixture把30元素action数组与tiny视觉18元素上限混用导致12失败，已将上限显式涵盖action/proprio/visual，再通过全部测试；不删失败记录、不把fixture虚构的cuda_executed布尔当真实CUDA证据。worker未对真实输入调用prepare/verify/consume或cuda。

## 剩余科学/资产前置

本地合法原生预测接口已确认，不需要PointMaze冻结统计来做当前PushT接线。但本轮未实际prepare、严格CUDA load或预测，实际reportSHA unknown；988项闭包可能因源漂移、临时DINO源码tar丢失而失败，必须保留阻塞，不能下载/换source绕过。原checkpoint许可、远端revision独立认证仍沿原候选边界，不能凭本地hash扩大合法性主张。

真实action witness仍缺：独立预声明的任务/goal/接触可操纵性前置、精确native fresh reset+原prefix10全帧重建的审查和物理adapter、实际raw10执行及earlystop/失败分母、原生物体pose/coverage与agent-only变化分离评估、合法horizon与模型递推对真实任务校准。当前任意word对比不能保证接触、成功或动作域在native实际执行中的安全性，不能按预测差异挑root/program物理追成功。本模块不释放物理命令，后续MAIN若执行须另立明确协议，不借本工具将known根升级fresh。C2同根非贪心能力/任务收益合取、强基线predict预算与P1/P5独立准入依然缺失；不放宽门槛。
