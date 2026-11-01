# P3 同节点 H1/H3 HOST CUDA signal adapter

独占source/tests/innerdoc；不改原封存、共享总账或MAIN文件。不调用generate/replay/physics，不派生子代。experiment-code用于固定全分母、即时日志与失败回归；强制非零效果/训练/制图模板不适用，保留零、负与失败结果。

## 可执行范围

NativeSignal.signal(request, ledger)是真实冻结模型callback实现，不是接口清单。只消费runtime prepare01固定两known tasks的原根context、goal、identity Q与原float32 mean/std；每根固定H1/H3，不筛根、不读取物理outcome。暂只准精确原根的同节点请求；不把本根信号冒充未来子节点信号，不提供generate。

模型原生ABI为[1,3,192] latent、[1,2,10]历史token。新增raw[5,2]用原float32 subtract/divide，追加为[1,3,10]后走原生action_encoder→model.predict→predictor→pred_proj。动作坐标为直接raw float32零向量，不额外tanh/clip或/100；J=[192,10H]是d terminal/d raw。零是明确固定nominal，不是缺失补零。

nominal和AD分别独立forward，原精度数组逐次即时fsync。AE/predictor/pred_proj使用新加载模型上的临时instance forward包装并恢复；只观测/计量原生delegate，不改旧文件或任何module globals。predict由显式入口计量。所有input/output、raw/normalization、根与每包parent-child latent/history、nominal、AD terminal、J、192个onehot scalar target和raw gradient均保存原始float32字节+SHA。每个J逐192次torch.autograd.grad，不vectorize、不改动权重；每次实际raw leaf hook独立计费。4J共768 VJP/grad/raw callbacks，数学方向数不冒称backward engine次数。

HOST CUDA only，原模型hash须匹配4b12c01f5859f4b46bea478f3a8d5e2d3f516df456a622a3feec31bde7e33546。原state_dict与所有buffer（包括nonpersistent）前后独立hash，所有模块eval/参数冻结。prepare封存原source闭包、原weights/config、自身三文件、安装的stable-worldmodel/stable-pretraining/transformers/torch等Python源码；run逐native类源校验，最终完整重建闭包。Python source hash不是整套driver/CUDA二进制证明，host元信息独立保留；模型loader仅反序列化checkpoint到CPU随后搬CUDA，不在CPU做神经forward。离线环境禁下载；失败无CPU fallback。

## 计量与严格saved消费

事前工程profile：全局predict API/samples=16、context tokens48；AE/predictor各16calls/16samples/48tokens，pred_proj16calls/48rows；4J，768 VJP/grad/raw-hook，4array SVD，4signal compile。此次接线新增固定4次schedule计费，神经/AD/SVD账不变；encoder/projector/model-SVD/solve/generate/replay/physics为明确未调用0。nominal+AD每根2(H1+H3)=8predict。所有资源predispatch扣费，不退款；失败保留4项信号及4项task×method决策分母和未派发槽位，不重试、不换动作。未知backward engine/CUDA ns/FLOPs为null。wall为900s、output总cap仍256MiB、disk floor200MiB、CUDA allocated3GiB/reserved4GiB。新增完整core请求队列重复携带local_response，因此在任何CUDA执行前将该总cap内部终态reserve及JSON读取上限定为16MiB（旧8MiB），总输出/神经预算不增加。均工程拒绝上限而非科学阈值；不逐结果调宽。

单一exclusive追加events.jsonl保存完整原精度base64 buffers、哈希链和嵌套attempt/return，逐条flush/fsync。报告绑定prepare、artifact SHA和journal tail；输出仅新stem。finally保存完整rows/errors、模型/源/资源/cleanup状态，失败不释放信号。filesystem reserve或进程被强杀仍可能使最终report不可得，durable prefix保留但strict consume不放行。

strict consume不导入torch：校验外部report SHA、真实prepare/source闭包、artifact全集、原字节哈希链、实际resource记录与嵌套顺序、每步ABI/AE→predict→predictor→pred_proj wiring/normalizer、192个方向与J逐行raw-hook一致、nominal/AD两条完整路径，以及原精度数组。它从保存J计算float64 SVD一次/H，复算每H自身nominal残差chi及共同短H nominal残差chi；signed_gain=chi_short_ref−chi_long_ref可负，调度gain=max(0,signed)，保留二者。沿用原runtime relative cutoff1e-6，不新增科学阈值。秩亏条件数null而非伪0。该consumer验证保存数组内部一致性，不凭元数据认证GPU真实性或导数数学正确性；MAIN必须审读native trace和源码。

## 授权续接：真实signal到同任务type/horizon/depth

当前contract_revision为same_task_runtime_decisions_v2。直接调用未修改sealed runtime的policy()/choose()/request()，asdict完整封存每任务RuntimePolicy及其源SHA，不另实现可调阈值。沿用chi=.5、gain=.05、relative cutoff1e-6、base4/factor2、depth3、144 expansion cap；这是既有工程规则，不是新科学认证。共同短H残差的chi和max(0,signed_gain)进入带task+parent seal的FrozenSignals，own-nominal chi仅诊断，不能替换调度量。

每个原task的完整H1/H3数组先返回并持久，再生成固定顺序fixed/adaptive两决策；各自实际调用core.choose并计schedule attempt/return。adaptive高chi选reconfiguration，否则progress；gain达到沿用界限选H3，否则H1；高chi或gain达到界限允许depth3，否则depth1。fixed始终progress/H1/4 candidates/depth3，但保留相同真实信号。此处只共享每task一次局部信号获取，不冒充两个独立完整搜索arm或相同实际运行成本的收益对照。

horizon是每个程序chunk的packet数，depth是chunk深度上限，不把H3/depth3误报为总共3packet；未来最大累计9packet/45步仍须逐chunk真实生成、计费和独立回放。本轮未执行任何这些后续动作。

报告新增decision_rows、runtime_rule和revision，严格consume先完整重算saved信号，再复算同任务决策、FrozenSignals seal与完整root请求队列，拒绝跨根/旧parent/自调阈值、horizon/depth/type篡改、伪prefix、漏方法分母及免费schedule。队列沿用core请求字段，包含精确parent、空的真实root prefix、signal SHA、完整local_response、candidate index和选定horizon/type/depth。只是待compiler请求，不含虚构actions、路径或成功；program_plan/model_plan均null，队列status=blocked_missing_generate_and_replay。未来非根composition必须用真实返回且逐边闭合prefix，并在实际子节点新付H1/H3；本轮不制造子节点信号。

旧prepare01（SHA9dccc2f8193c263e1738b7dbf4a0ba2aaa6d1c0f4c01ddc0a270c78bccd6c71b）原字节保留并作为历史依赖；不能以它执行当前改动源码，严格verify应拒绝。新prepare02封存当前三文件与接线规则。没有重写或删除旧结果，更未把旧跨根零差包装成同任务C3收益。

联合测试中torch可由其他模块先导入。无新增torch导入的非法请求测试使用精确builtins.__import__拦截（即使sys.modules已有torch仍有效）；冷导入测试在独立纯导入Python子进程检查torch/gym均未出现，不改删sys.modules。该子进程不是子agent，不执行模型或CUDA。训练/强制非零效果等技能模板不适用。

## 边界和下一步

此交付只补signal缺口，不补program compiler/reconfiguration objective/chart或generate/whole replay。两known roots不是fresh；P1、P3、freshness、scientific_admission、physical_release及compute/walltime优势一直false。原PushT成功位置含agent+object，不能把object-only诊断称完整任务失败。既有跨根零差负结果不改、不据此声称C3测全。

尚需MAIN源码审读、主机CUDA明确调度、实际执行封存与strict saved consume；随后仍需generate/reconfiguration/replay独占实现、独立P1/适用P2准入、认证fresh任务与完整acquisition账、等预算对照与真实后果。本位不启动模型/GPU/physics，不编未来report SHA。RAM fixtures非研究资产、测试不等于科学证据。

## CLI（cwd=/home/gbk/code/LeWorldModel/le-wm-main）

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/agent_p3_local_signal_cuda_20261028_v1_test.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p3_local_signal_cuda_20261028_v1 prepare --output results/agent_p3_local_signal_cuda_20261028_v1_prepare02.json
# PREPARE_SHA必须取实际prepare输出，已存在不可重写。
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p3_local_signal_cuda_20261028_v1 verify --prepared results/agent_p3_local_signal_cuda_20261028_v1_prepare02.json --sha256 "$PREPARE_SHA"
# 以下run只由MAIN审读后在HOST终端执行；worker不执行。
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p3_local_signal_cuda_20261028_v1 run --prepared results/agent_p3_local_signal_cuda_20261028_v1_prepare02.json --sha256 "$PREPARE_SHA" --output results/agent_p3_local_signal_cuda_20261028_v1_cuda_MAIN02 --authorization MAIN_APPROVED_HOST_CUDA
# REPORT_SHA取实际完成report输出，不能填未来值。
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p3_local_signal_cuda_20261028_v1 consume --output results/agent_p3_local_signal_cuda_20261028_v1_cuda_MAIN02 --sha256 "$REPORT_SHA"
# 冷环境联合（sealed runtime原测试本身仍含全进程sys.modules断言，不在本写集中改它）
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/agent_p3_local_signal_cuda_20261028_v1_test.py tests/agent_p3_runtime_contract_20261028_v1_test.py
# 模拟MAIN其他测试已导入torch；仅import框架，不构造或运行模型。
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -c 'import torch; import pytest; raise SystemExit(pytest.main(["-q", "-p", "no:cacheprovider", "tests/agent_p3_local_signal_cuda_20261028_v1_test.py"]))'
```

个人完成条件：三文件实现、RAM变异/失败回归、saved prepare/verify或具体阻断、SHA与精确MAIN命令；不是研究goal完成。实际测试数和冻结SHA在交付消息列示。
