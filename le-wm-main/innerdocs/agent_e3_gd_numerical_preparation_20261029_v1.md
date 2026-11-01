# E3 known2 16cell GD数值准备（非生产runner）

只新增同stem source/tests/doc；冻结resolver与所有旧产物不变。known2、非fresh、非Full，所有科学门false。不训练，不进行NN/physics/fullstrict/生产prepare。本件实现可执行RAM callback优化核，不是把status包装称生产；原生CUDA backend、16格真实反馈调度、最终completion消费者尚未实现，不提供虚假run CLI。

## 已实现接口

`optimize_node(context,goal,K,domain,backend,tape)`：K严格1/3，完整33候选与32次scalar reverse，zero初始化/eta1，无warmstart/调参，逐packet原ball投影及原f32 inward cast算法，原NumPy tanh(f32) chart，Dx只mask新logits/gradient，raw_y正零。解析tanh链一次；fresh AD原始raw点与context精确相同，scalar loss必须f32 0d，raw梯度shape为5K×2；稳定(cost,index)选优之后另付三次完整replay。全部候选保留，失败attempt不退款。禁止生产backend，只有显式RAM_FIXTURE接口；测试FakeCUDA是解析线性RAM递推，不是CPU神经替代。

backend.packet接收完整latent3/history**2×10**与下一raw5，返回新完整context及最多32项f32 IO（每项至多768元素）；backend.reverse接收原raw与goal、必须且只能调用一次传入primal回放，返回fresh context/raw-gradient/scalar。检查latent滚动前两帧、原始raw按原mean/std恰好一次normalize后append为history3，再移除首行回history2；Dx的normalized-y不错误归零。RAM对齐数值算法，不声称验证真实模型导数。模型IO逐项连接仍需未来native adapter；不能把fixture的9项IO当真实native闭包。

`load_planner_inputs()`只读固定actual resolver prepare：`/home/gbk/code/LeWorldModel/le-wm-main/results/agent_e3_frozen_model_task_resolver_20261029_v1_prepare_MAIN01.json`，SHA `b44699e6a10bb4c257ed8872dc86b99a37508d7022850f8b4fcd257bcd232b26`（3,742,842bytes）。验证SHA/完整16ready/顺序/原模型根/identity192/原domain/task_body SHA后只返回16份`cell.task.body`，绝不返回完整cell/evaluator/native7；不回调resolver.verify，不重新读上游132MiB/report/fullstrict。此authority继承MAIN已完成own verify，不假称本件重建上游。`optimize_body(body,backend_factory,tape)`向factory也仅交白名单body；原mean/std从该body读取，不使用fixture默认归一化。只有直接调用底层RAM核且没有显式normalizer时才有测试默认值。

`axes27(K)`给成功节点预期的原27轴；不是已观察production receipt。predict/AE/predictor/proj各68K，samples同数，frame/projrows为204K；grad_API/VJP/rawcallback各32；replay3/metric36；其余本数值节点无调用。未调用原H2 receipt也不伪造其成功。

## 调用与资源预算

每节点(33candidate+32ADprimal+3replay)×K=68K predict、32 scalar reverse；每格3node：H1为204predict/H3为612，各96reverse。8格H1+8格H3：6528predict、1536reverse、19584context frame samples；非AD3456、ADprimal3072。新编码32/32是未来真实cp10/15协议预算，本数值核不编码。原完整路径480public/4864physics不在本工具执行。

原27轴无独立scalar-loss-backward、AD-primal及engine-backward轴，扩展单列：本算法torch.autograd.grad意图而非loss.backward，因此scalar_loss_backward=0；1536scalar reverse/VJP不能再相加，engine_backward与FLOPs均null。27轴mapping复用原resource_contract名称，不把未知轴填0，不跨不同goal/history/domain调用同任务可比性认证。均为预声明，无equal-FLOPs/总算力结论。

## 整数运输上界及边界

采用独立未压缩NPY+JSONL，因此gzip/Z_SYNC_FLUSH开销精确0，不假设压缩率。每packet最多32数组、每数组最多768个f32（3072bytes）＋4096bytes NPY头保留；重复IO不去重。每格3node还保留384附加数组（实际kernel为300），事件最多4×packet+784，每event payload≤16384bytes另LF1byte；原生每packet IO将由单一packet事件承载，不另偷偷加模块事件。metadata16MiB与failure/completion64MiB另计。

预算由`transport_budget(K)`整数生成：arrays=32×(204K)+384；events=4×(204K)+784；总额=arrays×7168+events×16385+16MiB+64MiB。均必须≤512MiB。Tape写前单数组/单event守卫与工作区448MiB上限；本工具尚未提供整job native事件inventory签名，故`native_array_inventory_certified=False`。**这是有界候选运输合同的fit证明，不是未来真实模型所有IO已落在该合同内的证明。**原完整encoder/RGB/native运输未纳入；集成总512MiB仍需另列原生/编码inventory验证，不能用本上界自动放行整闭环。

| 每格3node数值运输 | H1 | H3 |
| --- | ---: | ---: |
| predict packet | 204 | 612 |
| arrays含重复＋附加 | 6912 | 19968 |
| arrays NPY含头上界bytes | 49545216 | 143130624 |
| events含LF上界bytes | 26216000 | 52956320 |
| metadata＋failure/completion reserve | 83886080 | 83886080 |
| 完整数值运输上界bytes | **159647296** | **279973024** |

raw leaf及gradient分别H1=10f32/40bytes、H3=30f32/120bytes；loss标量4bytes；每packet完整latent=576f32/2304bytes，history=20f32/80bytes，outgoing=30f32/120bytes，输入raw5=10f32/40bytes。最大数组768f32是上述原生primitive各保存副本的保守包络，不是模型activation显存估计；future native模块若出现更大IO或更多副本须blocked，不能删IO凑预算。采用未压缩NPY避免依赖“预计压缩”与逐event flush，实际最大32数组单event临时磁盘测试覆盖头与descriptor费用。

每job1800s，16job cap之和28800s不是串行实耗；从verify前到report fsync/hash及独立completion。Tape可注入该started并在dispatch前检查，RAM覆盖已耗尽时不dispatch；本件不出版带inclusive声称的report。生产completion接线未实现。GPU allocated3GiB/reserved4GiB是原硬限，不是shape可推保证：K3 AD图、参数/缓存/workspace峰值未知。无真实NN运行故耗时估计、GPU峰值/fit均null/false；MAIN未来需源闭包、冻结权重before/after、实际native IO、runtime guard与completion审读后再批。

## RAM、真实临时磁盘与MAIN命令

`Tape.put/emit → read_tape`使用真实NPY文件和单LF hashchain，fsync在open文件内部。build绑定自身三文件、冻结resolver源与resource源；verify整个build重建，自绑定文件漂移拒绝。测试仅tmp目录：完整H3实际32步→reader、scalar0d、数组SHA篡改、source证明篡改、排他写、失败不退款、超时不派发、Dx正零、4种AD ABI错误。

在 `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_e3_gd_numerical_preparation_20261029_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_e3_gd_numerical_preparation_20261029_v1 inspect
.venv/bin/python -B -m experiments.spectral_reachability.agent_e3_gd_numerical_preparation_20261029_v1 inspect-inputs
```

inspect只stdout输出源绑定预算，不创建生产结果，不读取旧实际outcome；inspect-inputs只定向读取上述3.74MB actual prepare，打印16份body digest/根/horizon/domain，不优化任务。未来若MAIN保存该proof，可用`verify --saved CANONICAL_ABSOLUTE_PATH --sha ACTUAL_SHA`验证；没有prepare/run神经CLI。本件不重复resolver实际输入重建或上游fullstrict，不把16格ready当未来runner授权。
