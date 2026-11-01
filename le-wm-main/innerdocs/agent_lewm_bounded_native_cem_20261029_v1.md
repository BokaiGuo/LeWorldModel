# bounded-native-CEM known2：实现交付

本版为新方法，不是官方CEM复现，不称等预算/等FLOPs、Hi训练scaler认证或P1准入。
H2与P2共同controller H3的预测时距和资源不同，不能直接排名。保留全部known2/off，非fresh。
前瞻protocol：`agent_lewm_bounded_native_cem_20261029_v1_protocol.md`，SHA
`b3f9527cf9b73b81560bfe42736be8a38845f3527dd4ae301af88c389b7cf23b`。
seed0为MAIN本轮事前固定，安装默认1234保留；不改变原已封存官方mean越界结果。

## 接线与身份

独占source/tests/本文，与protocol分开冻结。旧native CEM与paid mean evaluator仅提供
已审输入闭包、原loader、Store编解码、原生module hooks和saved验证函数；不monkeypatch旧globals，
不调用旧CEM结果选择器，不重消费旧完整fullstrict。prepare复核原任务、blind RGB、原编码、
模型weights/config、原normalizer以及闭包所有source SHA，保存原两根顺序及mean/std。
原actual0/5/10 RGB/context和warm2不变。原8encoder/8projector继承单列一次；新增encoder0。
文件SHA、loader身份和前后model tensor hash不是同buffer loaded-bytes证书或训练统计证明。

安装版CEMSolver N300/I30/topk30/batch1/var_scale1，独立每root seed0、零mean/单位std。
候选和elite更新均在logit空间，std沿torch correction1；sample0=currentmean。
每候选一次CUDA-f32 tanh，raw[-1,1]精确检查，原P2 normalize_live_packet一次；无clip/逆变换/除100。
完整保存logits→raw→normalizer input/output→warm2+future2→nativeIO→cost→callback更新。
目标不进predictor，只供criterion。每轮300完整预测先durable再criterion/update，不能称全部30轮先预测。

最终返回logit mean，一次tanh成为raw10，落盘后由严格NPY reader重新读取实际字节，
再normalize一次用于独立whole2/segmented1+1实际重放。segmented第二步只用自己的pred15移窗。
全两根四路预测先seal才算4个final criterion。elite_average_cost明确不是交付mean评分。
全future384/frame15/terminal20均报bitwise/maxabs/ULP，不升级parity、不给物理release。

## 计量与验证边界

全job：cost-oracle60、官方model.get_cost0、rollout62、segmented predict-wrapper4、criterion64；
AE66API/18006samples/72020tokens；predictor128API/36008samples/108024contexttokens；
pred_proj128API/108024rows；tanh62/normalize62。encoder/projector/AD/physics/training0。
wrapper不能与内部module相加作总predict。两根各9000候选，失败保留完整两根槽和attempt/return。
规划hooks拒绝encoder/projector，检查真实CUDA-f32/shape/顺序/inference/autocast/TF32；
final hooks同样限制11dispatch/root。criterion期间所有神经模块调用均拒绝。

输出1GiB含64MiB reserve，disk floor200MiB，wall900秒含verify/load；GPU allocated3GiB/reserved4GiB。
临时fixture完整shape保守census含每文件64KiB overhead，上界825,591,808 bytes，低于cap减reserve；不缩精度/分母。
这些是合作式边界，不是抢占式CUDA中止。partial保留durable事件/数组，严格consumer拒绝partial。
MATH只称requested context，实际观测守卫不冒充kernel证书。

saved consumer检查全文件census/SHA/NPY header、单换行canonical JSONL hashchain、完整事件顺序、
所有candidate/nativeIO/update、source/goal/history、final raw重载和部署tokens、四路因果链与criterion。
绑定的CUDA tanh实际调用及其保存输出为mapping来源证据；NumPy float64 tanh→f32仅另列
bitwise/maxabs/ULP诊断，**不**用CPU libm相等认证CUDA数学，也不以容差放行mapping。
这不是在无模型情况下重新执行CUDA tanh的证明；若source-bound调用收据不可信，此消费者不能补证。
normalize由NumPy f32单独subtract/divide精确镜像检查，若实际CUDA除法字节不同则failclosed，
不能据生产差异新增容差。原criterion/elite reduction仅沿旧封存的浮点累加误差界；
该界不用于whole/seg parity或raw domain。

## MAIN执行命令（worker未执行）

工作目录 `/home/gbk/code/LeWorldModel/le-wm-main`，无PYTHONPATH覆盖。PREPARE_SHA/REPORT_SHA
必须取各自实际输出，不能预填未来值。输出路径不得已存在；无重试/自动补跑。

```bash
.venv/bin/python -m experiments.spectral_reachability.agent_lewm_bounded_native_cem_20261029_v1 prepare --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_lewm_bounded_native_cem_20261029_v1_prepare_MAIN01.json
.venv/bin/python -m experiments.spectral_reachability.agent_lewm_bounded_native_cem_20261029_v1 verify --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_lewm_bounded_native_cem_20261029_v1_prepare_MAIN01.json --prepare-sha "$PREPARE_SHA"
.venv/bin/python -m experiments.spectral_reachability.agent_lewm_bounded_native_cem_20261029_v1 run --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_lewm_bounded_native_cem_20261029_v1_prepare_MAIN01.json --prepare-sha "$PREPARE_SHA" --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_lewm_bounded_native_cem_20261029_v1_cuda_MAIN01 --authorization MAIN_BOUNDED_NATIVE_CEM_HOSTCUDA_ONLY
.venv/bin/python -m experiments.spectral_reachability.agent_lewm_bounded_native_cem_20261029_v1 consume --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_lewm_bounded_native_cem_20261029_v1_cuda_MAIN01 --report-sha "$REPORT_SHA" --receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_lewm_bounded_native_cem_20261029_v1_consume_MAIN01.json
```

## RAM回归与未运行项

```bash
.venv/bin/python -m pytest -q tests/test_agent_lewm_bounded_native_cem_20261029_v1.py tests/test_agent_lewm_native_cem_slice_20261029_v1.py tests/test_agent_lewm_finalmean_paid_evaluator_20261029_v1.py
.venv/bin/python -m experiments.spectral_reachability.agent_lewm_bounded_native_cem_20261029_v1 --help
```

fixture仅新模块N4/I2/K2，旧globals/solver不改。真实临时Store写两根全部数组、raw重载、
canonical journal、完整saved reader；含mapping/domain/normalize/更新/因果链/缺槽/资源/封存顺序篡改拒绝。
实际本版41tests及旧helper联合102tests通过（13.61s）；CLI help exit0，冷导入无torch/pymunk。
直接mapping/normalizer接线使用NumPy-backed fixture tensor，不冒充真实CUDA数学验证。
生产prepare/CUDA/physics/fullstrict均未运行；测试不是预测、控制收益或科学通过。
MAIN必须实际审读、prepare并HOST CUDA后才能产生本版真实数值，不能借旧mean值预填。
