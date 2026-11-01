# P3 equal-cap parent A：真实HOST接缝与RAM回归

仅本stem source/tests/doc，旧源码/结果/协议不改。使用experiment-code技能的独立实现、保存与回归要求；不训练、不拟合、不调阈值/候选、不建agent、不在worker加载权重/生产prepare/NN/physics/fullstrict。MAIN审读冻结后运行。

## 固定合同与复用边界

协议SHA `36dc4e70eefd1a361ca4c7bc8a8181ea587c381640b85679921fa86bccb46861`。共同B逐轴复制原parent `default_caps('adaptive')`，两策略都562predict/34J/6528VJP上限，完整27轴，不改原defaults/choose。每arm W=3600s；整job18000s；输出1GiB含256MiB final reserve，disk floor200MiB，CUDA allocated3GiB/reserved4GiB。四槽root0 fixed/adaptive、root1 fixed/adaptive，每槽144请求位，共576。unknown scalar backward/AD-primal/J元素/FLOPs保持null，不签equal actual/FLOPs/wall/P1/P3/C3/freshness。

显式复用冻结parent.run_core的caps入口、原NativeTree模型/信号/完整history/replay方程、model_sources、原native buffer/ledger校验与SavedBackend。新prepare/report/consume均自有schema，不调用旧run/consume、不重标旧report，不复制旧驱动再改名字，不patch globals。build复用旧build的真实task/model/source authority后增加本三件套/协议/base源码身份；MAIN prepare/verify完整重建，worker不调用build。

R请求原样整8拒绝且plan=false，原progress child policy/beam/depth/稳定rank保留。失败attempt不退款、无padding/重启，fixed114正常停止不强补到562；不同actual不改口匹配。unsupported/cap仍整树无release并继续下一固定槽；runtime/deadline阻塞后续槽，但全部槽/请求位分母保留，blocked费用unknown。

## 真实计时/同步与保存

Timer每arm首个CUDA同步后建立时钟，调用前后同步并检查真实precision/memory/output/whole-wall守卫；同步用torch.cuda.synchronize(cuda:0)，单调纳秒计时，完整guard回执、sync attempted/returned、elapsed/overrun及失败写入独立timing事件。elapsed含policy/IO等待，加载/前后来源核验另列whole-job，不叫纯GPU time或equal elapsed。

core非native资源只在resource_attempt已记账后检查，不能在持久resource_return回调抛错形成双terminal。native用本地TimedLedger代理原Ledger，每实际调用前/返回后同步；返回工作已记completed，随后发现超时不退款，异常向外正常展开。中间raw/VJP/预测错误保留已返子工作和失败父。新Backend显式记录native_call attempt/return/failure并延迟构造真实NativeTree，任何入口检查失败仍有完整native disposition；无AD/模型方程变更。

arm_final总做同步并记录。即使原core在最后同步前正常返回，deadline/guard失败也令本driver `plan_available=false,released_plan=null`，后续槽blocked；原core result仅作为保存诊断，不能从其model_plan绕过本driver release。Timer不声称抢占已运行kernel，overrun原样保留。

新Store沿原hashchain/fsync格式，wall/memory guard不放在已提交return后；写入/磁盘硬cap始终检查。普通journal allowance耗尽时，触发记录尚未落盘；即使原ledger已append/收费，也不能把缺失事件的账称为完整。recovery仅允许失败timing、资源failure/return展开、native failure、arm_return及来源/模型/precision收尾，禁止新dispatch，普通check直接失败；在事前final reserve内尽力写unwind/报告，不重试实际计算，不放宽1GiB总cap。此类输出是partial-only，完整strict必须拒绝，不属于可验证partial成功；触发点为return时也不修造缺失事件或双terminal。reserve不足/磁盘失败可能连收尾都无法写全，亦不伪称完整。

## 自有saved consumer

MAIN consume验证实际report/prepare SHA、完整source/model/classes/precision、HOST preflight、全部artifact字节/链、四槽576、原27轴、B/W和guard placement。计时事件不能随意filter忽略：逐probe时序/elapsed/overrun/guard独立核验，并与实际resource/native attempt/return的前后位置一一绑定；出现guard failure后不得新dispatch。

原生数组/packet/AD/VJP/J/完整raw/context数学用冻结低层audit验证；原parent.run_core通过惰性SavedBackend重建同caps原choose/queue/树。native内的失败沿原失败资源树重放，core非native predispatch失败按唯一实际op index注入对应RuntimeError，再核完整原result；不重跑模型、不篡改旧全局或旧异常表。不用计时重算值冒充实际时间；只对原algebra比较排除原已定义的测量timestamp字段，原timing本身完全另核。

完整settled与可验证partial分开，R合法拒绝可strict成功但无plan。加载/setup/模型守卫等未具备完整原生闭包的故障保存报告但consume拒绝，不签假完成。consumer无RAM自动回退；只有RAM tests显式permit_fixture。

## MAIN命令

在`/home/gbk/code/LeWorldModel/le-wm-main`，全部actual SHA使用前一步返回值，无未来SHA硬编码：

```bash
.venv/bin/python -m pytest -q tests/test_agent_p3_equal_cap_parent_20261029_v1.py
.venv/bin/python -m experiments.spectral_reachability.agent_p3_equal_cap_parent_20261029_v1 prepare --output results/agent_p3_equal_cap_parent_20261029_v1_prepare_MAIN01.json
.venv/bin/python -m experiments.spectral_reachability.agent_p3_equal_cap_parent_20261029_v1 verify --prepared results/agent_p3_equal_cap_parent_20261029_v1_prepare_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA
.venv/bin/python -m experiments.spectral_reachability.agent_p3_equal_cap_parent_20261029_v1 run --prepared results/agent_p3_equal_cap_parent_20261029_v1_prepare_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA --output results/agent_p3_equal_cap_parent_20261029_v1_cuda_MAIN01 --authorization MAIN_APPROVED_HOST_CUDA
.venv/bin/python -m experiments.spectral_reachability.agent_p3_equal_cap_parent_20261029_v1 consume --prepared results/agent_p3_equal_cap_parent_20261029_v1_prepare_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA --output results/agent_p3_equal_cap_parent_20261029_v1_cuda_MAIN01 --report-sha ACTUAL_REPORT_SHA --receipt results/agent_p3_equal_cap_parent_20261029_v1_consume_MAIN01.json
```

仅run导入torch/加载权重，且先检查MAIN授权字面量。不提供物理release、不含R生成器、不签等实际资源/FLOPs或科学gate。

## 测试与冻结边界

首次RAM回归1 failed/20 passed：pre-backend guard失败的backend_kind实际为null，已限定为零native调用且真实core_predispatch失败时接受，不伪称CUDA。随后22 passed/51.43s；加入final同步抑制release、真实Store partial持久化和reader回归后24 passed/69.69s。新增普通allowance耗尽窄real-Store回归1 passed/0.34s，证明已收费未落盘attempt不退款、无native dispatch、后续槽阻塞、拒绝recovery新工作且完整strict拒绝。最终全套session50468：25 passed/88.08s；三文件git diff --check通过。全部为RAM/临时文件，不是生产证据；MAIN尚需独立审读/prepare/真实HOST CUDA/完整consume。
