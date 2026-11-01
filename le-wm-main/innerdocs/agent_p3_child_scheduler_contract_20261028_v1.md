# P3 observed-child预算调度合同

仅新增本stem source/tests/doc。旧P3cost及全部producer/封存结果继续冻结。本worker不执行神经、CUDA、物理、网络，不创建下代或写MAIN结果。

## 已复用与实际交付

已核现有P3 runtime的`local_signal/policy/choose/Ledger`、root-only CUDA signal adapter、common-budget合同，以及P4 moving multiboundary/rolling saved consumer。现有runtime已能callback逐节点；这里不重写算法，也不把旧根信号作为新child信号。

`prepare`完整调用P4 rolling visual seam consumer，绑定外部真实report/prepare SHA，读取自己的PNG编码/native192及实际返回raw动作，重建原float32 subtract/divide历史。P4全部8episodes×3boundary=24child槽位，每槽fixed/adaptive两方法=48分母；earlystop未到达保持not_reached/null，不补图、不挑成功。step20只终点评价，不能成为控制child。两known roots，不是24独立任务。

I和R均重建**真实观测**context：I的旧compiler使用imagined context，不能冒充真实RGB。新context以原root历史逐次追加本episode的actual encoded latent，保留旧compiler context SHA供追溯；R额外exact比对原request context。每child只暴露task/goal/Q/model/normalizer、截至该boundary的实际raw prefix/last10、RGB/latent身份与事件顺序。native state/snapshot/reward/后续动作/终点outcome不进入调度请求。离线身份consumer会读取原terminal report校验完整性，这不是用其未来结果选配额；当前实现不是在线实时控制器。

P3两份独立实际paid signal panels经原`load_panels`完整consume，保留每根8predict/2J/384VJP等账，**仅为原资源profile参考**。P4 moving与P3原根不是同任务，绝不跨源借J/chi；不同task明确保留。source/inputs全SHA闭包与自身三文件写入prepare；verify重新构建并拒绝漂移。

真实字段接线：moving export的`native_root_sha256`实际为null（不是64hex），禁止塞进原runtime binding或放宽validator。prepare沿已绑定original_manifest→sourcepack读取明确row.root_sha256，严格匹配source_slot及blind_payload SHA；export原null不变，保存sourcepack身份边。task_contract仍用原spaced JSON codec核验，不用runtime compact digest替代。此为metadata身份桥，不新增物理等价/freshness证明。

## 可执行接口，不是cost报告

CLI `prepare/verify/plan`均saved-only，无run开关。`plan`生成完整48槽、精确parent context/task binding的H1/H3 acquisition请求；缺真实child信号时明确blocked，不默认chi=0或progress。

Python `plan_child(child,tasks,backend_factory,sharing=...,permit_ram=False,authorization=None)`会实际通过原`core.local_signal`请求同child H1/H3，计量nominal+AD、完整native模块hook、384方向、两次SVD，再调用未修改`core.choose`生成候选请求。默认拒绝RAM/CPU adapter执行；RAM必须显式permit_ram，未来HOST callback需MAIN授权`MAIN_APPROVED_CHILD_SIGNAL`且提供durable event_sink。backend元数据/哈希不是GPU真实性证明；MAIN尚需审核新child adapter与独立saved consumer。callback失败保留paid attempt/error，不重试、不填信号，全部方法槽仍保留。

两种事前固定公平口径：`independent`每方法独立完整8predict/2J；`shared_equal`只在**同一个child**实际计算一次完整信号，两方法引用同一group，并各记有理数1/2份实际acquisition（不能把8次改写成两个“独立8次”，也不称免费）。不允许跨child/cache借用；原encoder获取成本继承P4而非免费。schedule每方法独立实际记账。

原分配规则完全不调：chi=.5、gain=.05、SVD cutoff1e-6、候选4/8、H1/H3、depth上限3。observed child是新局部树的depth0；physical step/5不是模型组合深度，真实执行prefix不伪造模型edge。独立保留搜索cap24predict（原controlled budget）、selected-prefix reserve1；原规则若给H3×8，两个回放需48超cap，保留原请求整节点拒绝，不缩为4、不偷偷扩预算。这里**不套用**旧root common-budget显式12/4 override。未用quota不padding；相同cap不代表相同实际算力。此接口只返回候选请求，commands=None，compiler/reconfiguration/replay/物理仍待真实接线。

## MAIN命令与缺口

MAIN提供绝对路径JSON输入manifest，字段严格如下（所有SHA均必须来自实际文件，不可编未来SHA）：

```json
{"p4":{"output":"ABS_P4_DIRECTORY","report_sha256":"ACTUAL_REPORT_SHA","prepared":"ABS_P4_PREPARE","prepare_sha256":"ACTUAL_PREPARE_SHA"},"p3":{"fixed":{"output":"ABS_FIXED_SIGNAL_DIRECTORY","report_sha256":"ACTUAL_FIXED_SHA"},"adaptive":{"output":"ABS_ADAPTIVE_SIGNAL_DIRECTORY","report_sha256":"ACTUAL_ADAPTIVE_SHA"}},"sharing":"independent"}
```

真实P4 MAIN02已知report SHA `60489b7db3d4eff70d1be46d47d92209e6adcf333ae1d0a0e389b259ec1f6084`、prepare SHA `8528e8d7f9b688b1c93be154305ccc35b8ed6218de63e736b1ae567691c432a2`。本任务不运行实际prepare。rolling consumer只验证visual seam，不替代MAIN完整native/budget审计，prepare保留这个限制。P3 panels不能用root-only报告“认证child”。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
OPENBLAS_NUM_THREADS=1 /home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -B -m pytest -q -p no:cacheprovider /home/gbk/code/LeWorldModel/le-wm-main/tests/agent_p3_child_scheduler_contract_20261028_v1_test.py
# CHILD_INPUTS_SHA和CHILD_PREPARE_SHA取MAIN实际文件SHA；不填写不存在的产物SHA。
/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -B -m experiments.spectral_reachability.agent_p3_child_scheduler_contract_20261028_v1 prepare --input /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_child_scheduler_contract_20261028_v1_inputs_MAIN01.json --sha256 "$CHILD_INPUTS_SHA" --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_child_scheduler_contract_20261028_v1_prepare_MAIN01.json
/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -B -m experiments.spectral_reachability.agent_p3_child_scheduler_contract_20261028_v1 verify --input /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_child_scheduler_contract_20261028_v1_prepare_MAIN01.json --sha256 "$CHILD_PREPARE_SHA"
/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -B -m experiments.spectral_reachability.agent_p3_child_scheduler_contract_20261028_v1 plan --input /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_child_scheduler_contract_20261028_v1_prepare_MAIN01.json --sha256 "$CHILD_PREPARE_SHA" --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_child_scheduler_contract_20261028_v1_plan_MAIN01.json
```

只写results内本stem全新文件，排他xb/fsync、32MiB metadata cap，不覆盖封存。CLI plan不会跑callback或生成动作。

真实saved投影回归确认P4 MAIN02为23个observed child、1个未到达（e01_R_b15）；保留全部24 child/48方法，其中46方法槽缺新child信号，2槽not_reached。**不能判定全部已知child因预算拒收**：没有真实child H1/H3，measured_search_budget_refusals及all_observed_children_budget_refused均为null。不是把46个缺信号槽记作46个模型预算负结果。RAM回归证明高chi+H3时adaptive原8候选需48>24被拒，fixed仍4个H1；这不预告真实child分布。

尚缺真实child CUDA signal adapter（旧NativeSignal只准根）、完整数组journal/独立child consumer、progress/reconfiguration compiler、独立回放及prefix replay、实时物理桥/真实闭环后果、fresh cohort与P1/适用P2/P3前置。全部scientific/compute/walltime/freshness/physical_release门False；只known-root工程校准。API/sample/J/VJP及inclusive wall各自单位，FLOPs/CUDA ns/backward engine未知不填0。RAM测试不是研究证据。
