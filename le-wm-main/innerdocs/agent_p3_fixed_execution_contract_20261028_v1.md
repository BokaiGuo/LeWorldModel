# P3 固定真实请求执行长度：独占事前合同与准备接口

新stem source/tests/innerdoc，不改冻结common-budget/native/signal/runtime/结果，不汇总MAIN新report结果。仅RAM/saved metadata检查；无CUDA、神经CPU、physics、网络、训练/拟合、阈值或根筛选、后代agent。遵循experiment-code的事前设计/日志/回归原则；用户明确限制优先，不做训练、择优或研究图。

## 新比较到底固定什么

真实**请求执行长度固定5步（一个原始动作包）**；规划lookahead仍允许H1/H3（5/15个优化动作），两者不是同一变量。H3是同一个root上联合优化的30个raw坐标，不是三个有新观测/J的child；实际child扩展深度0、实际执行extension数1。原requested_tree_depth=3只保留为上游策略字段，本任务不执行完整树。

选中完整计划后，动作consumer只可接收 **该实际计划首5个原float32动作**。H3后10步是前瞻尾段，明确不执行；H1绝不重复/补零到15步。不是原common-budget整段计划被偷偷截短后伪称旧试验已经同长度，而是**新的、在任何本轮执行前固定的first-packet执行规则**。新的实体实验和结果仍待MAIN单独安排。

规划选择继续用原H自己的终态modelQ，H1/H3决策规则、damping=.002、normalizer、Q和生成器均不改。这些分数跨不同规划终点不能当共同执行终态结果。真正对比必须另在共同物理step5用原native任务指标评估；原PushT成功包含agent+object，不能用object-only投影或latentQ冒充成功/失败。请求中另保存首包预测modelQ，仅诊断，不替代物理。

contract/结果明确 `equal_requested_execution_horizon=true`，但 **`equal_actual_physical_execution_horizon=null`、`actual_physics_status=unknown_not_run`**。合同给出相同长度，不等于环境实际上已执行相同长度。native早停/错误不得补步；完整分母保留，实际长度在未来真实receipt中记录。不能把提前终止当执行满5步，也不能因未满长度删任务或换动作。

## 无padding的可行预算

复用sealed common-budget的真实请求/生成器/native replay，但这不是读取/复用其modelQ结果。prepare只验证其事前prepare及两份原独立signal回执，不读取本轮common-budget CUDA report的winner/score。已有signal是相同root上的已付费模型可见输入，按完整成本列账；不是本轮新执行的AD，也绝不借作未来child signal。若需要本轮独立重新采集，MAIN可用新signal回执构建另一个base prepare。

每task×method：完整独立H1/H3信号8predict（nominal+AD各H）；搜索24predict（H1十二候选×两遍×H1，H3四候选×两遍×H3）；选出赢家后首包另独立replay **1predict**，验证实际待执行prefix的endpoint与完整计划首边界精确一致。该调用验证了真实要执行的动作，失败即停止，不是无用途padding。共 **33 predict API/33 samples**。不必补齐FLOPs、wall或其他轴，不声称总算力相等。

本轮若仍为两根fixedH1/adaptiveH3，全部四分支总模型账132predict/132samples：两份signal已付32（8J/1536VJP总账），新search+prefix100。新AE/predictor各100calls/100samples/300tokens；pred_proj100calls/300rows；4solve/4arraySVD、4program_compile、68replay（候选64+首包4）、36metric（候选32+首包4）。每branch新25predict，加signal8=33。signal每branch2J/384VJP及对应grad/raw callbacks完整保留；搜索无新AD。unknown backward-engine/CUDA ns/FLOPs/物理步数均null，不填0。

候选均实际生成、回放且有资格获选，不早筛、不去重、不保证一定分离；原负/零结果保持。固定H1也完整付H3信号，是受控dual-signal pipeline消融，**不是最省成本固定H1强基线**。比的是不同root联合规划lookahead/搜索候选配额的控制策略，固定最终请求5步；不是完整Proposal C3树深度效果。

## 可调用接口与明确停点

- `material(common_prepared, sha)` / CLI prepare：验证已有事前base prepare、source closure和真实两份独立signal，绑定自身三文件、所有旧依赖与输入、固定执行合同、四分支队列与新增预算，不读旧comparison结果。
- `queue(base)`：原task/model/goal/Q/root identity精确绑定，给真实search请求和“赢家首包独立验证”后置操作；无模型时commands/model结果/物理结果为null。
- `execute_panel(prepared, factory, ledger, store, rows)`：真实可调用接线，直接复用common.panel实际搜索与`NativeProgram.replay`，获选后再调用同native adapter回放5步prefix；实际prefix动作/endpoint精确闭合才返回blocked execution_request。MAIN必须提供已加载冻结HOST CUDA的NativeProgram factory、真实durable Store/Ledger和完整source/model前后校验。默认拒绝RAM；测试必须显式permit_ram。该函数不自行加载模型，不隐式授权CUDA。
- `execution_request(task, winner, path)`：校验独立回放的实际首包路径、task+model+root身份（即便不同task有相同latent也不可混），只返回真实首包，不生成剩余动作，不放行物理。
- `audit_budget`：按actual ledger逐轴复算每分支33API/33samples，保留全部J成本，拒绝部分、虚记/多余prefix调用。
- `audit_physical_receipt`：仅未来物理consumer的**工程metadata边界检查器**，要求原请求SHA、实际返回动作前缀、native7维状态、native_success bool或null、stop/error真实处置；它不验证journal/图像/物理资产真实性，也不证实实际实验发生。返回`artifact_authenticity_verified=false`、physical_release=false，不能用RAM receipt放行物理。

**本交付止于合同/准备和可调用panel**。尚无新CLI CUDA runner（source/model/hashchain前后收尾与工程cap执行需MAIN审查的新runner承接），尚无该新search+prefix产物的strict native-buffer saved consumer，也无真正固定5步物理consumer、recipe/root/goal7维身份/RGB/warmup/计量闭包。旧物理consumer是10步/8计划，不可直接冒充5步版本。execution_request的physical_root_binding/physical_goal_binding=null；P1/freshness未准入，全部release/admission false。不存在把缺失adapter伪装成可运行physical的run命令。

随后要做第二次闭环动作，需要实际新观测history+准确state identity、该child新付H1/H3 J；这些adapter不在本任务中，无root信号复用通路。

## MAIN命令

工作目录 `/home/gbk/code/LeWorldModel/le-wm-main`。prepare仅saved metadata/NumPy；不会加载模型。第一条不需要任何实验结果：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p3_fixed_execution_contract_20261028_v1 plan
sha256sum results/agent_p3_common_budget_20261028_v1_prepare_MAIN01.json
```

本工作位实际验证该base prepare SHA为 `c49dff7a7fe25de4f967c6acb9052091436702a7bddc54ffa87d85b7e9dd40d7`；不能用MAIN报告SHA代替。随后：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p3_fixed_execution_contract_20261028_v1 prepare --common-prepared results/agent_p3_common_budget_20261028_v1_prepare_MAIN01.json --common-sha256 c49dff7a7fe25de4f967c6acb9052091436702a7bddc54ffa87d85b7e9dd40d7 --output results/agent_p3_fixed_execution_contract_20261028_v1_prepare_MAIN01.json
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p3_fixed_execution_contract_20261028_v1 verify --prepared results/agent_p3_fixed_execution_contract_20261028_v1_prepare_MAIN01.json --sha256 "$PREPARE_SHA"
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p3_fixed_execution_contract_20261028_v1 queue --prepared results/agent_p3_fixed_execution_contract_20261028_v1_prepare_MAIN01.json --sha256 "$PREPARE_SHA"
```

PREPARE_SHA只取prepare实际输出；不得覆盖已有结果。本任务不会自动启动后续模型或物理作业。回归命令：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/agent_p3_fixed_execution_contract_20261028_v1_test.py tests/agent_p3_common_budget_20261028_v1_test.py tests/agent_p3_native_program_20261028_v1_test.py tests/agent_p3_local_signal_cuda_20261028_v1_test.py tests/agent_p3_runtime_contract_20261028_v1_test.py
```

RAM fixtures沿用已有纯NumPy动作/状态算术，不是模拟真实模型科学结果；测试冷导入另用独立Python进程，不删除sys.modules。旧sealed测试保持原样，联合回归从新进程启动。任何contract测试通过不等于P1/P3/完整C3或真实物理收益。

## 本工作位验证记录

最终代码五文件联合210 passed /27.95s；已import torch但未构造/执行模型的进程中，新测试36 passed /3.00s。RAM覆盖真实search→赢家首包独立验证的同一panel接线、每branch33API/33samples及完整J成本、同latent不同task身份拒绝、native早停/工程失败完整分母、不padding/不补步、7维state/null native结果、预算/child/缺失请求/动作/端点变异拒绝、prepare/verify闭包与冷导入。

实际读取并验证上述真实saved base prepare及其信号/source闭包，得到四分支规划H `[1,3,1,3]`、请求执行步数均5、新predict cap100；只在RAM中构造合同，不写研究资产/结果/prepare。无神经/physics调用，未读取新common-budget report的得分或赢家。三独占文件whitespace diff-check无警告。模型/物理结果仍unknown；可执行准备接口完成不表示未来runner、strict consumer或物理adapter已经完成。
