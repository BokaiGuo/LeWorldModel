# Actual-cp5 reach-secant：固定四组/十六槽生产接缝

实现仅供MAIN审读后HOST CUDA运行；worker只执行RAM/NumPy与保存文件fixture。未运行生产prepare、NN、physics或生产fullstrict。不是exact梯度、原R等价、等预算、model delivery、P1/P3/full-C3或物理收益证据。

## 文件和固定权威

- 数学核：`experiments/spectral_reachability/agent_p3_reach_secant_kernel_20261029_v1.py`。
- runner：`experiments/spectral_reachability/agent_p3_reach_secant_actual_cp5_cuda_20261029_v1.py`。
- RAM测试：`tests/test_agent_p3_reach_secant_actual_cp5_cuda_20261029_v1.py`。
- 本doc；四文件全部进入prepare source闭包。

批准协议`agent_p3_reach_secant_actual_cp5_protocol_20261029_v1.md` SHA `449873d6d60248cdeb34641842aa6a827eba4ab02dd0346e4ee570bc844cb37c`，不修改。
原goalbank prepare/report/receipt固定为 `d54438930818af9bfd65c0f23a0666a19b320d3a62fb05256d5ac05beb798771` / `9bedec46c5c0e74b73c145935280b2efe400f617f8bbe7b72205fedb60a20d4f` / `b50268a2018e6da1fbc46a4d19d701cd4a9a575cfaf581807cf29c01861d7ba8`；路径详见runner AUTH。
完整调用冻结producer.consume，与持久receipt整份canonical相等才提取。不是只验同名schema或摘要；MAIN的prepare/verify/consume都会重建该上游，因此saved CPU耗时不代表新NN。
原producer源码/tests/doc精确PINS、原classes/模型/完整source闭包、actual-cp5双task/context、goal/Q/normalizer均继承。actual历史abs5/10/15，不把control0冒认cp5。

## 数学和真实ABI

固定root0/H1、root0/H3、root1/H1、root1/H3。原零+八项全部保留origin pointer/完整行hash/动作hash，原大半径不抹账。
源small索引0..3严格d0−、d0+、d1−、d1+；radius=sqrt(5)/4。源bank directions恰为d0/d1，null原状态`blocked_degenerate_direction`，不是0评分。真实own-schema使用`full_C3_demonstrated=false`；嵌套numeric核的`C3=false`另存，不互代。
源local_task/native_response/mapping/source-capsule认证后，核逐位核params与tanh→f32 raw。解析器只接受已固定完整no_witness源cohort，不将缺字段/partial装成完整来源。

用二维中心割线sI/sR、非正交Gram G，一次付费SVD和solve(G,I2)，按批准闭式求最小范数共同方向。保存D/G/M、SVD、割线、κ/c、约束残差、实算方向范数。κ越界不clip；64eps仅数值可解析性判定，不是gate tolerance。
新四半径严格r0/2、/4、/8、/16；float64 tanh后一次cast f32、原normalizer不变。全部有效候选完整edge、真实postcontext H1 J、原bounded_ridge与真实suffix诊断；R取own_reach_value，不混作suffix非线性cost。
每项资格严格I>原paid-zero.I且R<原paid-zero.R，无epsilon。即使首项eligible仍跑满四项。无eligible/有eligible都不交付计划，delivery始终null、model_delivery_verified/physical_release等均false；没有whole/segmented或physics。

## 计费与失败

27轴attempted/completed/caps、嵌套parent/失败/时间逐项记录，unknown FLOPs/scalar backward保持null。
继承实账196predict/40J/72replay/40solve/40arraySVD/40programcompile/72metric；原AD-primal44、J元素84480另列。
新H1组16predict/4J，H3组24predict/4J；每组8replay/5solve/5arraySVD/5programcompile/8metric。全4上限80predict/16J/32replay/20solve/20arraySVD/20programcompile/32metric；新AD-primal16、J元素30720。累计上限276predict/56J，不称本次重新执行继承阶段，也不签FLOPs。
整组surrogate+四完整item派发前原子预检。源方向null不派发surrogate；秩/zero/cone退化仅记实际已派发数学费用，四槽显式null，不padding。异常保留failed付费前缀，停止后续组；完整四组/十六槽分母保留。outer failure保存但不签strict成功。
2GiB新output含512MiB finalreserve，diskfloor200MiB，总wall3600s/组1800s，CUDA allocated3GiB/reserved4GiB。源字节只引用不复制。失败不删文件、不重跑退款。

## saved strict

自有schema为NAME.prepare/report/group/consume。成功报告`four_groups_settled_NOT_delivery`，receipt `all4_actual_cp5_secant_groups_verified_NOT_delivery`，兼容明确退化四null。
严格重建surrogate/全部native IO/事件及call census/原checked_path/metrics/ridge/资格/费用；拒绝unknown、orphan、漏/重复事件。完整文件census仅binding.json、events.jsonl、report.json，逐字节SHA/hashchain/source前后/模型class精度闭包。RAM fixture不能获生产receipt。

## MAIN CLI（worker未执行）

项目根`/home/gbk/code/LeWorldModel/le-wm-main`，模块如下；P/R替换为实际输出SHA，不能预填未来值。路径必须新独占、不存在。

```bash
.venv/bin/python -m pytest -q tests/test_agent_p3_reach_secant_actual_cp5_cuda_20261029_v1.py
M=experiments.spectral_reachability.agent_p3_reach_secant_actual_cp5_cuda_20261029_v1
.venv/bin/python -m "$M" prepare --output results/agent_p3_reach_secant_actual_cp5_cuda_20261029_v1_prepare_MAIN01.json
.venv/bin/python -m "$M" verify --prepared results/agent_p3_reach_secant_actual_cp5_cuda_20261029_v1_prepare_MAIN01.json --prepare-sha "$P"
.venv/bin/python -m "$M" run --prepared results/agent_p3_reach_secant_actual_cp5_cuda_20261029_v1_prepare_MAIN01.json --prepare-sha "$P" --output results/agent_p3_reach_secant_actual_cp5_cuda_20261029_v1_cuda_MAIN01 --authorization MAIN_APPROVED_HOST_CUDA
.venv/bin/python -m "$M" consume --prepared results/agent_p3_reach_secant_actual_cp5_cuda_20261029_v1_prepare_MAIN01.json --prepare-sha "$P" --output results/agent_p3_reach_secant_actual_cp5_cuda_20261029_v1_cuda_MAIN01 --report-sha "$R" --receipt results/agent_p3_reach_secant_actual_cp5_cuda_20261029_v1_consume_MAIN01.json
```

测试涵盖非正交独立active-set RAM对照、H1/H3、源索引/raw/null/status/全receipt/local配对、全部16槽与完整saved重建、退化和失败不padding、原生数组/数学/费用/事件篡改、文件SHA/census/hashchain拒绝、cold import/CLI和worker授权拒绝。独立QP只用于RAM数学测试，不是独立模型实现；不把fixture通过等同真实生产结果。
API设计技能用于分开纯数学核、上游整份authority认证与执行/消费边界；未引入新科学gate或授权。剩余工作仅MAIN审读、实际prepare/HOST CUDA/strict及结果解释；本版不提供后代/物理交付。
