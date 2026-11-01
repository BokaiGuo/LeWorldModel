# P3 完整承诺三策略：只读准备与可执行节点适配器

## 边界

本版本仅 prepare / verify / accounting，以及可供后续真实驱动调用的
`make_request / plan_node / Commitment / remaining_caps`。没有生产 run 命令、
没有新生产结果、没有 CUDA / 物理执行；RAM 回归不等于真实证据。
已见两个 known roots，不是独立确认。P1/P3/freshness/scientific admission 全 false。
此次使用 experiment-code 的事前固定协议、日志与回归结构，不进行训练或结果择优。

## 原输入与严格权威

固定工程输入为 fullcommit MAIN01 report
`6c507b95a1cc60605d580032c0d92acfcd302abfbd79ebf9fa3c0a43e630b9ec`，
strict `f7424d13342295e30fd432094913e2735b55f94155dcf3a171c313499d432721`，
原 prepare `80ac2306344fd1d63d68d5867ba733347d1c7b87343741ec1bb1e3f5afc4edfd`。
prepare 和 verify 都真正调用未改的 `live.consume`，完整重建原封存实验的数组、
RGB、原生 journal、计量与输入闭包，并逐字段对比原 strict；不是只接受 status。
因此只读准备可能较慢，但不会调用神经或物理。新 prepare 复制原 tasks/layers/
policy/modelbinding，而不把旧 outcome 喂给新策略。
新三件套和原 sources 全部哈希绑定；prepare 后禁止原地编辑。

## 固定协议与不匹配预算

固定 2 known roots × fixedH1 / fixedH3 / adaptive = 六个 episode，18 个节点槽。
每个 episode 重新构造、原 reset/placement、warm10，原共同 prefix5，控制目标20。
节点在实际5/10/15观察真实 RGB，并按自己实际 raw 历史滚动；不能借用别臂 history。
每个到达节点都原样支付 H1+H3 nominal/AD/SVD 信号，即使正在执行既有承诺。
终态20编码仅评估。原 goal、identity192 Q、冻结模型、normalizer、动作域和
native reward/success 合同均不得变化。

| 策略 | 空闲节点决定 | 候选 | 全部20完成时 predict/episode |
|---|---|---|---|
| fixedH1 | 固定H1 | 固定4 | 51 |
| fixedH3 | 固定H3 | 固定8 | 75 |
| adaptive | 原 chi=.5 / gain=.05 规则、含原unsupported拒绝 | 原natural4/8 | 未知，cap75 |

fixedH3 是此版本事先声明的新固定策略，不冒称旧 adaptive 一定有8候选。
完整 H3 在5处承诺15步；10/15只观察并支付信号，不更新动作。
原 adaptive 若后来选择的完整 H 不适合剩余长度，整节点拒绝并终止该臂；
绝不截断、强制H1、补零、重选或重试。native done 原样早停，故同 raw20
是相同目标长度，不是强迫所有臂真的执行20。未到达、拒绝、失败都留在6/18分母。

计账沿用原 `core.AXES / Ledger`：调用前记录 attempted，实际返回再 completed；
失败尝试不退款。每节点收到的是 `remaining_caps(mode, previous_results)` 的余量。
信号与程序成本分开；H,n程序部分为 H*(2n+1) predict，另每节点8 signal predict、
两次192方向J/384 reverse、两次signal SVD。每候选完整与分段 replay，stable(cost,index)
选后独立再完整 replay。unknown FLOPs / GPU时间 / 实际物理成本写 null。
预算明确 `notmatched`；同共同上限不是等 FLOPs，不能用padding伪匹配。
物理 warm目标60/control目标120只是意图，不是已执行计量。

## 可执行接线与未完成生产部分

节点适配器真的调用原 backend.signal、spectra、scheduler、bounded generator、
backend.replay 与原Q，不是函数别名或只生成manifest。
`factory(task, store)` 可使用未改的 HOST CUDA native backend；本文件不加载模型。
`permit_ram=True` 仅在显式 RAM_FIXTURE 时合法，不能当 CPU 神经 fallback。
`make_request` 绑定 base task/current observed context/raw10/PNG与编码哈希；
原验证器仅以显式 validation-only structural view 检查观测 ABI，
新请求/结果始终 own schema，不发旧receipt。

后续新 live driver 应复用原初始化、renderer、原生 metrics 与 precision/weight guards，
但必须有自己的六臂 roster/journal/consumer，不能 monkeypatch MODES/旧全局：

1. 对每根每策略重新初始化并证明 prefix5 的原 state/RGB/goal 一致。
2. 自己实际 RGB 三帧与实际 warm+returned raw 的末10形成节点请求。
3. 新 Ledger 使用该臂剩余 cap，durable store 接原 native module/AD/resource events。
4. 空闲时执行 `plan_node`；sealed 后创建 `Commitment(request,result)`。
5. `packet(boundary)` 只发完整承诺的下一段，不允许未ack就替换；原native每步返回后
   收集实际 raw，`acknowledge` exact float32bits。短段仅在原 terminated/truncated 时合法。
6. 活跃H3的 `active(boundary)` 传回下一节点，先付双信号，再继续已封存动作。
7. 后续自己的 strict consumer 须重建全部 RGB/raw滚动、原生逐操作计量、承诺 seal、
   action/state/goal、native done 和完整数组。不拿原四臂 strict 为新六臂背书。

`Commitment` 是包释放/实际返回动作门，不替代全 native/PNG 认证；它也不证明调用方
传入的执行返回是真的。这一证明属于尚未启用的新 live journal 和严格消费者。
MAIN审读及显式 HOST CUDA+physics 授权前，不启用生产。

## MAIN 可执行命令

在 `/home/gbk/code/LeWorldModel/le-wm-main`，使用项目环境：

```sh
.venv/bin/python -m pytest -q tests/test_agent_p3_commitment_controls_20261029_v1.py
.venv/bin/python -m experiments.spectral_reachability.agent_p3_commitment_controls_20261029_v1 accounting
.venv/bin/python -m experiments.spectral_reachability.agent_p3_commitment_controls_20261029_v1 prepare --output results/agent_p3_commitment_controls_20261029_v1_prepare_MAIN01.json
.venv/bin/python -m experiments.spectral_reachability.agent_p3_commitment_controls_20261029_v1 verify --prepared results/agent_p3_commitment_controls_20261029_v1_prepare_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA
```

worker 不运行生产prepare/verify、CUDA或物理。只运行轻量RAM回归与accounting。
先交只读prepare/计账及节点接线；完整六臂生产 driver/strict 与授权是明确未完成项。

实际回归：上述新测试命令 **23 passed in 5.34s**；accounting CLI exit0。
实际只读 SHA 核对三个原始输入全部符合以上固定值；尚未执行完整 saved consume 重建。
该重建会在 MAIN 的 prepare/verify 中强制进行，不能将这些 RAM 测试称为科学通过。
