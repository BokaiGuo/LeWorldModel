# P3 complete-commitment controls v2：六臂 live dispatcher 与 saved strict

## 范围与权威

仅新增本 v2 source/tests/innerdoc；v1、原 fullcommit 模块和全部 MAIN 结果保持不变。
本版本由 MAIN 审读、授权后才能实际 HOST CUDA + native physics 执行。worker 只做
RAM 转换/数组回归，不运行生产prepare、模型、物理、训练、网络或 CPU 神经推理。
使用 experiment-code 的固定实验/日志/回归结构，不采用结果驱动择优调参。

工程输入沿用冻结 v1 的完整权威链：

- 原 fullcommit report `6c507b95a1cc60605d580032c0d92acfcd302abfbd79ebf9fa3c0a43e630b9ec`。
- 原 strict `f7424d13342295e30fd432094913e2735b55f94155dcf3a171c313499d432721`。
- 原 prepare `80ac2306344fd1d63d68d5867ba733347d1c7b87343741ec1bb1e3f5afc4edfd`。

`build()` 调用冻结 `controls.build()`，因此实际经 `legacy.consume` 全面重建原
report/strict；不是仅查 status。v2 增加自己的三件套和全部工程输入/source SHA，
得到自己的 prepare。verify 每次完整重建，不冒充新六臂已经通过旧四臂消费者。
本次没有新生产 report/receipt，也不预填未来 SHA。

## 预固定实验

2 个已见 known roots × fixedH1/fixedH3/adaptive，固定 6 episode / 18 node slots。
不是六个独立根，也不是 fresh confirmation；P1/P3/scientific/freshness 均 false。
所有失败、未到达、拒绝与早停均保留。相同 raw20 是完整控制长度目标，不是强迫
env 在原生 terminated/truncated 后再执行，也不通过补零或截断伪造20。

每臂独立 constructor → reset(seed=原reset_seed) → 原 sample_state placement recipe →
reset(seed=0,state=requested,goal=原生goal) → 原 raw warm10 → 绑定原 task goal。
**不恢复 prefix poststate，不共享 env，不借用上一臂执行历史。**
每臂构造前新增 `initialization_intent`，持久化 reset/root seeds、recipe SHA、原
warm raw SHA、完整 reference 身份；随后每个真实 native operation 都有原 metered
attempt/return。每个已返回 warm/control 状态、snapshot、goal/goal_pose、RGB 原图
与 guard 均保存。warmup 的原生 goal 在后续 bind 时才切换成 task goal；不得将
warmup 旧goal的 reward 当作当前 taskgoal 度量。控制段 reward/earlystop 严格沿用
原 native7 reward 和 jointXY+angle success，不以 object-only 指标替代。

先执行原固定 prefix5，再在实际5/10/15处编码自己的 RGB，使用自己的原两帧+
新实际 latent 滚动，action history 为原warm10 + 自己 returned raw 的末10，原样
归一化一次。固定策略为 H1/4 candidates、H3/8 candidates；adaptive 仍为原
chi=.5/gain=.05 调度与原 unsupported 拒绝。固定 H3 不是旧 adaptive 的改名。

每个到达节点均调用**冻结 v1 `controls.plan_node`**。同时支付原 H1/H3 nominal/J/
384 reverse/SVD 信号；每个候选 whole/segmented replay，并完整 terminalQ stable
(cost,index) 选择，再独立 whole replay。无 env/state/native metric 传入 compiler。
H3在5处完整承诺15步；在10/15处付费观察但不替换其余动作。各 packet 只能取该
封存完整 buffer 的对应5步。adaptive 选出的完整 H 无法放入20剩余长度时整节点
拒绝、结束该臂；不强制H1、不截断、不重选、不退款、不重试。

## 双 ledger 和资源上限

物理/视觉为自己的 Store/Meter/journal，模型为每节点自己的 NativeJournal + 原
core.Ledger，native backend 保持 `NativeProgram`。均在实际 entry 前计 attempted，
返回后计 completed；失败保留 pending/attempted。不是用RAM receipt冒充执行。
全输出立即 flush/fsync，closing/final-report 保留空间不受普通阶段 wall check阻断。

| 上限 | 预声明值 |
|---|---|
| predict/episode | fixedH1 51，fixedH3 75，adaptive 75 |
| predict 全六臂上限 | 402（含各到达节点 nominal+AD forward；不含视觉encoder） |
| J / VJP方向全六臂上限 | 36 / 6912 |
| 视觉 encoder/projector | 各24 |
| constructor/reset/set_state | 6 / 18 / 36（reset预留上限，不是声称实际调用18） |
| public / physics steps | 180 / 1836 |
| explicit render / close | 192 / 6 |
| wall / output / final reserve / disk floor | 2700s / 768MiB / 96MiB / 200MiB |
| CUDA allocated / reserved | 3GiB / 4GiB |

模型轴沿用 v1 保守 arm caps，不把相同上限说成相同实际工作。全成功完整20的
native实际应为 reset12/set_state24/public180/physics1824/render192/encoder24。
不同策略、拒绝与 native stop 使实际成本不同；明确 `budget_match=notmatched`，
equal_FLOPs/equal_wall=false，未知 FLOPs/CUDA_ns/backward-engine calls=null。
分臂保存 signal/program/total attempted/completed 与 inclusive wall，不对嵌套wall求和。
run 的 wall 从 verify 前开始计时；prepare 单独命令不计入 run wall。

## 独立 strict 验收

v2 自己的 `consume` 验证自己的 schema/6臂18槽、prepared SHA 和 execution_binding，
重建完整 physical hashchain、已知事件 exact fields、逐操作嵌套 lifecycle、每臂
每region每资源计量、独立初始化意图及全部原warm关键帧。原生拒绝/早停可为
完整工程记录，但不冒充 completed20；partial 保留且 complete=false。

对 clean 工程记录进一步逐项重建：

- 原始 PNG 与 alias **原bytes**、图像像素及 request 的绑定；复用无策略依赖的
  `harden.audit_rgb` 重算所有预处理 float32 bits，读取完整 native192数组。
- 自己每 boundary 的三帧 latent / last10 raw，原 task/goal/Q/normalizer 和 observed
  编码 event顺序；模型规划不可修改原生 state/goal。
- 调用自己的 `audit_plan`；复用策略无关的原 `audit_resources/audit_trajectory`，
  重建原生 module 输入/输出、normalization、全部192方向J/raw callbacks、每个
  候选 whole/segmented path、全部Q/stable winner及独立selected replay。
- 请求/结果/native journal/dispatch seal 及每步full-commit offset，完整 action/goal/
  native7 reward/done、末state/metric，所有PNG/native计量；三策略相同根prefix身份。
- 模型参数与全部 buffers before/after、源闭包、TF32/autocast off、MATH SDP和资源峰值。

复用的低层函数仅消费当前 v2 数组，不调用旧四臂 consume 来给六臂盖章；没有
monkeypatch旧globals、动态exec、strategy relabel或伪造旧receipt。saved检查不重新
做神经推理，也不能证明模型预测正确。生产尚未运行，所有新结果未知。

## MAIN 命令

工作目录 `/home/gbk/code/LeWorldModel/le-wm-main`：

```sh
.venv/bin/python -m pytest -q tests/test_agent_p3_commitment_controls_20261029_v2.py
.venv/bin/python -m experiments.spectral_reachability.agent_p3_commitment_controls_20261029_v2 prepare --output results/agent_p3_commitment_controls_20261029_v2_prepare_MAIN01.json
.venv/bin/python -m experiments.spectral_reachability.agent_p3_commitment_controls_20261029_v2 verify --prepared results/agent_p3_commitment_controls_20261029_v2_prepare_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA
.venv/bin/python -m experiments.spectral_reachability.agent_p3_commitment_controls_20261029_v2 run --prepared results/agent_p3_commitment_controls_20261029_v2_prepare_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA --output results/agent_p3_commitment_controls_20261029_v2_MAIN01 --authorize-known-cuda-and-physics
.venv/bin/python -m experiments.spectral_reachability.agent_p3_commitment_controls_20261029_v2 consume --prepared results/agent_p3_commitment_controls_20261029_v2_prepare_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA --output results/agent_p3_commitment_controls_20261029_v2_MAIN01 --report-sha ACTUAL_REPORT_SHA --receipt results/agent_p3_commitment_controls_20261029_v2_consume_MAIN01.json
```

receipt 独占新路径，输出已有路径拒绝覆盖；不要在prepare后改动三件套。
MAIN已独立审读源码；生产prepare/CUDA+物理/strict尚未执行，不能将本地RAM测试等同真实工程通过或科学准入。

## Writer ABI 与终态回归

实际文件系统测试调用本版 Store 继承的 `moving.Store.emit/put` 和本版
`NativeJournal.event/close`，不是 RAM codec 代替。分别写两个 physical events 和
两个 native events，验证原bytes、独立reader、首尾/previous/hashchain。
`old.raw` 已有末尾LF，继承Store的emit/put不再追加；`core.canonical` 无LF，
NativeJournal明确追加一次LF。两种格式实测均为严格单LF，不过滤空行。
native双LF和保持外层格式但篡改payload都会被真实reader拒绝。

另加 clean terminal disposition 硬检查：short-control 若真实返回没有 native done/
truncated，不得仅靠改 `status` 变成合法早停；completed20必须实际20且初始化完整，
非原生短结束必须有明确整节点拒绝。clean report 必须记录 notmatched/非等FLOPs。

首轮 v2 RAM/saved 回归30 passed（85.67s）；真实writer与终态新增定向回归
4 passed（2.69s）。最终冻结前 v1+v2 联合回归 **57 passed in 88.19s**，命令为
`.venv/bin/python -m pytest -q tests/test_agent_p3_commitment_controls_20261029_v1.py tests/test_agent_p3_commitment_controls_20261029_v2.py`。
原v1三件套SHA已复核未变。上述回归无生产NN/physics。
