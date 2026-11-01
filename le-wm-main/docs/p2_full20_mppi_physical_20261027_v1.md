# MPPI-inspired 四条封存见证的独立物理 consumer（v1）

只新增本模块/测试/本文及新prepare，不修改producer或旧consumer的白名单。
主线程已接受40完整H2=80planning+4replay，并自行运行CUDA；本worker仅读取
`p2_full20_mppi_20261027_v1_prepared01/report.json`，不执行GPU/physics。
使用experiment-code技能约束明确来源、完整失败持久化及回归测试；不运行其训练
或调参步骤。MPPI-inspired heuristic边界保持，不能冒称官方MPPI。

## 保存证据与动作绑定

prepare复核producer全部source seals、原full20上下文/权重/normalizer绑定，
按稳定root SHA join四根，不依赖报告数组顺序。每根仅MPPI一臂，全部40候选、
80planning+4replay、168 actual predict/action_encoder入口、252 context tokens，
PredictionBudget root/model/method/阶段绑定逐项核对，合计336实际predict samples。

按原seed、temperature=1、3轮×12，使用已保存40个cost重建原kernel的proposal
轨迹并逐个比对float32 raw action；这只是保存收据的非神经确定性重建，不调用
模型、不新评估候选或改选择。initial/updated nominal全部计入40；核对kernel
receipt、best-ever first tie、原goal0/Q分数、单flat edge与完整独立replay见证。
导出四条`[10,2]`计划，绑定receipt SHA、完整root/goal/context、action domain、
candidate ID及selection SHA。物理root RGB/history stack/goal身份也绑定原官方来源。

prepare不创建环境、不reset/step/render；只解析已注册官方环境class，封存class及
eval_state/reset/render/step源码。旧durableguard、action mapping、normalizer、
相关运行依赖、producer、新consumer/测试/本文均进入source seals。哈希是内容
绑定，不是实际执行或科研效能证明。

## 持久化与执行边界

真正`run`只供主线程另行执行。使用原durable逐步guard与physical_branch，每次
env.step前attempt先fsync，返回立即记账，图像/trace/branch先于任何终态encoding
持久化。原任意20维动作逐步发送，不压为constant2、不重新规划或调用predict。
RGB不可见、终止、reset/评估/编码失败均保留；未运行分支也写blocked账本。
固定十步与guard提前终点区别保存；仅完整H2进入可选终态编码，中止不替代终态。

默认physics-only不加载神经模型；显式`--encode-terminals`才在已保存完整物理报告
之后主机CUDA编码，predict入口禁止，参数必须frozen eval。没有CPU神经fallback。
四个selected path最多40物理步、4次终态编码，不是四个新独立根，也不是完整预算
对照。reset/warmup/root重建有额外模拟器步骤，selected-path计数不涵盖它们。
P1/P2、物理效能、总资源匹配均false，不输出统计提升或faithful MPPI结论。

## 本次只执行测试与prepare

cwd `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p2_full20_mppi_physical_20261027_v1.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p2_full20_mppi_physical_20261027_v1 prepare --output results/p2_full20_mppi_physical_20261027_v1_prepared01
```

输出为独占protocol/prepared/source seals，包含四条选择见证、原CUDA报告SHA、
源代码SHA；状态`physical_consumer_prepared_not_executed`。若原CUDA报告缺失、
账本失败、任一见证不完整或哈希不符，prepare直接拒绝，不补计划或运行任何物理。
本测试采用保存报告和纯Python fake Env，不运行真实模拟器或神经网络。

主线程review后自行运行，output也必须不存在（本worker未执行）：

```bash
.venv/bin/python -B -m experiments.spectral_reachability.p2_full20_mppi_physical_20261027_v1 run --prepared results/p2_full20_mppi_physical_20261027_v1_prepared01 --output results/p2_full20_mppi_physical_20261027_v1_run01 --encode-terminals
```

省略encode参数则不做任何神经编码。所有分支完整与完整性门通过才exit0；否则
exit非零并保留逐分支失败现场，禁止覆盖或自动重试。旧consumer和producer文件
及其sealed字节不变，producer的protocol SHA保持
`d61e9b50a438c959544bb7677aabf584d8e7a8bedce3814a104924634be9e919`。

本次实际验证：新consumer专属30项通过；含admission、MPPI producer、原sampling
kernel、full20 sampling、旧physical与witness/contract共8文件联合317 passed
（18.02秒）。仅有第三方pygame的pkg_resources弃用警告，无CUDA或真实physics作业。
