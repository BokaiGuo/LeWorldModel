# 真正 RGB cp5 反馈后缀重优化：数值核心与待绑定接口

源码：`experiments/spectral_reachability/agent_nonlinear_suffix_observed_recompile_20261028_v1.py`；测试：`tests/test_agent_nonlinear_suffix_observed_recompile_20261028_v1.py`。本阶段按 MAIN 指令先交设计/数值 unit core，不猜未冻结的 Bohr 外部 schema；没有生产 prepare/run，不伪称已完成主机接线。

## ABI 当前事实与阻碍

本 worker 两次在源码/tests/innerdocs中查找 `agent_nonlinear_suffix_midpoint_rgb_20261028_v1`，交付时尚无文件；工具清单无可用跨线程消息接口，未能直接联系 Bohr。需 MAIN/Bohr 提供最终冻结 source/tests/doc 及实际 input/report/strict receipt path/SHA。不存在的 future SHA 不填占位假值，不把数值fixture当实际终态或证书。

本模块 `core_request` 仅是自己的内部数值接口，不宣称等同 Bohr ABI。唯一CLI `contract` 输出 blocked 的接口状态；不提供假 prepare/run/consume 绿灯。`production_ABI_bound/production_ready/production_run_authorized` 均false。

## 已实现干预（不是仅manifest）

复用冻结 `agent_p1_nonlinear_suffix_library_20261028_v2` 的 chart、cost、proposal/core validators，不修改旧v2（source SHA `0a3e475af9271a0769a3829b4e714fa4e2d6f00eaed38ec8c3ea5d7b9a163fbc`）。`optimize` 实际执行数值更新/全候选评价循环，step/gradient回调与冻结H1 Backend接口一致，但本worker仅传入 NumPy doubles。

固定原known2×off/on四个已选plan，每plan的prefix5已经真实执行，后续绝不重新选择/重编prefix。不重用其 imagined mid 作为 observation，而由未来adapter传入本臂真实cp5三帧encoder context及原生实际raw10。`request/validate_request` 验证 `raw10_cp5 == concat(root_raw10, selected_prefix_raw5)[-10:]` 逐位相等，原mean/std float32一次normalization对应两个action tokens，原goal192/identityQ不变。

`selected_incumbent` 核原完整9iterate/8梯度数值递推、stable原parent selection及prefix/suffix/logits位值；它不是原parent strict native consume的替代。原选中suffix可能是ridge iterate0，也可能是原新投影iterate：两种合法域分栏。ridge0按原float64 solver witness+cast/tanh rawbits合法，postcast norm略超cap不拒绝/不修补、不加epsilon；原投影iterate严格按原完整递推及float32 inward feasibility。禁止inverse-tanh或把suffix重新初始化为zero。

当前原后缀作为 observed iterate0，在**同一个真实cp5 context**上重新评分，不能挂旧imagined Q；然后固定8次scalar动作梯度/eta1，解析 `(1-raw²)` 链一次，原projection生成新iterate，保存原incumbent与全部8新iterate的预测context与terminal Q。每一步梯度AD-forward必须与同点评分context完全相同；stable `(actual nonlinear modelQ,iterate index)` best9，tie保留incumbent。输出full10 = 原prefix5+新suffix5，原parent候选/结果digest与新actual-context/goal digest独立记录。旧Q不当新预测，物理outcomes不进入selector。

`verify_numeric` 独立复算全部8步、9个Q、stable winner、完整数值event序列，拒绝删候选/缺梯度、事件移位、unknown字段、raw/history/goal/Q污染。`panel` 固定全4分母；失败保留已用数值events、failed row及未到达rows，不以替补根或旧动作填充。

## 费用与诚实边界

本核心固定每槽9 score-forward+8 AD-forward=17predict、8 scalar reverse；全4共68predict/32reverse。不是只有9次总forward，AD的8次必须单列。prefix已执行不再预测；新增J/SVD/视觉encoder/projector/physics=0；cp5视觉采集编码、原library编译与实际prefix执行均是继承费用，待Bohr真实receipt绑定，当前null。此数值核心不新增独立winner replay，若生产最终要求独立replay必须事前另列新增预算，不能藏进68。callback schedule是计划/数值调度，不是实际GPU账；`actual_native_resources/inherited_compute/physical_metrics`全部null。

冻结native Backend可用真实实例级observer+ledger接入，不能monkeypatch globals；生产adapter需记录真实scalar grad/VJP/raw callback和全native forward/modules/cp5 tensor IO，不能拿此RAM日志当GPU receipt。scalar runtime loss必须0维，codec[1]只是传输；rawleaf[5,2]严守。保持CUDA-only、冻结模型、TF32/autocast off、MATH及原normalizer。无训练、调参、改Q、挑根或科学准入。只输出数值best9，不声称真实response修复、任务收益或P1通过。

## Bohr 冻结后最小接线清单

不猜任何外部键名；以下是必须由实际schema逐项匹配的语义，而非假定该schema已有字段：

1. 调用Bohr最终真实strict consumer并绑定report/receipt及其全部sources/models/input SHA，证明4原selected plan顺序与parent strict、prefix/full10 digest/rawbits/goal/normalizer完全相同。
2. 每slot取得真实cp5三帧PNG bytes/pixel/实际encoder输出及action-history对应关系、未改root/goal/model的绑定；cp5 state不得传入compiler。原生warmup/executedraw与context动作tokens严格对应。
3. 原完整library strict读取selected prefix的parent candidate/result，`selected_incumbent`还原原suffix logits而非inverse；核与Bohr固定后5预测literal动作完全一致，原incumbent observed Q独立重算/核对。
4. prepare绑定own三件套、原冻结库和Bohr全部闭包以及精确68/32预算；生产run接冻结H1 Backend.step/gradient，但阶段label与资源auditor须按真实schema显式适配，不能过滤unknown或把numeric事件当native事件。
5. strict消费按每slot真实durable intent/input/return、原native tensor IO、9评分/8反向和全账对齐，失败保四分母；新full10 plan seal含parent与new observed-context/new suffixQ/newwitness，禁止新动作挂旧Q。
6. 最终生产前source freeze；由MAIN审读主机CUDA执行，真实physics/终态评价另行授权。本次未预造未来report/strict SHA、未授权physics。

若Bohr最后接口与以上语义不能闭合，则prepare/run应failclosed，并精确列缺字段/资产；不靠RAM或metadata认证真实RGB。

## 当前可执行入口与测试

在 `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_agent_nonlinear_suffix_observed_recompile_20261028_v1.py
.venv/bin/python -m experiments.spectral_reachability.agent_nonlinear_suffix_observed_recompile_20261028_v1 contract
```

30项轻量回归覆盖原parent真实格式数值fixture、完整四槽执行顺序/预算、合法overcast iterate0不改变、stable ties、mutation/firewall、失败分母。fixture是NumPy代数替身，不是神经/物理。未运行生产prepare/CUDA/physics/network。当前明确缺项：Bohr冻结ABI与其真实strict authority、production adapter/完整native账接线；待实际ABI交付后继续独占实现，不修改任何旧封存文件。
