# Fixed40 输入历史与反转定位（saved-only）

本独占实现绑定 rank-error 报告 `e23e60588fa9766b690e605022e089656c4c80ad185853773e23a942b5e0228f`。只读已有 known2 root 的 compiler/RGB40/physical 证据；不打开 reserved4fresh、不运行神经/物理/网络，不修改旧文件。旧个人 goal 为 blocked，本文件只记录本有界交付，不覆盖或完成父研究目标。

## 实际连接与完整分母

入口复用 frozen `rank.build()`，重新调用原 native saved validators、重算39实际与40预测 native192 Q/误差，再独立 `rank.assemble()`重算固定80组内对、20同word Z/S对。它不是重复日志hash检查。新层逐槽核验：

- 原task export/anchor/plan/physical branch/task_id/blind/condition/goal完全同源，原identity192 Q与mean/std不变；原goal state7与默认outline pose分别报告，outline不冒充taskgoal。
- warmup raw10固定前5个`[.1,0]`、后5个`[0,.1]`；teacher rawhistory等于warm6–10+自身prefix11–15，token使用原float32归一化。原root latent0/5/10滚动为真实z5/z10/z15，不使用预测endpoint代替观测。
- 实际physical journal每槽全部20个`raw_return`逐项校验raw_bits==actual_bits、float32、phase及顺序；suffix journal1–5映射绝对16–20。signed zero逐bit保留。
- 物理保存warmup5/10、prefix5的同condition PNG/pixel SHA与原task/anchor对应。每个word的evaluate与replay共80次原full module AE/predictor/pred_proj数组接线逐项复核；预测输出context必须保留z10/z15，并使用该suffix的原token。
- 每槽保留prefix/suffix/rawhistory bits、明确绝对输入/输出索引、target binding、原数组三SHA、模块输入绑定digest、实际return事件SHA。顶层bindings覆盖原闭包、被加载数组/PNG/journal及独占三文件。

始终40行、8组×10对。index12 actual_Q保持null，涉及它的4组内对及1个ZS对unknown；其五成员完整rank误差均unknown，不能删掉缺项得到“完整”排序。零tie不是反转。12个反转及全部其余pair同时输出，分组只按预定root/condition/role，不择winner、不据反转选根。历史一致不等于确认动态误差原因，`causal_error_source=unknown_not_identified_by_input_alignment`；latent Q不是物理收益，known roots仍2不是40，P1=false。

## 下一轮输入协议预注册草案（未运行，不授权dispatch）

问题：完全相同z5/z10/z15、token6–20及冻结goal/Q下，官方rollout API与当前cached suffix API是否逐输出等价？保留原全部40槽，不仅12反转，不增删root/word、不新physics、不重建缺失on图、不拟合。每臂逐槽提交相同literal float32输入，保存完整三层模块输入/输出、归一化前后token和API调用trace；两臂均40个suffix预测槽，禁止共享某臂结果伪装另一臂执行。物理39端点仅用于固定描述性Q与原80/20匹配统计，缺12照旧null。

此处**没有实现或运行**这两个API实验runner；MAIN需先固定官方API adapter与缓存策略/模型closure，再批准CUDA。两臂各40个语义suffix预测；真实API内部预测/encoder/AD/FLOPs/耗时费用当前unknown（null），不能称等FLOPs或把未来配额当实付。入口/API差异的数值容差若需使用必须MAIN事前声明；本consumer只做现有保存数组的精确字节与原Q重算。实验结果即使等价也不补齐P1/freshness，也不证明组合/控制成立。

## CLI与验证

从`/home/gbk/code/LeWorldModel/le-wm-main`执行，解释器不加载模型、不构造env：

```bash
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p1_teacher_library40_history_localization_20261028_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_p1_teacher_library40_history_localization_20261028_v1 analyze
.venv/bin/python -B -m experiments.spectral_reachability.agent_p1_teacher_library40_history_localization_20261028_v1 analyze --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p1_teacher_library40_history_localization_20261028_v1_analysis_MAIN01.json
.venv/bin/python -B -m experiments.spectral_reachability.agent_p1_teacher_library40_history_localization_20261028_v1 verify --analysis /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p1_teacher_library40_history_localization_20261028_v1_analysis_MAIN01.json --analysis-sha <实际输出SHA>
```

输出不可覆盖，verify重建整个报告，篡改/缺失/未终态输入failclosed。单测只RAM，saved运行是既有数组接线证据而非新科学通过。后续修改必须新版本，不原地改变封存闭包。

## 本次实际工程验证

新消费者38个RAM用例；与rank-error、原teacher cross-library、RGB40 v2测试联合运行，实际 **234 passed in 20.02s**，无模型/物理执行。真实saved `analyze` 已通过40槽接线，八组反转数按root0/off/Z、root0/off/S、root0/on/Z、root0/on/S、root1/off/Z、root1/off/S、root1/on/Z、root1/on/S的固定顺序为`0,1,1,2,1,3,2,2`。这是既有数据的描述性定位，不是新独立样本或P1通过。保存版将使用本独占stem的`analysis_NASH01.json`，MAIN可另用上面的`analysis_MAIN01.json`命令独立重建，不覆盖任何已有产物。

精确未决项：index12条件on terminal RGB仍缺失；模型输入对齐并未识别12反转的因果来源；官方API与cached API的独立统一输入实验未实现/未执行；外部API内部成本未知；无新root/freshness认证。无任何physical release。
