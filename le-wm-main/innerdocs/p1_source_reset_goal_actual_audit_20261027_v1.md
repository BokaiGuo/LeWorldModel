# P1 calibration2：实际 reset / goal 保存证据独立复算

## 结论与边界

本次主线程实际诊断完成；本审查仅读取保存的 JSON / hash-chain journal / NPZ，
用 NumPy 复算，没有运行模拟、模型、CUDA或新的物理步骤，没有改动任何封存输入。

本批全部8个attempt、两类reset共16次goal render中，实际goal state与请求goal的
位置、agent velocity、wrapped angle差均为 **0**。所有goal capture的block线速度与
角速度均为 **0**。因此，先前静态审查指出的“目标render发生几何漂移或携带hidden
velocity”风险，在这批已观察调用中没有出现，不能据此解释两条selected source的
模型负结果。10/16次raw angle差为+2π，wrapped差为0，仅为等价角表示，不能算几何错位。

一个未选attempt确有第二次setter的实际placement修正；producer保存的是修正后的
实际state，且该state与当前render、历史记录精确对应。这是“请求state不能直接
冒充实际state”的实证，但不是本批selected source错绑或已证假负面根因。

范围仅为两个calibration seed及这8个固定attempt，不是所有reset输入、历史数据、
模型感知/时序/动作合同的普遍排除。P1_admission仍为false。

## 冻结证据绑定

下列路径均相对 `/home/gbk/code/LeWorldModel/le-wm-main`：

| 输入/源码 | SHA256 |
| --- | --- |
| `results/p1_source_reset_goal_audit_20261027_v1_main01/report.json` | `2f6b7c4ad8b54a3157f1ac2f514698b96a5cce1550a6c81a4f57d206df7ebe24` |
| 同目录 `journal.jsonl` | `2091b7dd70ed8c1e703ed528148bf66a65d513754ecbff4adddc6e3de228fd25` |
| `results/p1_source_reset_goal_audit_20261027_v1_prepare01.json` | `743bb32f5ceaf7081c3f267f3e428a10804593930163437804e0c74489c2e2a0` |
| `results/p1_conditional_calibration2_main_20261027_v1_request.json` | `a53a6671e67eb6a79d0ccb6614bf6d5dfb4b6417ec3eb3b548ab63a41fb67d68` |
| `results/p1_conditional_calibration2_main_20261027_v1/sourcepack.json` | `728f14cb83c4fcd0ea4f5348bad2bff5b5dd372ab0185b6e6b3a26dbfda58057` |
| `experiments/spectral_reachability/p1_source_reset_goal_audit_20261027_v1.py` | `983f39d71d9090f15167a70fba16b741de14eef1fc89dd982bf47be7130f2a0b` |
| `experiments/spectral_reachability/p1_conditional_procedural_source_20261027_v1.py` | `faddd18ae8d7e0a97c50348929cb4c860321c88a1aedfb755eedc8a0f717deb3` |

重新调用诊断模块的纯读取 `material()` 验证原request digests、sealed source closure、
sourcepack原journal链、全24条原始operation receipts、两个selected NPZ SHA/root
及完整8attempt census；不刷新历史exclusions、不重选。独立验证诊断journal SHA及
整条hash链；8条 `attempt_finished.receipt` 与report.rows逐条相同。

## 实际分母、调用和复算结果

报告 `status=diagnostic_completed`、`all_reproduction_checks=true`，
`first_error=null`、`secondary_errors=[]`。8/8完成，未遗漏未选attempt。

| 资源 | attempted / completed | 保存journal独立核对 |
| --- | --- | --- |
| public reset | 16 / 16 | 16 reset_started、16 reset_returned |
| explicit render | 8 / 8 | 8 explicit_render_started |
| 内部goal render | 16 / 16 | 16 goal_render_started |
| `_set_state` | 32 / 32 | 32 set_state_before、32 set_state_after |
| 所有render | 24 / 24 | 24 render_before、24 render_after |

这不是env.step训练或控制实验：public steps=0，neural calls=0；每次setter内部
physics由冻结env源码确定，不能把setter hook冒称native engine逐substep instrumentation。

独立从actual/request数组重算，而非直接相信 `goal_actual_difference` 字段：

```python
delta = actual_state7 - requested_goal7
wrapped = delta.copy()
wrapped[4] = (delta[4] + np.pi) % (2*np.pi) - np.pi
```

复算值与报告raw/wrapped数组精确一致。原始角差为0的6次、+2π的10次；
16/16的wrapped向量全部为零。32次setter的before/after block linear velocity与
angular velocity也全为零；这只约束捕获时刻，不声称逐内部积分时刻均为零。
24/24 render的before/after snapshot相同，未观察到render推进state。

以下表按稳定 `(candidate_id, proposal_index, attempt_index)` 列出，不用列表序作身份：

| identity | selected | random/placement goal raw角差 | 两类goal wrapped最大绝对差 | 第二次placement setter实际修正 |
| --- | --- | --- | --- | --- |
| (0,0,0) | 否 | 0 / 0 | 0 | 最大位置分量 2.842170943040401e-14 |
| (0,0,1) | 是 | +2π / +2π | 0 | 0 |
| (0,1,0) | 否 | 0 / 0 | 0 | 0 |
| (0,1,1) | 否 | +2π / +2π | 0 | 见下文真实小修正 |
| (1,0,0) | 是 | +2π / +2π | 0 | 0 |
| (1,0,1) | 否 | 0 / 0 | 0 | 0 |
| (1,1,0) | 否 | +2π / +2π | 0 | 0 |
| (1,1,1) | 否 | +2π / +2π | 0 | 0 |

未选 `(0,1,1)` 的第二次setter：

- random reset：block dx=0.28551498302729783，dy=0.46161116066485874，
  wrapped dθ=-0.004587979651671681 rad。
- placement reset：block dx=0.34300441819530647，dy=0.4067566673744807，
  wrapped dθ=-0.004587979651670793 rad；raw dθ=6.278597327527915，
  不能把其全部视为真实转动。
- 两类goal setter本身的wrapped差仍为0。修正发生在各reset恢复start的第二次setter。
- 本attempt的原source domain accepted=true，但按预先固定的first-legal规则未选；
  不能据这次诊断改选、删分母或重定采样域。保存信息不足以认定修正具体来自哪一
  接触/cache机制；这里不把“发生过内部physics”进一步解释成已识别碰撞根因。

8/8原random/placement reset summaries精确重现，8/8 explicit RGB SHA与原journal
相同；8/8 returned placement state与explicit render前state相同。两条selected根
的state、goal、root SHA独立重算相同，原NPZ的current/goal RGB byte SHA分别与实际
hook记录相同；producer另记录了当场逐数组exact equality全部true。本次只读复算
不能重新获得未另存的实际RGB数组，故独立RGB验证准确说是保存数组到实际hook SHA
的一致性，而不是新render。选中raw history的10×2 float32全零padding也经数组核对。

## 后续source观察历史：具体可执行且不追认旧seal

1. **下游reset必须重建原请求。** 保持原root seed、reset span、proposal/attempt、
   placement seed、source/env hashes作为输入；先random reset再sample_state再placement。
   不把NPZ的post-state传回reset。执行前后核对selected state/root/RGB及hidden velocity。
   本批已有精确重现证据，但不等于任意savedpoststate reset也成立。
2. **新独立版本保存实观测history，不给旧padding改名。** 在任何新结果前固定同一
   有限source池及每根10个raw动作的具体float32字节、2个5步packet、frame采集时点。
   按真实初始RGB F0 →执行A0[5]→F1 →执行A1[5]→F2保存history与outgoing动作，
   root state/identity必须对应F2，而非原F0。`zero raw`也必须实际执行并观察，不能
   推断为静止帧；当前旧三帧仍只是ONE observed frame的重复padding。
3. **时序及费用逐项封存。** 每个public step记录attempt/completed、raw action、
   returned state、render SHA、frame index；记录全部失败/部分history。若沿用8attempt
   全分母并每attempt执行10步，则新增上限80 public steps及16个F1/F2显式render，
   连同每attempt原F0共24显式render；原16 resets、16内部goal renders另记。
   编码/评分属单独主线程协议，不能因history变真实而自动复用旧context或旧task ID。
4. **不得按screen结果挑prefix或换root。** 新source变更会改变真实root与任务，旧
   direct结果不能追认为该history的结果；保留calibration/confirmation区分并重新
   freeze必要输入。上述是下一协议建议，本轮没有实现/执行或授权新的source实验。

另保留具体评分合同观察：安装env.eval_state用未先取模的
`min(abs(delta_angle), 2π-abs(delta_angle))`。当未规范化goal使角差超过2π时，该项
可为负；`state_dist`也使用未wrap的state向量距离。这是独立纯算术可验证的表示
敏感性，不是此次goal RGB几何漂移证据，也不是已发生的P1假负面。后续新consumer
应同时保留原official指标与独立wrapped pose诊断，明确区分，不修改旧official门槛。

## 证据级别

- mock：原driver的12项接口/失败留存测试只验证工程合同。
- 实际：主线程完成固定8attempt reset/render；本审查复算其保存数组、计数和hash链。
- 科学准入：未验证真实history改善模型或控制效果，不开放P1，不把风险未出现等同
  于所有代码原因已排除，也不继续假定这些风险已造成负结果。
