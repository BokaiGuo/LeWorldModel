# P5 H1 JEPA 零响应与 native callback v2 量化集合

独占 saved-only source/tests/innerdoc。不运行 NN/物理/重编码，不训练或修改 renderer/模型/旧资产。不做专职代码审计；这里实现完整80对的身份连接与集合分析，不从“两个数量同为24”推断同集合。

## Authority 与读取边界

CLI 显式输入 H1 report、H1 strict receipt、callback v2 report、callback v2 strict receipt 的四个实际 SHA。H1 receipt.pairs 只是整数80，不能当逐对列表；具体响应/label tensor在其实际 report。callback receipt.pairs 才是完整80条 per-pair equality。两份 strict authority 只读复用，不重新调用两个上游 consume，不冒称再次完整重建4GiB callback transcript/模型IO闭包。报告明确 full_upstream_consumers_rebuilt_here=false。

本版核实际 prepare 字节/hash、共同历史 physical prepare/report SHA、需要的原RGB源绑定、原164 jobs、每分支 root/condition/task/selection/raw10 与源label对应。读取分析实际使用的保存数组/PNG字节并前后rehash；不重新执行 encoder，不解码重画PNG，也不将已认证 flag 冒充新跑 native callback。

## 固定80身份与 unknown

固定四panel `(root0,off),(root0,on),(root1,off),(root1,on)` × h(.01,.001) × raw10 coordinates(0..9)。每对 exact key 为 slot/root_slot/condition/h/coord/minus_index/plus_index，正负方向还必须与原 job.sign/id 和各自 raw10 位对应。不可仅按序号、root、数量或 nearest h 连接。顺序变化可按完整身份连接；重复歧义、缺字段或不可映射来源报告 missing/null，保留完整80分母与 blocked_missing_pair_mapping；不替补/猜测。

逐分支要求 H1 context global5/10/15→20，使用共同 historical physical source 的原 terminal plan10 PNG；callback v2 新重放 selected terminal 的原观察关联和实际224 SHA/pixel SHA必须完全对应此label。同一 raw10、task/selection、图像来源完整闭合才把本次重放 equality 与历史label JEPA观测响应放到同一行。

## 数值与集合

从原 float32 label_patch `(1,256,384)` 全量保存数组，独立 float64 plus-minus L2；分母固定全部80，无epsilon/近零合并。重算值须等于 H1 report.observed_response_l2；observed_response_zero 仅为精确 norm==0。与 callback receipt 的 integer_primitives_equal / surface512_equal / rgb224_equal 三层分别报告完整成员ID、intersection、equal_only_known、zero_only_known、2×2 contingency 和缺项。只有无unknown时 full_sets_equal 才给 true/false，绝不以同计数代替集合核验。

不强制三个层的membership必须相同。原receipt标出的 unexplained_same_primitives_changed512 / unexplained_same512_changed224 等异常保持，不为得到因果解释而拒绝或删除这些行。callbacks_equal 和 first_loss_stage 逐行继承明确标记，不重新推理量化唯一机制。

## 历史限制 / 科学边界

callback v2 是本次真实重放的 tap→integer primitives→512→224 捕获；原历史执行没有完整callback/cache证据。因此即使实际224/state/raw逐位对应，historical_callback_membership=null、sufficient_original_cause_certified=false仍保持。分析只能报告新重放链与同源历史JEPA标签的精确集合关系，不能宣称旧callback必然同一、唯一充分原始原因、P1失败已解释或公平跨模型效能。两known roots、off/on非独立根；P1/P5 admission=false。

## MAIN CLI

工作目录 `/home/gbk/code/LeWorldModel/le-wm-main`，仅worker fixture tests，真实saved分析交MAIN审读执行。下列均为已存在真实SHA；无future report。worker未执行生产analyze。

```bash
.venv/bin/python -m pytest -q tests/test_agent_p5_jepa_callback_zero_sets_20261029_v1.py

.venv/bin/python -m experiments.spectral_reachability.agent_p5_jepa_callback_zero_sets_20261029_v1 \
  --jepa-output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p5_raw164_jepa_cuda_20261029_v1_cuda_MAIN01 \
  --jepa-report-sha e7ac4ef9eb7491f619de25b04c55f5e56560e2833a840dd5305f3db6207b3a5d \
  --jepa-receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p5_raw164_jepa_cuda_20261029_v1_consume_MAIN01.json \
  --jepa-receipt-sha 4f6bbed041f077818627e68157b0cfab99ddba2e38100827db10f46c4aa1b759 \
  --callback-output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_raw_native_callback_capture_20261029_v2_MAIN01 \
  --callback-report-sha 9f61d3ab2da00d8b1ef3f3426602ce5bd2a5c37c3b0697f599f81abc5567bb74 \
  --callback-receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_raw_native_callback_capture_20261029_v2_consume_MAIN01.json \
  --callback-receipt-sha 49c9a4d38fa8d271bfdd6924f56d9be97683c57fa213437d59310edbc3abdc9f \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p5_jepa_callback_zero_sets_20261029_v1_MAIN01.json
```

metadata读上限256MiB（原physical prepare较大），单NPY4MiB，独占新输出32MiB、disk floor200MiB；文件内flush/fsync后fsync_dir，不覆盖旧结果。fixture 用明确RAM marker/NumPy保存数组，不是神经/真实PNG/物理执行。覆盖同24但错位集合、完整80连接、shuffled identity、minus/plus/root/overlay/h/coord/符号/原raw/任务/selection/PNG/time缺项、歧义、数组/外部authority篡改、unknown及历史限制。最终测试结果与冻结三SHA见交付消息。
