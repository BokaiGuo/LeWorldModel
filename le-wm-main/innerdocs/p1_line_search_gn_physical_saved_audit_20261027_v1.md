# Line-search GN physical：独立 saved 证据审计

本NEW审计只读取已完成产物，以CPU NumPy/Pillow核验；不加载模型、不导入物理producer、不调用producer的 `analyze/verify/material/collect`，不调用gym/Pymunk，不产生新物理或神经执行。只复用task core的 `validate_task/task_id/check_prediction`，范围为原任务身份与保存的winner预测合同，不以该core独立证明模型/J正确。

输入固定：

- physical实际report：`results/p1_line_search_gn_physical_20261027_v1_main01/report.json`，SHA256 `3a41fe83f375e5bb3f390e9f3a0e8e17dd70c2d49a7ebe3a99079f5774de85c1`。
- prepare02：`results/p1_line_search_gn_physical_20261027_v1_prepare02.json`，SHA256 `f36f915ccfc2687da056810d1241da433190eff430bb54fa7b7b35a6af3818d5`。

## 独立检查范围

外部SHA认证physical report/prepare，流式重哈希prepare中源码闭包；从原search prepare/report绑定实际CUDA root winner、task文件，再沿observed-source prepare绑定原sourcepack/request和既有reset_goal_audit真实收据。全部访问过的输入在审计末尾重哈希，拒绝TOCTOU；不读取MAIN正在live的新RGB CUDA结果。

两根root/task/原goal/identityQ/native192、winner commands/prediction/steps/selection digest逐项绑定。raw必须10×2、有限、[-1,1]且精确float32可表示；命令不clip、不二次tanh、不从physical结果选优。观测0/5/10数组逐字等parent、raw与normalized history同task，仍是原两根，不新增独立样本。

逐根复核rootseed/proposal/attempt与有限resetspan；实际random reset seed分别202781011、202781014，placement seed均0。journal必须先random reset取得原native goal，再提交原placement request，再placement reset；options不得替换为saved poststate。独立重算SeedSequence/rootseed/proposal/attempt产生的object xy jitter、angle与零velocity；完整pusher placement xy则与SHA绑定的此前实际reset_goal_audit `placement_requested_state`精确比较。**这里不重新执行polygon sampler，不独立证明Pymunk polygon vertex顺序或随机reset内部实现。** 原source/environment/recipe SHA同时认证。

严格Cursor重建282条hashchain事件的总顺序：两reset/branch、逐command attempt→return→termination→state/info→native/wrapped→render→fresh guard，检查全部原始step receipt与最终report一致，禁止缺项、额外事件、重复/乱序dispatch；branch_finished是未附加object_only的原收据，object_only_result/branch_closed是新收据，不混淆阶段。

解码六张0/5/10 PNG，检查原生RGB224×224、文件SHA、像素SHA、checkpoint state；初始图与原source RGB逐字相同。PNG中RoyalBlue像素数量独立重算并比对guard。其余16次render没有保存PNG，**只能核对journal的RGB hash/guard count收据及因果顺序，不能从不存在的图独立证明像素内容。** 新reset的goalRGB相等仅有producer检查收据，原goalRGB数组可hash绑定，但没有单独保存本次goal render供独立像素重验。

每步native info的pos_agent/vel_agent与state7一致，unwrapped block_pose角经mod与state角一致；用实际goal_state独立重算原native公式、distance、negative-distance reward、wrapped diagnostics、native/terminal discrepancy和object-only位置/角度误差。比较native与wrapped但不覆盖native terminal。官方info中的 `goal_pose`是另一字段，不能代替本批options的实际goal_state；审计另计其与任务goal的失配次数，不据此改目标。

精确public计账：2branches、4reset、20step、22render、6PNG；目录仅report/journal/六PNG，总量≤2MiB。内部reset physics8/goal render4仅源合同×完成reset的推导值，实际substeps仍null；本审计不将其升级成新hook测量，也不能从saved文件证明执行时空闲磁盘。

## 运行与测试

在 `/home/gbk/code/LeWorldModel/le-wm-main` 运行：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_line_search_gn_physical_saved_audit_20261027_v1.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_line_search_gn_physical_saved_audit_20261027_v1
```

测试只读真实saved输入；篡改在内存deepcopy中，重算hashchain后仍须被语义检查拒绝。不创建pytest tmp/cache/results，无GPU/physics。CLI仅stdout，不另落盘报告；三新文件之外不写研究证据。

本版已实际执行上述saved审计与纯数组/PNG测试；最终测试记录见交付消息。不是MAIN实际物理的再次执行。

## 本批独立数字与边界

第一轮saved审计已实际执行通过（不是新物理执行）：282events、两branch、4reset/20step/22render、6PNG、errors空。

| root | native distance 0→10 | object position error 0→10 | wrapped object angle error 0→10 |
| --- | --- | --- | --- |
| 0 | 242.2396808055→301.4088554290 | 9.1670739711→9.1670739711 | 3.0176566157→3.0176566157 |
| 1 | 319.3081963126→321.9008420121 | 13.6913478521→13.6913478521 | 1.7825133276→1.7825133276 |

两根native终态success均false。所有11个保存的state（初态+10 successor）中object pose均逐字stationary；20步native info contact count均0。本批没有negative native angle term，也没有native/wrapped success分歧。`goal_pose`静态字段与实际任务goal pose不一致，不用于本审计的目标定义。

这些数字只支持同两calibration根的保存证据一致性及本组动作无物体pose变化；不证明模型物理效能、direct plateau、two-stage机制成功、全局freshness或P1 admission。globalfreshness unknown；没有新增独立样本。浮点1e-12容差仅用于重算收据一致性，不是科学准入阈值。
