# 第二独立P4批次前置条件：saved-data准入审计

独占新增同stem审计器、tests、本文；不修改attribution v2、旧runtime、terminal20 encoder或任何旧seal。只读取保存源与AST、做hash/census，不导入模型/runtime/compiler，不运行神经、AD或物理。实验技能用于组织实现及mock验证，不生成实验图片或额外模型产物。

## 判定对象与范围

审计的是目前已经存在的旧geometry planner→capability→feedback response→三轮runtime完整source/protocol链，加上报告中逐项列出的本地P4 protocol/report/receipt与geometry protocol inventory。它不是遍历全磁盘证明所有任务绝无新数据，也不把其他环境/P5数据冒作PushT同协议根。

“第二独立批次”至少须与首批root SHA不同；同根两臂、三个role、重复执行、换输出目录、terminal20派生编码、保存数据审计均不满足。与首批root不同的旧校准根，也不自动成为前瞻fresh heldout根。报告分开`root_disjoint_from_first`、`source_chain_complete`、`engineering_prepare_eligible`、`fresh_holdout_verified`与P1/P4科学门；不把P1未通过单独说成不能准备任何工程校准。

## 完整已知四根排除链

97MiB原planner报告经capability协议保存的精确路径和SHA允许一次128MiB读取例外，不复制到新目录。其余JSON/源码默认16MiB，解析累计256MiB；流式hash单文件1GiB上限。四根顺序与原runtime original_root_order一致。

| root前缀 | geometry selected / greedy | 第二批排除 |
| --- | --- | --- |
| 9173218f | prefix_7 / prefix_7 | 已用校准根；旧筛选不进入capability；无同根response/runtime pack |
| 23b8de78 | prefix_6 / prefix_10 | 唯一旧model-qualified根，但正是首批根；两臂/三role不增加根数 |
| 4afefa3b | prefix_7 / prefix_7 | 已用校准根；旧筛选不进入capability；无同根response/runtime pack |
| cf1b9bb9 | prefix_7 / prefix_7 | 已用校准根；旧筛选不进入capability；无同根response/runtime pack |

完整SHA及全部逐根排除理由以机器报告为准。旧selected≠greedy仅是有限模型诊断的producer筛选规则，不是P1通过、物理收益、或其他根“不可能支持反馈”的证明。严禁根据已保存任务结果重选根或改阈值来补齐第二批。

capability仅这一根具有3role×41probe=123条已存分支。response同根selected/greedy/zero三个role不是三个根；runtime源码ROOT=response.ROOT、root_index1、source/census/重置root fingerprint均固定。修改seed或root_index而保留旧protocol/source seal不会形成合法新任务。新Q/其他调度器的旧根模型选择也不替代这条同协议反馈source pack。

## 合法真实反馈消费位置

从原源码AST读取行号及表达式，并与原runtime source manifest中的SHA相符：

- 在step5/10/15前，`backend.encode(frames, between, ...)`消费本臂实际三帧及两条已完整执行packet。
- `chosen_context = observed if arm=='actual' else predicted`规定只有actual选优消费真实latent；imagined虽同样计编码成本，但仅选用自身预测context。
- `backend.compile(deepcopy(chosen_context), ...)`消费该context并在后续动作dispatch前封存selection。
- 完整五步guarded packet之后才`advance_observed`，滚动为旧frame1/frame2/本臂terminal RGB与旧between1/本臂executed packet；partial/abort不得推进。
- step20派生真实latent仅做末端评价，不能倒灌到step15选择，不能变成第二批。

报告区分“准备前必需的source pack”与“未来执行中才产生的真实反馈”：不要求尚未运行的新批次step10/15/20观测预先存在，不写占位观测或伪造predictor context。

## 当前不可准备时的实际缺口

1. 与首批根不同、可复核身份的root-specific source pack：RGB3/raw历史/native context、原goal/Q/model/reset来源、prefix预测/实际RGB见证及同根capability/response绑定；现有链交集没有第二根。
2. 支持该根的NEW reviewed runtime/prepare合同与源码：旧sealed root/source/census不可被静默替换。
3. 若要独立泛化，须在新outcome前冻结有限root roster/seed/task/exclusion manifest；现有四根都已用，不把旧校准升级fresh holdout。
4. 独立P1科学准入仍缺失；不因本次工程/来源核验而打开P4门。

故本次只输出带source seal的小`report.json`，`can_prepare_second_batch=false, new_protocol_written=false`；不创建`protocol.json`或空source pack。若未来出现未知的新完整候选，当前工具应拒绝自动冒用旧producer，需独立新版本审查其source consumer后才准备真实协议。

## 运行

工作目录`/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p4_feedback_second_batch_preconditions_20261027_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.p4_feedback_second_batch_preconditions_20261027_v1 --output results/p4_feedback_second_batch_preconditions_20261027_v1_main01
```

所有读取文件复hash；记录v2原报告input closure未变，新审计自己封存源码/tests/本文，不覆写旧文件。audit完整性错误另写failure.json，不把坏hash或程序异常混作科学负结果。禁止神经/物理/下载/训练、没有删除资产或代理派发，输出报告硬上限1MiB。
