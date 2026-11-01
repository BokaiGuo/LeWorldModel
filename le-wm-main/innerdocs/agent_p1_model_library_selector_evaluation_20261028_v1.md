# 封存selector的合法同位执行复用评价（探索性）

独占新增source/tests/本文，先交MAIN审读，不执行真实saved CLI、不写生产结果、不做神经/物理/网络/训练。旧selector、panel、所有封存结果与共享文档不改。使用experiment-code技能组织受限实现/RAM验证。继承旧blocked P4 goal不覆盖；个人交付为实现、轻量单测与冻结三SHA，不称研究完成。

## 固定authority和单向边界

输入selector为MAIN实际`results/agent_p1_model_library_selector_20261028_v1_MAIN01.json`，SHA `f807d2ffe2d7ffb4d012852302f199b9edb64afc5a23cc36ac264c6adea6c1bb`。先独立selector.build并要求完整相等，之后才打开本评价器的physics/panel输入。selector函数从不获得物理labels，不重新选词或阈值。算法身份是新探索性model-Q stable argmin，不能改称原策略已经采用。

完整panel为最终NASH02，SHA `7eda1333a3bf2c8ef1d46deaf67d74c6d464498c018b52cf35e9802204501e5d`，重新完整panel.build，包含原full partial40 native consume、全部输入/模型/源码/prepare/artifacts/数组/真实journal闭包。绑定原physical MAIN02 SHA `464656755bb83cec14b7acdd457641592b11be6d335f285166687df15368bc01`及其prepare、原teacher report/receipt。两闭包有同路径不同SHA即拒绝，不未来补签或仅日志hash绿灯。

## 逐槽同位核验，不按word或索引猜

全部40槽（含未选槽）逐项核：原model plan digest、task/root/condition/role/library/anchor/blind身份和原physical job digest；相同prefix5+suffix5逐位rawfloat32、full10，实际warm10+prefix5+suffix5全部20条raw_return的actual bits/dtype/phase/native无早停。所有40原预测Q/原plan/原prediction binding保存，不能只给8个有利结果。

同一个原compiler context digest、原latent goal192/Q与原physical prepare teacher_input精确相等；prediction shifted-context receipt与panel原预测数组一致，只继承已修正的endpoint语义。原native goal7、outline3、同condition goal pixels与panel原goal_binding一致，不把renderer outline当taskgoal。

真实root10/start15/return20严格核state；suffix-start checkpoint15的原PNG/pixels必须与teacher anchor一致，pre/post snapshot不变。原teacher_history_verified事件核warm后5+原prefix5的真实raw10、原normalized tokens及context/PNG digest。每role的五个独立旧执行必须同实际start15及snapshot。它们是原recipe独立重建的原执行，不是新reset(poststate)或近似恢复；不能把只同word或相近状态冒同位。selector动作仅suffix5，不重复派发prefix。不同role起点可以不同，不跨角色错接。

缺身份或不相等即整体fail closed；不补根、不挑最近者、不删除失败槽。index12的末图缺失使原actual latent/Q未知，但实际20 native returns完整，故物理state/metric可合法复用。保留其physical status=failed及完整分母，不把它当horizon_completed、不把latent未知填0。

## 结果与身份

输出完整40 panel rows、40 exact reuse witnesses、8同role策略对。每对分别列封存selector选中原plan、原生成diagonal（Z:uZ/S:uS）、自身zero控制及selector相对diagonal变化；每项原native distance/success、joint/agent/object/angle等指标与returned reward分列，原predicted/actual latent Q另列。success ties不算改善，coverage原无authority仍null，不合成分数或事后选指标。step10旧goal/reward与bind后taskgoal基线整行保留，不混用。

本consumer只评价封存见证在**已看过的真实旧执行**上的对应结果，不声称新selector实际dispatch过；两种策略身份不能合并。只有2known roots、4条件×2roles，不膨胀为8/40独立样本。不是fresh确认，不是P1/C2，不作可控集合或非贪心因果结论。新模型/物理调用0，不修改原selector，不把seen outcomes反馈入选择。

## MAIN审读后saved入口（本次未运行）

在 `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p1_model_library_selector_evaluation_20261028_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_p1_model_library_selector_evaluation_20261028_v1 analyze --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p1_model_library_selector_evaluation_20261028_v1_MAIN01.json
.venv/bin/python -B -m experiments.spectral_reachability.agent_p1_model_library_selector_evaluation_20261028_v1 verify --evaluation /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p1_model_library_selector_evaluation_20261028_v1_MAIN01.json --evaluation-sha <实际SHA>
.venv/bin/python -B -m experiments.spectral_reachability.agent_p1_model_library_selector_evaluation_20261028_v1 consume --evaluation /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p1_model_library_selector_evaluation_20261028_v1_MAIN01.json --evaluation-sha <实际SHA>
```

只独占新建输出、上限32MiB；verify/consume完整重建而非hash-only。RAM测试复用冻结selector的RAM fixture，不运行其生产build；不是新算法科学证据。实际新增39项、联合selector与panel共116项RAM测试通过，包含原producer完整布局mock、单向读取顺序、40槽逐操作字节与身份变异、12未知/失败分母、JSON round-trip和重签名篡改拒绝。相同原candidate身份与相同raw动作分别报告，重复库词不合并身份。未决：MAIN真实verify selector及本consumer真实saved接线/同位结果验收；本次不产出生产结果或虚构其SHA。冻结后修复另起版本。
