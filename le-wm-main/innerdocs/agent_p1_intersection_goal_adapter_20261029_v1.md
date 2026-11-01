# Intersection v2 → P1 原目标前瞻 spec 投影

该独立 saved-only adapter 读取**本次新执行** intersection v2 physical prepare，严格绑定实际文件 SHA `fa4a694e24652fbd7edb804eb45814e7f70ae40e5579ff87d4352e5b4ba8b527`、prepare 内原物理源码 SHA `c6dc8d641f679a6e17079a921d3fd101dd25bc1f59620e6925cf97ee239eab3d` 和模型完整 receipt SHA `bcaba6c3c4cbdd05dafc475ad45ad859930efd3587fdb8e8adba47f7da7b7432`。不重复上游 full strict；`complete_consume_rebuilt` 和 receipt 相等声明继承已保存的物理 prepare，不是本投影重新认证模型。MAIN 须独立审阅 source closure、prepare 和时序后决定使用。

投影固定六槽：root0 joint/composition/grasp，root1 grasp/composition/joint；原 root SHA、goal7 按 float64 位级与每 plan `native_root` 核对，检查点 0/5/10/15，`rgb_pairs=[]`。两个都是已知根，不声明新根、P1 准入、同图不同速度或真实物理结果。输出只是与 `agent_p1_prospective_goal_semantics_20261029_v1.spec` ABI 一致的输入；authority 指向原物理 prepare，而不是本 adapter 自签。

**已知接缝：**该 prepare 的 `contract.fixed_denominator` 是继承的 4，但 `order` 和实际 plans 均为 6、物理报告 own denominator 为 6。这里显式验证该原值并保留六槽，不伪造改写原合同；MAIN 必须核查该不一致是否妨碍其物理合同验收。若其修复导致 prepare/source SHA 改变，本 adapter fail closed，须另行授权新版本。投影不证明 physical prepare 在具体执行之前封存，也不证明分析者未看到相关历史或模型 outcome；须 MAIN 用实际可信时序确认分析 spec 在本次六条物理结果可见前完成并封存。已知历史 P2 结果与本次新运行不能混为一谈。

本工具仅事前投影，不调用 `prepare/verify/consume` 原生产入口，不启动 NN/环境/物理；真正运行之后需要独立 receipt adapter 与原完整 saved 验证，当前未实现，不能用 fixture 当真实报告。原目标和 gate 不修改。

从 `/home/gbk/code/LeWorldModel/le-wm-main`，绝对路径且独占写入：

```bash
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_agent_p1_intersection_goal_adapter_20261029_v1.py
.venv/bin/python experiments/spectral_reachability/agent_p1_intersection_goal_adapter_20261029_v1.py project --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p1_intersection_goal_adapter_20261029_v1_spec_MAIN01.json
.venv/bin/python experiments/spectral_reachability/agent_p1_intersection_goal_adapter_20261029_v1.py verify --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p1_intersection_goal_adapter_20261029_v1_spec_MAIN01.json
```

MAIN 如审查接受可用投影文件的实际 SHA 调用原 P1 合同的 `prepare --spec ... --spec-sha ...`；原合同明确 `ready_for_execution=false`。不得事后将 spec 标成事前冻结。
