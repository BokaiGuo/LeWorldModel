# P3 child physical saved-only v2：精确 census 修复

本版只消费冻结 v1 的原报告，不执行 prepare/run/neural/physics，不改原 producer、日志、PNG、报告、
prepare、阈值或科学门。原三文件 SHA 固定在新源码 PINS，消费前后核对。

根因：原 native PushT.__init__ 只初始化 space=None 等字段，没有 reset；native_factory 的 gym.make
也不应虚构一次 reset。P4.setup 实际显式 reset 两次。每次 reset 两次 _set_state，各积分一次，
且一次 goal render/setter；再加独立 attach_goal setter 一次。因此逐分支精确值为
reset2 / set_state4 / goal_render2 / goal_setter3 / setup physics4，而非 v1 consumer 的3/6/3/4/6。
这是消费端代码假设错误，不是新物理失败或理论负结果。

完整复制冻结 v1 validate_saved，只替换两个精确 census 字面表达式：

- reset=3,set_state=6,goal_render=3,goal_setter=4 → 2,4,2,3。
- physics_step=6+10*(10+b+n) → 4+10*(10+b+n)。

测试逐字比较整个函数，保证没有其他验证变动；不 monkeypatch 旧模块。
constructor/close/goal_bind/public/render 的精确值与原所有 native/PNG/source/context/raw/goal/metrics/
seal/dispatch/ledger/早停/48分母验证保持。旧 prepare cap 是保守上限，**不改 cap、不改账、不冒充实际值**。
外层仍调用冻结 v1.verify，完整经过 material → compiler.consume → 真实full strict → P4.consume，
不调用带错误精确 census 的旧 base.consume。新 CLI 只有 consume，没有实验执行入口。

## 真实 saved authority 与回归

报告绝对路径 `/home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_child_native_physical_20261028_v1_MAIN01/report.json`，
SHA `56c3e97208597bdc8424dcfb263c8840683c6b3413abeb79d01fa46762dc5df6`。
prepare SHA `e2a8752f351d47d780a49774c07b9bead1a33c75690ca286ef9bff4727681f95`。

只读逐分支账确认41分支都满足新精确公式；状态37 returned5 +4 native_stopped，原5拒绝+2not_reached全保留。
四条 native_stopped own步数按原顺序为5、2、5、1（到第五步才stop同样不能改成未stop）。
实际 own198、恢复795、public993、physics10094=41*4+993*10、PNG1075；
reset82 /set164 /goal_render82 /setter123 /constructor41 /close41 /goal_bind41。
native jointXY+angle成功与native7 reward不混同。不padding四早停，不把同两根扩展成41独立root。
其中e01 I/R b05 fixed是相同动作及原生终态的重复轨迹；4条成功不是4个独立成功，完整保存不去重也不扩大独立样本数。
P1/P3/C3/scientific gates均保持false。

测试是saved/RAM回归，包含原v1精确失败、新完整validate、额外一次reset/setter等仍拒绝、raw/context/
2pi/snapshot/goal/metric/早停补步/分母/资源篡改；不执行原 full verify 的真实外层闭包。
worker不写MAIN产物；实际完整 consume 仍交MAIN审读后运行。

## MAIN 命令

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -m pytest -q /home/gbk/code/LeWorldModel/le-wm-main/tests/agent_p3_child_native_physical_saved_20261028_v2_test.py
/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -m experiments.spectral_reachability.agent_p3_child_native_physical_saved_20261028_v2 consume --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_child_native_physical_20261028_v1_MAIN01 --report-sha256 56c3e97208597bdc8424dcfb263c8840683c6b3413abeb79d01fa46762dc5df6 --receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_child_native_physical_saved_20261028_v2_consume_MAIN01.json
```

独占NEW回执，禁止覆盖，完整rows持久保存；CLI只打印compact path/SHA/bytes。
本修复不重新执行模型或物理，不改变历史真实预算，不成立等实际算力或C3。
