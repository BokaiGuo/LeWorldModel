# P3 无 shadow 面板的物理前置与成本

新增 saved-only 绑定 `codex_p3_noshadow_physical_bind_20261030_v1.py`，先以真实 report SHA 调用原 no-shadow 独立语义 consumer，再使用原 moving sourcepack 和原 task bridge，验证两根 SHA、task export、blind ID、原 RGB history 与 goal RGB、原 warmup raw history。六个赢家保留原模型选中 raw10 float32、逐位 SHA 和候选/任务/根身份，按 root-major 三策略完整输出；不从模型终点推物理结果。

产物 `results/codex_p3_noshadow_physical_bind_20261030_v1_01.json` SHA `64d565d18fe7ebe85d65022a7bbc7322b87bb64d648b72337300893f5d719413`。其中每根/臂的 action_encoder、predict、predictor、pred_proj 实测 attempt/return 分列；root0 uniform/signal/reverse 分别各轴 20/30/10，root1 为 20/10/30。对应 metric 10/15/5 与 10/5/15。每臂跨根各轴总数 40，但逐根不是等预算比较。

原权重来自先前已付、两根共享的谱信号；本次搜索没重采该信号，不能把同一历史 acquisition 写成每策略独立执行或免费获得。前序 panel 总账混有原搜索，不能把总 520 predict 等同于信号成本；本绑定保留未分配的历史成本，未证明整条成本、FLOPs、wall、公平或 per-root 信号账相等。历史 shared 的引用模式是描述，不是新公平合同。

现有 `chat_p3_20261027_physical` 固定另一 producer 的八臂和其消费者，`agent_p3_fixed_execution_physical_20261028_v2` 固定另一 producer 的四臂五步；二者不能通过改标签或复用旧 receipt 来物理执行这组六槽。真正可运行前须新六槽合同、动作长度（winner raw10 或首包）和相同根物理预算、原 PushT reset/goal/warmup/RGB 运行时重认、动作 ACK/早停、耐久 journal 与独立 saved consumer；不更改旧物理 gate。此次无环境或模型启动，无新增 GPU/物理，source join 不是 runtime root 验证或物理放行。

离线测试 `tests/test_codex_p3_noshadow_physical_bind_20261030_v1.py` 与 `tests/test_codex_p3_noshadow_20261030_v1.py` 共 4 passed；覆盖六槽与逐根计数、错误根和动作域篡改。所有根均已知、结果探索性，不声称 P3 通过。
