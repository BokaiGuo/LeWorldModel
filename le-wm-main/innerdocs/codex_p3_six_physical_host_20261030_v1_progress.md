# 六槽物理 host 进展：尚未运行

`codex_p3_six_physical_host_20261030_v1.py` 新增来源/赢家重建、独占 prepare、六槽固定分母、真实 moving branch、actual ACK、物理操作 attempt/return 和失败关闭。原完整来源 consumer 与历史 sourcepack 必须先成功。prepare `results/codex_p3_six_physical_host_20261030_v1_PREP_01.json` SHA `7250d1365a1d8530e003a84737ecaa5cc2fd5cd1d423523e821417a5e3c67e80`；实际 `verify` 已通过，6 plans / 2 references。旧结果不改。

检查当前 `results/*p3*physical*/report.json` 中 `selected_plan.commands` 的 raw10 和 root，与六槽三种不同 root/action identity 比对，未发现精确重复；这是有限目录、有限 schema 搜索，不证明整个工作区不存在重复。六槽内部存在相同动作：root0 uniform/reverse 相同；root1 三臂相同。它们仍是策略槽，不是独立样本。

新增 saved batch consumer v1 对事件、操作、ACK、frames 和指标做语义验收；审查发现其把独占 prepare 写到物理目录后的 JSON 重序列化副本当原始 SHA 来校验，字节不一定相同。新增独立 v2 consumer 接收原始 prepare 路径/SHA，核对副本语义并从认证来源重算 warmup RGB/state/snapshot 和根目标；v1 不应直接用于放行。v2 是 prepare 后新增的审计代码，不在本 prepare source list 内，且尚无完整六槽 event/frame 模拟往返与篡改测试；目前仅有 ACK helper、合同与 host 失败关闭等11项离线测试通过。

因此未启动六槽物理，避免在消费者尚不能验证生产结果时制造不可重跑批次。下一步需对 v2 验证完整真实结构的六槽成功、终止、frame/goal/ACK/census篡改及外部 SHA 失败，固定并绑定 v2 后重新生成独占 prepare；再核环境前置、明确已知重复边界、执行一次并保存所有失败。原行为无神经推理，不改变 P1/P3 gate。每根策略搜索调用不等；共享历史信号、物理成功与全管线公平均不可由此确认。
