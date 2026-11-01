# P1 双包候选：原赢家来源重建

主线程执行 `agent_p1_two_packet_winner_authority_20261029_v1.material`，对原 mixedmode 主机结果运行完整保存消费并重建四槽输入。原报告 `le-wm-main/results/agent_mixedmode_prefix_20261028_v1_cuda_MAIN01/report.json` 的当前 SHA-256 为 `d6166b98958af70cecb8529135396c77467b7c803bfe12a1a5de29c422c22ad2`；四槽赢家索引依次为 51、51、23、23。来源绑定检查和 prefix/suffix 原参数、动作 chart 逐位重放均通过。

独立运行 winner authority 与 uphill generator 两份测试，共 15 passed。这次重建未运行神经推理或物理执行；SHA 是对当前文件计算并供消费者校验，并非额外的独立时间戳或前瞻承诺。结果只支持新生成器可以引用经认证的旧赢家输入，不代表新候选已经生成、P1 通过、独立根确认或物理释放。未来 runner 仍需完整 native 调用账本、双臂计费和保存消费。
