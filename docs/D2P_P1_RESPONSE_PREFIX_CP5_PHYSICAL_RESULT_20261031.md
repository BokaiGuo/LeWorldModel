# 结构化前缀：真实 cp5 反馈规划及物理执行

固定 16 行中，14 行具有认证的真实 cp5 RGB；主机 CUDA 使用四个
相同初始化、每起点 20 次 Adam、十步后缀完成规划。另两行此前
在 cp5 前越界，保留为不可规划。CUDA 结果
`le-wm-main/results/p1_response_prefix_cp5_adam_20261031_RUN01.json`
SHA-256 `705b75f68bfbb6663e7731428580e1639344a7afb7e9c329aa2a9ebc17d3327b`，
前后模型参数与缓冲区哈希一致。代码路径计数 14 次编码、2352 次
action encoder、2352 次 predictor、1120 次反向；它不是独立 hook
账本，且不包含前缀生成成本。

14 个计划均在真实物理环境执行，执行前原 warmup、前缀、cp5 帧
及状态与采集记录逐帧核对。独立保存态消费者逐行验证动作、
journal、图像、终止与删失：

- 13 条完成十步后缀，均未达到原生成功；
- 1 条在后缀第 4 步越界删失，不补终点；
- 原 16 行中另外 2 条在 cp5 前越界，根本没有后缀规划。

不能将删失视为十步失败或虚构终点；固定分母与全部状态分开报告。
这批只含两个已知物理根，动作/任务重复；第二个响应多样候选
并不是算法最终赢家。前缀动作及结构成本尚未计入等预算合同，
没有共同后缀机制干预、新根或强基线，不支持 P1 通过，也不能
把 13/13 未成功归结为理论不可行。下一步应检查预测与真实动作
后果的同时间偏差和任务评价，再设计实际计费的独立确认。

物理准备 SHA-256
`0aa3b133b22e58e1673fb546ef8a4a7679a8c17c5b652cae9fbabc6c012c2102`；
运行报告 `le-wm-main/results/p1_response_prefix_cp5_physical_20261031_RUN01/run_report.json`
SHA-256 `0b64f7f65faae3bb7c13aa4b62bb5a53713480c05146f8b61597b52fe93b4aaa`；
独立验收 `le-wm-main/results/p1_response_prefix_cp5_physical_20261031_consume_RUN01.json`
SHA-256 `7f7c67cef27a790e9d64eacde6d707c92d0cf0c027bd9901c6c0825794399f75`。
