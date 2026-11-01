# E1 六槽真实 RGB 编码：主线程结果

2026-10-29 MAIN 审读六槽桥、测试与上游合同；桥与既有编码组件联合 49 项测试通过。真实 physical strict 重新重建、prepare SHA `b58c5eaea94ea7ecdcb5b6c07fdb6b535643f252d705c73938c843840b3e5159` 及 verify 完成。

主机 CUDA 执行 18/18 个实际检查帧，report SHA `7de6796f87878bb862cb2f27781412649bb99ce6310a8c7de339595d29491405`，saved consume SHA `86ee030e2b7a50e0412e41d9c16ae6a10c80a1b6beaf89831f04e943db2f33b5`；额外重新完整调用 consume 并与保存 receipt 比较通过。六个初始帧与原任务像素绑定，重编码 root latent 逐字节相同；预测使用原 raw10 whole replay，segmented context 分别保留。此次实际 whole/segmented 全等不作为其他运行的普遍保证。

新增 18 encoder、18 projector，0 predictor、AD 和物理；CUDA 峰值 allocated 110062592、reserved 115343360 bytes，through report fsync/hash 墙时 17.6511 秒。FLOPs 未知；两个已知根不是新根，也不等于各生成方案成本相同。

例如 root0 编译程序第5步对自身中间目标的模型 Q 为 0.05061，实际 RGB 编码 Q 为 0.25798；它是预测与实际的同目标对照，不可直接与另一个方法的不同中间目标 Q 排名。第10步相对共同任务目标，root0 编译/已学习先验的实际 Q 分别为 0.47634/0.41634，root1 为 0.28700/0.36061。原六槽物理任务均未成功；这些描述性 latent 数字不能替代真实任务成功或证明 E1 机制。

额外核验第一次使用相对路径触发既有 canonical-absolute-path 守卫；重试只改复核命令为绝对路径，不改源码、报告或门槛，随后完整重建一致。这个命令接线错误不是模型或理论负结果。
