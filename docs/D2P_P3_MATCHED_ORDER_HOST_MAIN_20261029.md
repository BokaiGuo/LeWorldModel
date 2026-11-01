# P3 匹配计数调度的主机运行

固定两个已知根、每根 fixed-order 和 signal-order 共四槽已在主机 CUDA 完成，report SHA256 `8b5d1f618143557bb8a31ef3b924681ed6b54d7dfd52681d5bd300bafd0dc560`。独立 saved consumer 完成，receipt SHA256 `9da648ebf1c511bba53f06d73cfb4ce2e8fb075bb3ed6ff0f41af18177967bb0`，两对的已记录 27 轴及 hook 形状多重集均匹配。

该模型预测 parent 的顺序对照不是真实反馈闭环；相同已记录计数与形状不保证等 FLOPs、墙钟或总计算，也不能排除未 hook 调用。`scientific_pass=false`。完整预定四槽真实物理接线正在准备，原 P1 门未更改。
