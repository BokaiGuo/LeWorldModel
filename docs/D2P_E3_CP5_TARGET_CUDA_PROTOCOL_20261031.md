# E3 已观测 cp5 目标：CUDA 编码接线探索

固定使用两根保存的 control5 图像及其原始五步动作见证，重新以
冻结 PushT checkpoint 在主机 CUDA 编码，比较同一图像此前保存的
cp5 latent。原图 file/pixel hash、两 bank 身份、来源 resolver、
checkpoint 和编码源码均需绑定；记录完整差异，不用物理状态选择
动作，不训练模型。两个目标初始不满足原生成功，但来自已知轨迹，
所以只是后续开发目标，不是独立多任务评测或原 G1 的替代。

两张图片、同一个 checkpoint、固定预处理和 FP32 MATH attention；
报告编码值差异及模型前后哈希。若已有 latent 和重编码不吻合，
先查设备、图像变换、dtype、编码/投影和历史绑定，不能直接当
模型负结果。仅本实验不包含计划生成、额外的真实闭环或公平成本。
