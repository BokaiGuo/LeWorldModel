# P2 完整20维基线：已封存动作的真实执行

新版本从`p2_full20_sampling_20261027_v2`读取八条选择见证，验证四根、每臂40个
候选、80规划+4复核的真实predict收据、全部源封存、模型/目标/历史身份。
动作是两个独立5×2包，不能再压缩为两个重复二维命令。
实现复用已封存的durable逐步物理执行与RGB guard，不改旧版。

先持久化物理分支、每步attempt与返回、RGB和中止状态，再做终态编码。
神经编码仅主机CUDA，predict被禁止；物理模拟器本身不需要神经网络，不能
把模拟器在CPU运行混为“神经实验降级CPU”。
新版本明确只编码完成十步的终态；guard/termination的提前端点单独保留，不能
替代固定H2终态。若不全，原`encodings_complete`保持false，报告退出非零。
执行器不增加规划、不按结果重选、不开放P1/P2。总资源匹配与方法有效性均不主张。

从`le-wm-main`启动，输出必须不存在：

```bash
SDL_VIDEODRIVER=dummy .venv/bin/python -m experiments.spectral_reachability.p2_full20_physical_20261027_v2 --planning results/p2_full20_sampling_20261027_v2_cuda_v1 --output results/p2_full20_physical_20261027_v2_cuda_v1 --encode-terminals
```

同命令加`--preflight-only`只核对保存见证，不执行动作或神经计算。
单元测试采用非神经fixture，只证明持久化/guard/固定终态语义，不证明任务收益。
