# P3 独立进程全流程计量合同

固定顺序 20 uniform/signal/reverse、30 uniform/signal/reverse，复用固定在线预算内核的已知根与无结果输入候选流；每个槽在独立新进程中持同一固定 inode 合作锁，分别建立 CUDA、从封存权重加载模型、计算该策略的信号和候选，不训练、不运行物理、不事后选点、失败不重试。信号和反向各轴照原合同分项限额；不浪费计算以填平实际计数。新运行是测量探索，不是预注册或新根。

每槽外部父进程单调时钟覆盖启动子进程至退出，包括 Python 进口、来源检查、锁获取、子进程、模型与 journal；不包括父进程事先准备及后续独立消费。子进程记录 CUDA preflight、同步模型加载 wall，加载期间 `/proc/self/io` 的进程级 rchar/read_bytes 差值，算法整体同步 wall、峰值 CUDA allocated/reserved。journal 对每个事件附单调时钟，nominal_request 至第二个 signal_complete 是含 journal 的在线结构工作区间；allocation_frozen 至 arm_complete 是含 journal 的搜索区间；uniform 信号为 0，结构信号未执行。区间包含 fsync 和测量开销，各阶段不是设备 kernel 时间，不能与嵌套轴累加当 FLOPs。文件系统 page cache 冷热不受控，IO 进程计数不等于单个权重文件的实际读取量；动态源码闭包、能源、FLOPs 和 CUDA 上下文隐含成本 unknown。

公平合同只检验：相同预算点每个策略使用相同的各资源上限（单臂子进程 wall cap 1800 秒；GPU allocated 3 GiB/reserved 4 GiB；四类前向各 48 或 68；Jacobian 2、SVD 2、VJP/grad API/raw hook 各 384；metric 24 或 34），不要求或者宣称实际耗费、逐根搜索或全流程 wall 相等。外部完整进程 wall 有实际记录但没有强行假设 1800 秒外部上限。uniform 实际 AD 为零，signal/reverse 各实际付费。独立进程测量不消除顺序、缓存、热态或其他非合作进程影响。科学 gate 不改变。
