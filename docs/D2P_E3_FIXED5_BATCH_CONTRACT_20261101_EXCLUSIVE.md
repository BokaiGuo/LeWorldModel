# E3 fixed-five 批量执行合同

范围固定为 05、06、07、14、15，顺序不变；输入为
`e3_uncovered_fixed_inputs_20261101_exclusive` 的五个 RUN01 authority
及其脚本内固定的 SHA-256。既有 control5 来源不是新的物理反馈。
只使用单机合作锁，不声称设备全局独占或跨主机排他。

## 封存

在 `le-wm-main` 目录，以新目录名执行（不能复用或覆盖）：

```sh
env CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -B -m experiments.spectral_reachability.e3_fixed5_batch_20261101_exclusive \
  seal --batch "$PWD/results/e3_fixed5_batch_20261101_exclusive_RUN01"
```

封存逐格校验既有固定输入，在 CPU 上读取 checkpoint、构建模型并
记录启动闭包，生成派生 authority；不进行前向、CUDA 分配或物理执行。
最终 `seal.json` 绑定五格输入、闭包、派生 authority、Python 和批量
入口的哈希。保存命令输出中的 seal SHA-256，用于运行时独立比对。
封存中断会留下未完成目录，不得手工补写或覆盖；使用新目录重试。
闭包绑定实际代码字节，但不是整个环境、硬件或外部历史的完整证明。

## 执行与恢复

仅在 GPU 主线程明确让出资源后执行，须设置可见 CUDA 设备：

```sh
env CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -B -m experiments.spectral_reachability.e3_fixed5_batch_20261101_exclusive \
  run --batch "$PWD/results/e3_fixed5_batch_20261101_exclusive_RUN01" \
  --seal-sha256 SEAL_SHA256 --confirm E3_FIXED5_REAL_MODEL_PHYSICS
```

运行前验证 seal，持有既有 `e3_p2_host_cuda_cooperative.lock`，逐格
执行 CPU precheck，再启动真实 chunked identity host，并调用既有
保存态消费者。子进程继承锁，父进程意外退出时子进程仍持锁。
锁只约束合作方；不应与主线程或不遵守该锁的 GPU 作业并行。
原始输出、日志、intent 和结果均位于独立的 `cellXX_attemptNNN`；
单次超时为 1800 秒。没有增量续跑单次物理 episode：可恢复性是
格级检查点，新尝试从头运行且保留旧产物。

`completed20` 只表示原生保存消费者接受的完成状态；四种原生停止
状态均保留原名并立即暂停，不能当作 20 步或反馈边界已完成。
检查产物后如决定继续后续格，重新执行同一命令并附加
`--continue-after-stop`；这不会重跑已停止的格。失败、超时、父进程
中断或缺失结果会暂停；必须先人工确认旧子进程已退出、查看其
日志和原生保存产物以及是否已经发生物理执行，再按需显式加
`--retry-cell 5`（替换为对应格）创建全新尝试。禁止在仍运行的旧
子进程旁重试。其他失败不会因该选项被自动跳过。恢复时重新验证
已完成或早停产物的 report/completion 哈希及原生消费者身份联结。

无论终态如何，`scientific_admission=false`；这不是完整严格封存、
独立泛化、公平预算对照或效果证明。模型、物理轨迹和科学结论必须
依实际运行产物另行审计。
