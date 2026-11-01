# P3 六槽构造所有权与 journal 接线

本次只新增独占代码和 CPU 测试，未修改旧模块，也未运行 GPU、模型、环境或物理生产。使用者可将 `p3_six_owned_constructor_20261030_v1.module()` 注入现有六槽 `Factory` 的 `meter_module`，并传入新的 `p3_six_durable_transport_20261030_v1.Store`；没有新增生产入口或授权。

构造适配器沿用原生 meter 的 constructor intent/return/failure 及 close cleanup。底层 constructor 返回后，适配器先持有句柄，再落计量 return 和构造包装器；这两步若抛异常，即使 journal 写入失败也调用 meter.cleanup 尝试关闭句柄，保留原异常并附清理错误。实际 constructor 自身在返回句柄之前分配后抛错时，外层仍不能取得句柄，无法保证回收；现有六槽 factory 会阻止后续槽。关闭本身失败也不能被称为回收成功。

独占 store 使用调用者显式指定的逐类别、总容量、终态预留、磁盘余量和 wall 上限，预先检查配额，每次记录写入、flush、fsync 和目录 fsync；异常后因磁盘与内存状态可能分歧而锁死，不继续写入。上限值仅是工程参数，尚未对真实六槽工作量审读或授权。旧四槽 publish/final/frame 路径被拒绝；新 store 尚未提供六槽终态报告和 completion 的原子发布，也未实现重启续写。

只读 `inspect` 检查外部给定文件哈希、完整记录链及 intent/return/failure，保留并报告最后不完整记录的字节数及哈希，不截断、不修复，pending 和失败不升级为完整。完整链仅是可读记录结构，不能证明异常时的 fsync 成功、记录来源、数据完整入账或物理执行。数值、图片、编码文件清查、六槽终态协调、独立原生 strict consumer 和部分执行的系统级恢复均仍缺失；journal 传输成功不是科学结果，也不是生产准备就绪。

源码合同通过新模块包装原 factory 源码合同，追加本次两个实现文件的哈希；不是事前注册或可信时间戳。原科学门不变，已知根不构成 fresh replication。

验证：禁用 CUDA 可见设备后，联合新测试、六槽 factory/准备与三策略 callback 测试，44 项通过；另只读重建并验证扩展源码合同。fixture 不能证明真实设备、文件系统故障或物理行为。
