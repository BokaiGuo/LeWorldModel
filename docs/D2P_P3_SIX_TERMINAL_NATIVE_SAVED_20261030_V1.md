# P3 六槽终态与原生保存记录核查

本次仅新增独立代码与 CPU fixture 测试，无 GPU、模型或物理生产，不修改既有合同或科学门。

`p3_six_terminal_20261030_v1` 将现有独占 store 的 journal 快照、固定六槽实际返回与未结项、pending、截断尾部、逐文件大小和 SHA、预算和 store 状态写入专用 report；report fsync 后才独占写 completion 并 fsync。两文件均计入终态预留及整体空间上限；失败时不返回成功且不覆盖或重试。构造及调用端应停止并保留未完成文件。完成标志固定为 false：终态发布成功不是六槽实验或原生 strict 完成。source contract 追加本次两个代码哈希，不是可信时间戳或预注册。

`p3_six_native_saved_20261030_v1` 接收外部提供的 report/completion SHA，重新检查终态绑定、文件清单、journal 哈希链与六槽前缀；对原生 meter 的意图、返回、失败、嵌套、上限、episode 和 stop 后派发进行核查。引用的 PNG/NPY 核对字节、格式及像素或数组元数据；未引用文件列为 orphan，不当成结果补齐。若六槽均有终态，会调用既有六槽 callback 记录消费者；完整返回且清理成功的槽进一步核对 20 个动作与 callback packet、32 个观测、四个 encoder/projector 生命周期及同帧编码。

**仍未完成完整原生 saved strict**：缺少独立认证源码和任务/根/goal/模型/精度、从真实编码重建全部上下文、反转策略的全部 signal/bank/Q 数学及 native model hook/资源 census、跨槽资源预算与完整存储结算。部分 native 轨迹只核已记录字段，未观测到的操作不能据此断言不存在。`physical_certified`、`full_saved_strict` 和 `scientific_pass` 固定为 false；终态和输入哈希不是来源真实性、设备执行或 fsync 可验证证明。orphan、截断、pending、失败和没有数据的槽保持可见，不得升级。实际 HOST driver、协调授权和生产前独立审读仍未完成。

CPU 测试覆盖终态和产物篡改、动作与帧绑定篡改、原生计量身份及 stop 后派发拒绝、未结槽/截断保留、储备不足和 completion fsync 失败。完整槽绑定单元测试替身绕过了 encoder 内容检查，不能替代真实来源的端到端验证。

禁用 CUDA 可见设备，联合运行新增测试及既有六槽/三策略测试：52 项通过。只读构建扩展源码合同，仍标记 `host_runner_ready=false`、`full_saved_strict=false`。
