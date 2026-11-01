# P3 主机 CUDA 设备身份检查失败

固定资源计划的 `p3-decision-host-20261030-codex-main01.service`
于 2026-10-30 启动，保留产物在
`le-wm-main/results/p3_decision_host_cuda_20261030_codex_RUN_MAIN01/`。
运行以 `runtime UUID` 失败，`failure.json` 记录
`native_pending=0`、`roster_returned=false`、`retry_authorized=false`；
native 目录为空。这不是调度性能或物理结果。

主机 CUDA 独立探针显示 `torch.cuda.get_device_properties(0).uuid`
是 `_CUuuid` 对象，其字符串表示不是设备 UUID；原 driver 直接比较
`str(...uuid)` 和计划中的 UUID，因此误拒。该对象的 `bytes` 为
`[78,137,115,255,2,120,83,243,43,174,38,21,212,97,228,65]`，
与预定设备 `GPU-4e8973ff-0278-53f3-2bae-2615d461e441` 的
16 字节一致；`nvidia-smi -L` 主机服务探针也看到了该 UUID。
这两个探针不构成完整实验验收。

修复应解析并严格比对设备 UUID，拒绝不支持的格式和不匹配，新增测试，
并以独立版本与新资源计划运行；不能覆盖失败产物、移除身份检查或
声称这次六槽完成。原计划禁止重试，因此后续如运行须使用新合同。

后续桥接原型为 `p3_cuda_uuid_bridge_20261030_codex_v2.py`：保留计划、
可见设备和 `nvidia-smi` 身份一致性检查，解析 16 字节并在后续属性读取时
再次检查。主线程独立运行该原型的测试文件，15 项通过；这些是 CPU 替身
测试，不是实际 CUDA 接线验收。该记录写入时新 driver 和资源计划仍在准备中。

更新：V2 driver 与 PREP02 新资源计划已绑定，主机只读预检显示指定
GPU 空闲 15763 MiB；随后启动独立服务
`p3-decision-host-v2-20261030-codex-main02`。启动时核验到服务活动、
3900 秒上限、control-group 终止、禁止重启，实际进程 cgroup 与服务一致。
这只是启动与监督证据，不代表 UUID 真实接线或六槽运行已经完成；旧
MAIN01 保持不变。该预检是瞬时快照，不是资源预留。

V2 运行随后写出 `uuid_binding_v2.json`，服务日志显示已进入冻结模型加载
和环境初始化，因而越过了 MAIN01 的首个 UUID 比较失败点。服务仍运行；
这不是六槽完成、独立保存验证或科学通过。

终态更新：V2 服务退出码 1，`end.json` 保留了六槽；两条 signal 槽
为 `callback_complete_unverified`，四条 fixed_h1/reverse_signal 槽为
`failed`，分别在执行两包或三包后因 `ValueError: unsupported decision`
停止。决策日志显示失败阶段的 schedule 调用本身已返回，但原实现先在
`decision_work` 的共同门检查决策，随后才保存完整 block；因此失败时
具体 decision 和 diagnostic 未被该决策日志保存。源码中的门要求
continue/progress/progress 和 horizon 1 或 3；不能仅凭错误消息确定
到底是哪项不满足。该预定的拒绝规则不能在原封存合同中事后放宽。
这是未完成的受限调度实验，不证明四个槽的控制效果或理论失败。
需以新版本在拒绝前保存完整决策，并为不支持的程序类型建立真实实现
或另行固定可解释的对照合同；不得将其偷偷映射为 progress。

进一步复算：从 native events 中按 slot 重组末尾两次完整 signal 返回
（分块顺序及每个事件 SHA 经 `p3_six_numeric_bridge.reconstruct` 检查），
用同一任务的保存数组和原 `spectra`、`runtime_signal`、`core.choose`
重新计算。四次拒绝均为 `program_type=extension_type=reconfiguration`，
`continue_expanding=true`、horizon 3、8 candidates：

| 槽 | χ | horizon gain |
| --- | ---: | ---: |
| 0、2 | 0.566575604 | 0.247371423 |
| 3、5 | 0.519917176 | 0.260371552 |

两组 χ 超过原定 0.5，触发的是未实现的 R 分支，而不是无法识别的 UUID
或对已执行动作的物理性能结论。同根对应槽的数值重复，不是四个独立样本。
native events 文件 SHA-256 为
`25845879aa6f206c03e51529d00b89411bfd817bd0d4b2d163b783b789e4a9c1`，
end 文件 SHA-256 为
`3cdf5f9acc94ae2a0cb764a5ca1eb1dabdfab632219551e52b75fc0b9b8ea51b`。
此为保存数组的事后复算，尚非完整 native 跨日志独立认证；不能补造
未执行的后续步骤。应实现真正的 R 分支并以新合同进行比较。
