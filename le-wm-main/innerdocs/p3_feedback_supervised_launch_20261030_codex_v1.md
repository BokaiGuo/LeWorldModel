# P3 四槽反馈对照：外部监督交接

选择现有 ONLINE_AUDIT02：两已知根，每根 fixed/signal 各一槽；signal 在真实执行反馈后继续规划。它是既有 RUN_MAIN01 的同协议重复，不是新负控或 fresh replication。固定 20 步末端计量工作向量对照，不把 cp5/10/15 的数值工作称为相等；同预算仅指合同中的已测轴，不是等 FLOPs、时间或能源。P1 未通过，P3 机制和科学通过均不成立。

此前 host 的 wall 检查不构成外部硬超时，cuda:0 不单独固定物理 GPU。本独占包装器通过物理 UUID 查询设备及空闲显存、禁用 MIG、设置唯一可见 GPU 后，以新进程组启动原 online binding；截止时发送组 SIGKILL，保留全部部分文件且不自动重试。只有退出码 0 不等于 strict 消费成功。异常写入是尽力而为，SIGKILL、不可中断内核等待、组外逃逸、断电或 I/O 故障不保证进程及记录已完整终止；应另外使用用户级 systemd cgroup 外层限制、人工检查进程及设备。显存查询仅是采样，不是预留。包装器不实现硬件证明或 saved 监督验收；原四槽 strict/预算消费者仍必须执行。

输入：计划 `results/p3_feedback_online_binding_20261030_v1_plan_ONLINE_AUDIT02.json` SHA `d43c5f20290114bb14c37e3c42a16f2b36e69eb5aceea764ded5750075d3b3d1`；prepare SHA `b6c8dcf650d4e752f6306ed4b6f9434861c224ced51c7c8de25c0e7ca6d323f2`；预算合同 SHA `5781d953d05477e633659b3e9bb58d630ea59447720964af4e8678829f565b4f`。包装器要求自身 SHA `e3b5bf9da85b7173c4b1ef7b79a6493443e757350bf187d8bd6ce3c225b2b00b` 由操作者传入，核现有 binding 源码及原闭包；启动前需要重新核哈希。计划、代码与结果必须经独立审查和可信外部时间戳后才可称事前固定；本地哈希、命令里的 reference 与授权字符串均不认证批准或先后。

主线程须核验主机用户服务、GPU 可见性/权限和无并发、真实可用显存及充足磁盘、显示/仿真资源、外层超时，以及计划内三条输出路径和独占监督目录均不存在。host 合同的 CUDA allocated/reserved caps 为 3/4 GiB，不等于实测全部 GPU 显存上界；外层显存阈值、时间、磁盘上限须主线程审定，不能凭以往耗时当保证。主线程检查现有 prepare 和所引用封存来源的授权范围。结果若失败或超时不能重试补槽；系统级停止后应检查所有子进程已退出并保留 partial。

在 `/home/gbk/code/LeWorldModel/le-wm-main`，以下仅供审核后由主线程执行，尖括号不是可直接执行参数：

```sh
systemd-run --user --unit=<unique-unit> --working-directory=/home/gbk/code/LeWorldModel/le-wm-main --property=Type=exec --property=RuntimeMaxSec=<outer-seconds>s --property=TimeoutStopSec=10s --property=KillMode=control-group --property=SendSIGKILL=yes /home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -m experiments.spectral_reachability.p3_feedback_supervised_launch_20261030_codex_v1 --authorization MAIN_THREAD_APPROVED_P3_MATCHED4_SUPERVISED --source-sha e3b5bf9da85b7173c4b1ef7b79a6493443e757350bf187d8bd6ce3c225b2b00b --gpu-uuid <GPU-uuid> --suffix <unique> --reference <reviewed-record-reference> --min-free-mib <free-mib> --min-total-mib <total-mib> --hard-seconds <child-seconds>
```

外层时间须覆盖包装器两次 CPU 源验证、GPU 查询、child 限时和收尾。需提前确认系统服务能获得预期设备、显示及环境变量；本包装器未在真实 systemd 或 GPU 端到端验证。完成且全部进程停止后，从原 run 的 report/completion 计算 SHA，以同一 reference 按 `D2P_P3_FEEDBACK_ONLINE_BINDING_20261030_V1.md` 的 `consume`/`inspect` 命令消费，但将过时 ONLINE_BIND01 计划和 SHA 替换为上面的 ONLINE_AUDIT02 和 SHA；不得重用旧 receipt。若要检查每个控制步预算，用 `p3_feedback_checkpoint_budget_20261030_v1` 并传入该 plan/reference，见其 `--help`。任何部分失败不提升为完整对照。
