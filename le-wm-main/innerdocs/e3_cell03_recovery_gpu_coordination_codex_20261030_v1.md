# cell03 恢复和 GPU 协调：未启动的工程方案

原 MAIN01 claim、partial report/completion/journal 原样保留。新尝试应使用
`CELL03_RECOVERY01_PROPOSED` 草案所列独立路径，不覆盖、不清除旧 claim。
恢复决策须外部明确记录并在执行前固定；本文件不声称已审批具体 manifest。
先调用既有 recovery acceptance 核验精确失败，再固定新 manifest、全来源
闭包、child 启动字节、环境和外部可信 SHA；运行时重验，不自动重封。
保留原 runner 的资源上限、source guard、before/after census 与独立 saved
consumer。该次仅是已知根探索，不是独立确认。

新组件 `e3_p2_cooperative_gpu_lock_codex_20261030_v1.gpu_lock(path)` 是
单主机、非阻塞的 advisory flock 原语。E3 和 P2 必须事先明确同一个
本地、可信目录下的固定常驻普通文件，并在 GPU 初始化前获取，直到
子进程退出和清理完毕才释放。锁文件必须预先创建、永不 unlink/替换；
使用相同设备的所有相关参与者都要接入，旧入口和其他主机不会被锁
约束。文件替换检查只能覆盖获取时窗口，不能阻止有权限者在持锁
期间替换；运维必须保护目录与文件，不能据此声称硬件独占。
组件不创建 claim、不启动子进程、不声明 P2 已接入、不提供运行许可。

在共同锁路径与 P2 接入/操作协调、经审阅的完整固定 manifest 及其
独立 SHA、真实模型后 census 与资源条件均未核验前，不启动恢复。
原先 claim 和 partial 仍是前驱证据，不得用新 claim 覆盖它们。新
launcher 应先验证前驱、清晰授权、源码和设备条件，再持共同锁并
独占创建新 claim；任何失败保留证据。锁是合作门而不是 source
闭包或科学 gate 的替代。
