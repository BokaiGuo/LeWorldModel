# E3 cell02 操作预算与跨入口去重核验

本工具只读固定的 cell02 authority 和 control5 prepare，核验其固定 SHA、
cell02 K3/Dxy/root/边界、16 格分母，以及 physical meter、数值预算、
feedback node、模型 source guard 等实际源码在原 authority / prepare
绑定中存在且当前文件 SHA 一致。manifest 必须由调用者提供独立固定
SHA，且其 cell、authority、claim、output、candidate 父源码与内嵌
precheck/child/audit 代码摘要要与当前固定入口一致。它不重建最终
环境 draft；仍须单独使用最终环境审核器并由主线程批准。

原 launcher manifest **没有直接列出** physical 和数值 cap；关联链是
manifest 固定 authority SHA → authority 固定 prepare SHA 和源文件
bindings → 当前实现。不可声称这些数字直接封存在 launcher manifest。
检查器及其传递依赖本身未纳入旧 authority，需单独审阅并固定摘要；
源码时点文件哈希也不是同字节 import 或进程内存证明。

已核验的 physical meter predispatch 上限：constructor 1、close 1、
reset 2、set_state 4、goal_setter 3、goal_render 2、goal_bind 1、
public_step 30、physics_step 304、explicit_render 32。失败尝试会计入；
cleanup 实现存在为关闭资源而尝试调用的例外，不能将日志写入失败
解释成未发生 cleanup。

K3 单个成功数值节点的实现计划为 candidate forwards 33、replays 3、
AD forwards 32、predict 204、reverse 32；后续 boundary 10/15 两个
节点若均完整则计划 408 predict、64 reverse。先前 control5 节点
属于单独的已准备上游，不应当作本次 online 新增调用。原 prepare
的三节点 transport envelope 有 612 packet、19,968 arrays、10,560
events；它覆盖三个节点的预算包络，不是本次 online 实际调用数，
也不是独立证明逐次 predispatch numerical cap。早停、失败的实际
消耗只能从真实记录确认。source guard 的 GPU peak 检查阈值为已分配
3 GiB、已保留 4 GiB，并非启动前可用显存或可容纳模型的证明。

去重只使用 results 目录中**明确拼写 cell02 的名称**；不打开、不
stat 这些条目，跳过名称含 cell03 的条目。固定 cell02 claim/output
和未分类同名条目作为潜在冲突；已知 cell02 上游 control5 与 prepared
authority 不误记成 online attempt。两次名称观察不一致报错；存在
潜在冲突时 CLI 输出结果但返回非零。名称清空也**不证明跨入口唯一**：
旧入口可用任意 output，其他目录、机器、进程及并发启动均不在范围。
审核与启动之间无共享原子锁，须由主线程协调全入口、核对作业和
进行授权；不能清理现有 claim，也不能重试已有失败。

一次只读现场名称观测中，显式 cell02 名称仅发现上游 control5 目录、
其 receipt 和 prepared authority，未发现固定候选 claim/output 或
其他显式 cell02 名称；这是时点且名称范围有限的观测，不是去重证明。
没有检查 cell03 内容或状态。

使用时在最终环境先完成原 final review，再以已审阅 manifest 的文件
字节 SHA 执行：

```sh
.venv/bin/python -B -s -m experiments.spectral_reachability.e3_cell02_caps_dedup_codex_20261030_v1 --manifest /home/gbk/code/LeWorldModel/le-wm-main/deployment_manifests_e3_cell02/e3_cell02_manifest_REVIEW01.json --manifest-sha256 REVIEWED_EXTERNAL_SHA256
```

SHA 占位符须替换；结果 JSON 应独立保存并审阅。生产环境、GPU、
存储资源、native renderer、运行授权、输出及 saved 检查仍为独立
阻断。本工具不启动 CUDA、模型 forward、物理或生产，不改变任何
封存或科学门槛，也不生成最终生产 manifest。

验证：新检查器、cell02 saved acceptance、原数值 HOST 与 native meter
共 **69 passed**。测试包括真实 cell02 authority/prepare 和实际 CPU
draft 的绑定、篡改与名称冲突/漂移；运行时的目录及 manifest 使用
临时 fixture，不能替代最终环境审核或生产证据。
