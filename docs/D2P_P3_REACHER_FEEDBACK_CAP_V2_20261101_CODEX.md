# P3 Reacher feedback-cap v2：可复核性合同

这是对同一已知根的开发性重复，不是原 RUN_MAIN01 的事后修补或独立确认。沿用 v1 两臂、15 物理步、5 步包、种子、CEM、扩展判据和六轴每臂共同模型入口上限；实际调用量不保证相等。锁覆盖环境、模型及物理实验；不从物理几何取值影响模型选动作。

两臂各保存完整执行 root snapshot 及 root/goal/history/warmup 哈希，单臂一旦采集即独立落盘；两臂齐备再保存汇总并要求整个 snapshot 相等，同时沿用 RGB、历史和暖机动作运行时逐项相等检查。报告分别绑定哈希。单臂采集失败或根审计失败不填造快照，保留未开始行与错误。仅保存 RGB 哈希而非图像或暖机原动作，因此无法仅据保存文件重算 RGB/history 和动作内容的跨臂相等性。

每个实际返回 step 记录 qpos、target_qpos、逐元素原始差值、L2 和最大绝对差，不做角度 wrap。默认注册环境实际为 `task="hard"`、dm_control `Reacher`；goal-image wrapper 动态附加 `target_qpos`，并没有替换原生 task。该任务没有与此目标 qpos 绑定的原生成功判据，因此 native_success、native_qpos_threshold 和 native_termination 均为 null；几何不是成功门。仅在实际 task 为准确类型 `ReacherQPosMatchTask` 的情况下，才调用其 `get_termination` 并核验逐关节绝对差严格小于该 task 的阈值；本 runner 不修改 task 类型。

root 同样记录几何，终点为最后一个实际返回 step 的记录；提前终止、截断、错误不补步。若几何读取失败，已返回的动作和步数仍保留并标记失败，不能声称该步几何完整。原生定义、goal wrapper、环境 wrapper、runner 和依赖源纳入 protocol SHA；实际运行前后检测来源漂移。

所有两臂保留，未能启动或执行失败明确记录；全 15 步以外不冒充完整等时域。模型入口计费为调用前 attempted，不代表 GPU 内核耗时或实际算力等同。无独立根、重复种子或真实结果前，不主张比较结论，也不是 P1/P3 admission。

修复记录：原 v2 RUN_MAIN01 在两臂 root_geometry 因错误假定 task 类型而失败，均为零规划动作；原报告保留。修复不改变模型、规划、实际任务或预算；新运行必须使用新目录。

主机调用（需用户审查并自行授权）：

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
.venv/bin/python -m experiments.spectral_reachability.p3_reacher_feedback_cap_v2_20261101_codex --output /home/gbk/code/LeWorldModel/le-wm-main/results/p3_reacher_feedback_cap_v2_20261101_codex_RUN_MAIN02 --authorize-host-cuda-and-physical
```
