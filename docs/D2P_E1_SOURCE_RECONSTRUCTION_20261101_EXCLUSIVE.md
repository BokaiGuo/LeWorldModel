# E1 原始来源追查与 arbitrary 重建

只读已保存来源，CPU 重算，未运行模型、物理或 GPU；仅新增本报告、独占脚本、
测试和结果。项目处于大量已有改动的工作树，本次不改动其他文件或原 gate。

## 新的可验证结果

`agent_experiment1_common_controller_20261028_v1.py` 的 `arbitrary()` 使用原
task 的 root latent、goal，固定 PCG64 种子生成 20 个候选，按 goal 均方距离及
index 选出目标。该函数没有模型调用，但 CPU 随机采样、向量运算及输入生成的
资源不是零。脚本校验旧 prepare、新 six-slot prepare、旧 prepare 绑定的实际
producer 源码和两份 gzip task 的固定 SHA，独立重算 draws、scale、20 candidates、
costs 和 winner 的完整序列化摘要，再逐位校验新六槽中的 float32 target。
两根均吻合；root0 winner index 13，root1 index 2。两次完整重建摘要分别为
`f2e9a959e623570424d9f12711bd0589b497106af9da3590fc28b05359badce2`、
`9dfd23f008e18f4d36a21927f8de8fb35a8d94985fa1d70671e87960bdb7f369`。

这是对已封存目标生成关系的真实 CPU 重建，而非历史实际执行的写前事件、完整
费用回执或对原始 root/goal 编码链的重放。数值重现依赖本机 NumPy 实现；如果
跨版本发生变化，脚本 fail closed。没有据此写入 proposal receipt 或将任何费用
标零。

结果 `le-wm-main/results/codex_e1_arbitrary_reconstruction_20261101_exclusive.json`
SHA256 `42f0745e9666e2d9d8662f594eee30b5fcac0ae6dd1eff8a04ee82ec04330fc7`。
输出排他创建，不能覆盖已有文件。

## 原始事件来源与精确缺口

* Arbitrary：旧 prepare 的完整候选摘要和固定任务可重算，producer 的
  `target_summaries` 调用 `arbitrary`；已检查的保存链没有将随机候选过程、
  失败/重试与历史 CPU 费用保存为独立事件 receipt。`root_context` 和 `goal`
  是已编码输入，来源计算、与其他臂的共享分摊仍未闭合。无法从一个确定性重算
  倒推出历史尝试次数或实际费用。
* Learned：`agent_hilewm_rgb_high_low_20261028_v1.py` 中 `Trace.call` 和模块
  pre/post hooks 将记录追加到内存中的 `row['events']`，`finally` 才写 report；
  现有 report 的两条 off 行及先前 receipt 可对应 FIRST 与 E1 目标。`high_search`
  在 FIRST 后返回，随后同一行执行 low cost，因此整行包含非 FIRST 所需的后续
  工作；不能直接作为 proposal 账。事件在内存中并不构成持久写前日志，已保存
  成功行也无法排除进程崩溃前未写出的尝试或其他运行；更早输入、模型构建和共享
  成本亦无已认证的分摊。不能仅凭行内 hook 分配完整费用或签 receipt。
* Compiled：已有独立 gzip journal 记录 `root_started`、dispatch attempt/return、
  `plan_sealed`，并与原报告和 E1 目标绑定；原消费程序还按 search schedule
  检查 journal。该范围是 beam 的 search/replay，不含此前生成 root/goal 等全部
  祖先计算，也未完成与 controller、replay 间的事件去重和共享分摊；因此 journal
  不足以充当整个 inherited phase receipt。

以上“未找到”仅针对本次追踪的对应生产代码和固定保存链，不是对所有其他项目、
机器或历史运行的穷尽搜索。要真正关闭端到端账，需取得历史运行的完整持久事件和
全祖先/共享资源分配证据，或者在事先定义成本边界的新的端到端运行中持久记录
attempt、return、failure、共享资源和跨阶段事件身份；后者是新实验，不能反证
旧运行的历史成本。六槽依旧 incomplete，等预算、臂排序、科学准入均未成立。

验证：在 `le-wm-main` 下设置 `PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''`
及 `OPENBLAS_NUM_THREADS=1`，用 `.venv/bin/python -B -m pytest -q -p no:cacheprovider`
运行新测试、已有 Hi trace 和 compiled ancestor 三个测试文件：31 passed。包含
真实来源、篡改 task/目标/摘要/身份和排他输出保护；没有运行新的神经或物理实验。
