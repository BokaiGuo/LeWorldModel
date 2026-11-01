# P1 同尺度固定探针 MAIN02：画布守卫中断

MAIN02 按固定 16 × 21 计划串行执行，anchor 0–4 报告完成，保存分片已由 `check_shard` 独立核验各 21 个完整端点。anchor 5 在第 17 个后缀中断，后四个后缀和后续锚点未执行；不能运行要求全部分片的最终消费者，也不能将本次称为完整 336 分支结果。

该锚点为 root 0 / on / projected_ascent / model_selected。probe index 16 `selected_-1.00` 的第五个物理步骤已返回；日志所记 state 的推杆位置为 `(445.35822639922776, 528.5548847416848)`，`terminated=false`、`truncated=false`。之后新鲜 RGB 采集触发 `fresh RGB/canvas guard`。对应守卫要求 agent 和 block 的四个位置坐标均在 `[0,512]` 且推杆的标准蓝色像素数非零；原始首帧复算该像素数为 0，因此越界和不可见两个条件均触发。

这是计划内较大反向后缀造成的越界状态与现有协议不兼容，不是模型预测、理论或 P1 gate 的负结果。环境未原生终止也不等于该画布外观测有效。现有 runner 将其标为工程失败并停止；已返回状态不替代通过图像和完整性合同的五步端点。保留完整失败现场，不覆盖、不拼接为完整样本。

- [anchor 5 报告](../le-wm-main/results/p1_matched_suffix_scale_20261030_run_MAIN02/anchor_05/report.json) SHA256 `73a05677016081d3feee49ce863f7f88dc17b7a531a77e80d0b30d18aba482de`。
- [事件日志](../le-wm-main/results/p1_matched_suffix_scale_20261030_run_MAIN02/anchor_05/journal.jsonl) SHA256 `e90d38cfad7514041683189f2dcf2c86b93c1d49fa312ff8c668ebf537eb5ed9`；第五步 `returned_transition` 为事件 11854。

新增 [只读复核](../le-wm-main/experiments/spectral_reachability/p1_canvas_incident_audit_20261030_main.py)：对固定 prepare/report/journal 做哈希、日志链和全部五步动作字节校验，核对返回状态与物理快照及分支关闭顺序。第 5 步的第一张 off 图像已保存且文件、像素哈希通过；该图像的保存不意味着 on 图像、完整三帧或该分支端点合格。实际复核及 5 项测试通过；测试包括动作、状态、顺序和错误信息篡改。该复核依赖保存的生产日志，不是新物理执行或独立硬件测量。

后续协议应事先定义并记录越界删失、保留固定分母、拒绝虚构 RGB 和补值，同时对其他异常继续 fail-closed；观察本次失败后制定的新运行只能作为探索，不能冒充预注册独立确认。新的执行器和消费器必须先验证后再启动，不能仅放宽画布守卫。

## 删失协议后续运行

新探索协议固定全部 16 × 21 分支，不改变原画布或 RGB 守卫；只有原守卫确实拒绝、返回状态证明坐标越界、且动作、分支、时点和原帧均匹配时记为删失，保留分母并继续其他分支。其他异常仍停止；删失端点不补值。

主线程审计发现新 runner 最初会在 `partial_or_failed` 时仍退出 0；修复后将失败报告保留并退出 1。修复后固定准备文件 [MAIN03](../le-wm-main/results/p1_matched_censored_runner_20261030_codex_prepare_MAIN03.json) SHA256 `f1a6cf4f57647a9379f0f8223bf7a1d764094b7f566ac0754330d14e8b1dac0f`；主线程复跑相关测试 21 项通过。此前 MAIN01/MAIN02 prepare 随源码修复而失效，予以保留但不能用于运行。

2026-10-30 已启动独占目录 `p1_matched_censored_runner_20261030_codex_run_MAIN01` 的真实物理作业；这是非神经模拟，没有 CPU 神经推理。运行尚未完成，须等待全部分片和 run report，固定其哈希后再执行 saved consumer。启动或测试通过都不构成能力收益、独立确认或 P1 admission。
