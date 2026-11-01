# 独立对话：独立新根

解决历史来源排重和 freshness 认证，生成独立确认实验所需根清单；禁止按实验结果挑根。

主线程负责 P1 和代码审计，不重复其当前任务。不使用子智能体；仅写独占新文件，保留已有改动和封存结果。神经实验只用主机 CUDA，不降级 CPU；不训练冻结模型、不放宽科学门槛。

工作目录 `/home/gbk/code/LeWorldModel`；先读本目录 README.md。用个人有界goal推进“排重证据闭包及可认证清单”，不能以采集新目录冒充独立认证。独占新文件前缀 `chat_freshroots_20261027_`；仅在 docs/ 和 le-wm-main/{experiments/spectral_reachability,tests,innerdocs,results}/ 下创建该前缀文件，apply_patch编辑，不修改现有resolver、catalog、source或结果。

优先读：

- `/home/gbk/code/LeWorldModel/le-wm-main/innerdocs/moving_root_exclusion_manifest_20261027_v1.md`
- `/home/gbk/code/LeWorldModel/le-wm-main/innerdocs/root_exclusion_authority_resolution_20261027_v1.md`
- `/home/gbk/code/LeWorldModel/docs/p1_conditional_root_preconditions_20261027_v1.md`
- `/home/gbk/code/LeWorldModel/docs/D2P_PARALLEL_CUDA_PROGRESS_20261027.md` 中真实来源路径/SHA及身份图记录。

当前快照：已有618项inventory不等于coverage认证；历史reset上界8来源/16阻断尚未全部清除，当前两moving roots已知且不能算fresh。数字和状态必须在执行时重新核查。

要求：显式path+SHA追溯身份、seed/reset envelope与历史源码依据，不用当前实现反推旧reset上界；能解析的unknown实际解析，缺失明确保留。新抽样规则、固定候选顺序、失败分母、排除条件必须在读取新outcome前封存。生成机器可读root manifest及coverage证据；不能认证时输出具体未决依赖和证据请求，不伪造全部通过。实现和测试可审计消费者并实际跑metadata闭包，不重复做P1模型/控制实验。最终提供文件/SHA/测试、认证与未认证状态、可交主线程确认的候选清单，不按收益排名挑根。不得自动删除/推送。
