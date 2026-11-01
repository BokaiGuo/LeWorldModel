# Parent tree saved-only summary

仅本版本source/tests/doc；全部既有parent adapter/CUDA源码与结果保持冻结。使用api-and-interface-design的边界契约：显式SHA输入、完整consume schema、固定分母、unknown不补零。不是生产runner、fullstrict或神经/物理复跑，也不是统计显著性/理论不可能性分析。

## 输入与权限

MAIN已提供的真实输入作为CLI默认值（仅绑定，不在worker读取生产内容）：

- report：`results/agent_p3_parent_native_cuda_20261029_v1_cuda_MAIN01/report.json`，SHA `75f4c162aaa75223dfd8523381e7614c8a35637afd52278df8a80d6affa34e40`。
- 完整已落盘consume：`results/agent_p3_parent_native_cuda_20261029_v1_consume_MAIN01.json`，SHA `622d1cdaf8d1ed69c4e78f8d1f5c7c6114285db6121f0021fbbaf6573a8a76ad`。

工具只读取这两个显式JSON及本工具/冻结producer三件套源码文件作SHA绑定；不打开report引用的prepare、journal、weights、root或任何其他artifact，不import producer/parent、torch、numpy或physics。外部SHA证明选定字节身份，不自行签署其authority；完整strict的数值与模型证明仍来自MAIN提供的上游receipt。读前读后核SHA、有界普通文件、拒绝symlink/重复JSON key/NaN。report cap256MiB、consume cap128MiB；新summary cap32MiB、disk floor200MiB。

`build(report, report_sha, consume, consume_sha)`核完整own schema、外部reportSHA与receipt.report_sha256、相同prepared literal/prepareSHA、四槽次序、slot statuses、每个有结果arm的完整receipt资源与原report逐值相同及原false flags。不会调用build/verify/run/consume等任何上游函数，不复算SVD、choose、梯度或NN，不重复fullstrict。新summary.verify只重新读取和汇总这两个已封存JSON。

## 输出字段及分母

固定顺序root0 fixed/adaptive、root1 fixed/adaptive四槽；每槽原144 expansion slots，总576。每个完整runtime必须实际保留144行；拒绝“只保留完成行”的输入。blocked arm无runtime时显式observed_job_rows=0、unobserved_expansion_slots=144，宽/深/H/J/resources/refusal/cap为null，而非伪造144个not_requested或0费用；plan_available=false只指没有可供本summary引用的plan。

- 宽：逐queue完整原decision、requested_width、实际allocated_jobs、returned_candidates、job status counts；allocated width不是全部成功执行的候选宽度。R8保留8个refused请求，不变为progress4。
- 深：原parent depth、decision.program_depth_limit、realized_progress_depth、各返回node depth counts、原child stop数。非root composition标签及extension_type原样保留，不能冒称任意composition/R生成器。
- H：每queue H、每条返回extension的H、每个已返回signal mapping的H1/H3。signal map计数与paid J attempted/completed分栏，失败前已付但未返回的J不伪造mapping；H不等于物理执行长度或共同预测终点。
- parent/J identity：完整root authority SHA/binding、每个mapping index、full3latent/2token context SHA、parent-task/native-binding/outer-binding/producer/mapping SHA、J形状与nominal/J的semantic JSON SHA。核原root→parent仅context binding重基、native request/response及显式adapter outward映射。child与root J不强加数值必然不同的条件；相同J不是漏接的充分证明，不同J也不自行证明正确神经计算。
- resources：原caps/attempted/completed的全部27轴、原实测wall/inclusive wall/unknown；操作event数量/类型/status计数。原详细operation events不重复展开以保持summary紧凑，输入report/consume SHA仍绑定原完整events。保留真实失败尝试，不退款，不把cap当actual。scalar_backward/独立AD_primal/J元素/FLOPs仍null；无equalFLOPs签名。
- refusal/cap/plan：R/cap节点、真实child stops、完整errors、diagnostic_best_available与plan_available严格分开；只有原完成状态允许plan。计划只汇总原模型Q、depth/indices/raw长度/动作与终态context SHA，绝不标物理成功率。先progress后child要求R时，保留此前候选和diagnostic best，但plan=false；此为unsupported工程停点，不是R/P1/C3理论不可能。

汇总resource_totals只在四槽资源均已观察时给all_four_total；有blocked arm时保留known_sum、observed_arms和all_four_total=null，不把未知费用填0。历史获取按原receipt独立携带，不与本次新增调用混算，不把本次0physics写成从未做过物理。所有P1/P2/P3/fullpolicy/fullC3/R/fresh/scientific flags保持false。

## MAIN命令

仓库根目录`/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
.venv/bin/python -m pytest -q tests/test_agent_p3_parent_tree_summary_20261029_v1.py
.venv/bin/python -m experiments.spectral_reachability.agent_p3_parent_tree_summary_20261029_v1 analyze --output results/agent_p3_parent_tree_summary_20261029_v1_MAIN01.json
.venv/bin/python -m experiments.spectral_reachability.agent_p3_parent_tree_summary_20261029_v1 verify --output results/agent_p3_parent_tree_summary_20261029_v1_MAIN01.json --summary-sha ACTUAL_SUMMARY_SHA
```

也可显式提供`--report PATH --report-sha SHA --consume PATH --consume-sha SHA`；完整receipt必须绑定该report，不能只给状态摘要或旧schema。analyze仅写新独占stem结果且不覆盖，verify不写文件。CLI帮助和全部路径均不触发producer/fullstrict。

## 实测与未完成

首轮25项RAM fixture通过40.67s；最终27 passed in39.14s，命令即上方独占测试文件。测试只调用已冻结纯核心numpy callback构造metadata（无NN/物理），不调用生产runner或上游consume；另有全四槽/144分母、root/child身份、R8、cap分支shape、blocked unknown、资源/plan/schema/外部SHA/重复JSON/事件外路径不读取等负向回归。cap和child-R专用schema fixture不声称重新验证原policy数值选择，原fullstrict仍由绑定receipt承担。

worker未读取生产report/receipt、未运行生产analyze/verify/fullstrict；实际summary结果待MAIN执行。MAIN报告的真实运行信息不被硬编码为summary输出或按它选配置；代码从SHA绑定输入产生描述值，无未来结果猜测、无新科学结论。
