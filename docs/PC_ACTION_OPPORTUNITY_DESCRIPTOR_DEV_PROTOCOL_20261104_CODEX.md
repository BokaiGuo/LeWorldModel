# Action-opportunity descriptor 开发协议（2026-11-04）

## 假设

对候选前缀 `p`，有限后缀作用描述

`d(p; U) = [C(p+0)-C(p+u) for u in U]`

可能比当前终距或局部 ridge value 更能预测后续可利用质量。这里的
“更能预测”只指开发集上对未用于构造描述的后缀和已封存 CEM 结果具有更好的
预先指定排序关系；不等于证明因果机制。

## 固定范围

- 数据：已封存的 16 个开发 root、spectral/random_direction/diverse 三个候选库；
- 候选：现有 `candidate_suffix_quality_posthoc` 中全部成功记录，不生成新候选；
- probe A：四个常值后缀，`(+x),(-x),(+y),(-y)`；
- probe B：四个不同时间结构后缀，交替/对角模式；B 不参与 A 描述；
- 评分：同一 exact predictor，zero tail 作为基准；
- 外部目标：候选后缀质量诊断中先前封存的 `best_score` 与 `success`；
- 禁止：oracle 前缀、物理 outcome、人工筛选、训练 encoder/adapter、改变正式搜索。

## 预定分析

对每个 arm 及总体报告：

1. A 描述最大改善与 B 描述最大改善的 Spearman 相关；
2. A 描述最大改善与封存 `best_score` 的 Spearman 相关；
3. A 描述 top-10% 候选的封存 `best_score`、success rate，与全体比较；
4. 与 `immediate_score` 的同样指标比较；
5. A/B probe 的实际调用数与 token 计量。

所有候选和失败分母保留。若没有预先定义的稳定效果，不启动 Stage-2 retention
对照；若出现阳性，也只能形成新版本假设。

## 证据边界

该实验使用同一冻结 exact predictor，因此不能认证视觉模型、真实物理响应或来源独立性。
封存 CEM 目标来自先前开发诊断，不能当作独立确认。结果只能回答：
“有限动作后果描述是否值得进入下一轮开发诊断”。
