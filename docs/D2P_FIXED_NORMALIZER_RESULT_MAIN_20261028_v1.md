# P1 固定双统计量主机 CUDA：实测结果与排因边界

## 已完成的干预

本实验落实[运行前固定协议](D2P_FIXED_NORMALIZER_ABLATION_PROTOCOL_MAIN_20261028_v1.md)，
不改该协议、原封存或部署normalizer。MAIN完整审读新source/tests/doc，独立
42测试通过（23.81s），冻结后prepare、主机CUDA、完整strict均已实际完成。
另有MAIN独立saved复算工具及5测试通过（0.69s）。没有CPU神经fallback。
最终Hi-LeWM v3、真实high-CEM、归一化双臂及MAIN独立复算四文件联合回归
122 passed /40.07s，git diff --check通过；测试数量不是科学样本数。

原两个known PushT根×off/on条件、全部164探针、两h均保留；不是新独立确认，
不能把本批结论转授旧Reacher checkpoint或所有任务。原latent3/goal192/raw动作
不变，只比较原公开统计与预先指定的归档mean/sample_std。两段incoming history
和outgoing suffix均从raw重建一次。归档统计不是权重训练血统证书。

实际328predict/AE/predictor/pred_proj调用各328（嵌套轴，不相加为FLOPs），
0encoder/AD/physics/training，errors=[]、全模型参数/buffer前后精确不变。
wall69.486282727秒（含verify）、allocated峰106388992字节、reserved峰111149056
字节。公开分支**全部164个完整context及objective逐位复现**，不是只核baseline。

| 产物（共同前缀 `le-wm-main/results/agent_raw_normalizer_ablation_20261028_v1_`） | SHA256 |
|---|---|
| prepare_MAIN01.json | e35710fbb588aadadf9c99bafa1872292ed6508604163450b753c03d0744468d |
| cuda_MAIN01/report.json | c676d40938695ecbb539bcc0bc66b4d3ecc370040c88e1c374650d2009861e26 |
| consume_MAIN01.json | 95d7d183dd98a4edd7df0bcd8750fc8cede2270939da04c32e16aa661813a8d0 |

## 结果：这两套统计的差异不是充分修复

同一动作、同一真实终态和goal下比较模型native float32目标差分与实际编码
float64目标差分。独立工具直接解码原loss blob并核dtype/shape/hash，重算真实Q；
不依赖producer汇总数。consumer另保留实际NumPy float32目标与全部192维secant。

| 统计量 | 同向 | 反向 | 真实目标tie | 平均终态192维MSE |
|---|---:|---:|---:|---:|
| public | 22 | 34 | 24 | 0.187245976747 |
| archive_sample_stats | 20 | 36 | 24 | 0.186779245014 |

80对全部保留，24个真实零响应不计同向。逐对迁移为：32反向持续、2反向变同向、
4同向变反向、18同向持续、24零响应持续。平均终态误差约降0.25%，却新增净2个
反转；不能把终态MSE略降误写成排序修复或实际任务收益。

| root/renderer | h | public反向/10 | archive反向/10 | 真实tie/10 |
|---|---:|---:|---:|---:|
| 0/off | .01 | 9 | 9 | 1 |
| 0/off | .001 | 0 | 0 | 10 |
| 0/on | .01 | 6 | 5 | 2 |
| 0/on | .001 | 5 | 4 | 5 |
| 1/off | .01 | 5 | 6 | 0 |
| 1/off | .001 | 6 | 7 | 3 |
| 1/on | .01 | 3 | 4 | 0 |
| 1/on | .001 | 0 | 1 | 3 |

四baseline终态MSE，public为0.281968457/0.263282632/0.113830020/0.086395903，
archive为0.282123893/0.263307986/0.115899338/0.082441629。两h、四条件不挑选。
不是“归一化无影响”：同一终态预测坐标最大变化0.0822626054；只是该预定替换
没有解决固定面板的问题，不授权替换部署统计。

MAIN独立结果：`le-wm-main/results/main_normalizer_contrast_20261028_v1_MAIN01.json`，
SHA `664540c0ceb34ae8a98ecd1e02ea7c75da7e4012f50f8253ff5966636f7ca015`。
工具 `main_normalizer_contrast_20261028_v1.py` 绑定已由MAIN全量消费的原文件字节，
不冒称自身再次递归认证全部上游。另列独立float64模型Q分栏，避免混用规约精度。
该分栏在两臂上逐对标签与native loss分栏全部相同；本固定面板不支持把这34/36
反转解释为目标平方和的float32规约舍入。不能外推所有数值kernel均无问题。

## 代码修改与尚未解决的因素

新normalizer Backend显式重建所有action token，新增逐项原生IO/时序/资源记录、
ALL164原输出对照，以及zero/tie/missing独立分栏。开发中发现并修复内部ndarray
误传literal-JSON校验器、RAM factory漏check参数；二者发生在新诊断路径，不是
已证实的历史P1根因。旧统计、gate、动作、模型和旧结果均未改变。

与oracle锚点替换后仍21反转的[另一项干预](D2P_ANCHOR_FACTORIAL_AND_HILEWM_REPAIR_MAIN_20261028_v1.md)
合看，共同偏移与这次固定统计替换都不能单独根治。训练run→dataset→scaler的
身份仍未建立；unknown真实训练normalizer/视觉域、动作条件响应、渲染不可观测性、
任务目标充分性与搜索覆盖仍须分开。不能据此推出理论不可行，也不能断言所有
代码问题已排除。

MAIN另只读查明官方DrawOptions.to_pygame对坐标round为整数，物体最终绘制再
resize；原24不可见对agent整数中心均相同。但另28个可见对也共享agent整数
中心，因此agent中心取整只是部分线索，尚未复算全部物体顶点/遮挡来认证唯一
栅格化根因。不改官方renderer制造新的训练域，更不把tie算作方向正确。

P1、freshness、training_provenance、deployment_allowed、scientific_admission均false。
这轮是执行完成的排因进展，不是整套研究闭环或P1通过。
