# P5 非零响应因素：saved-only 探索性分析

这是已读既有outcome后的探索性、描述性分析，不是预注册、独立确认或因果检验。指标合同的冻结只防止此后执行漂移，不改变这一证据等级。worker只交付source/tests/doc；MAIN审读后执行actual analyze/verify，无NN、physics、训练、拟合或新实验。

## 不重复的既有资产

`agent_p5_jepa_callback_zero_sets_20261029_v1` 已完成H1标签特征精确零集合与callback新重放三层集合连接，actual analysis SHA `35b4c01e4ce7ad846af740878b5399ef17247425f22cba44ceb330a55755c3be`。本工具复用其80个完整身份及raster记录，并核全部保存bindings，不重跑callback。不把24个零响应仅凭数量相同连接，也不升级historical callback membership=null。

`main_raw_hscale_20261028_v1` 已分析旧LeWM192的两尺度；本工具不复制它。新增内容是同一冻结JEPA模型H1/H2真实完整patch的时序控制、误差分解、有限步长局部非线性、固定因素分栏。作者值/跨架构MSE不进入比较。

## 固定分母和身份

保留80对、164分支、40个跨尺度对应；两known roots×off/on×h=.01/.001×10坐标。既有H1实际label特征差精确非零的56对是固定关注cohort，不是56个都方向反转，也不按cosine符号或大小选入。24个零响应仍完整保留。原56成员重算不一致或任何missing/重复/错时刻/错selection必须拒绝，不能重定义分母。

外部显式SHA绑定H1/H2 report与自己的strict、既有zero-set analysis；原FD prepare与原physical report固定SHA来自本批已封存原始资产。逐分支检查root/condition/task/selection、原float32 raw10/producer_job_sha256，H1/H2必须为各自原时间合同并指向完全同一terminal224原PNG。新分析只继承SHA固定的完整upstream strict，不宣称在此重新执行所有upstream consumer；其状态在报告明列false。

实际重读全量original float32 NPY、descriptor/inventory/SHA/shape/bytes/EOF/finite，原PNG mode/224像素SHA、完整native7；独立重算三时点endpoint MSE必须逐字等于旧report。源码/模型加载state绑定H1/H2相同。H1 encoder标签batch1、H2 batch2可能有末位差，逐pair记录两套terminal label bitwise/maxabs，不偷换一套标签假定完全相同。

## 三个时序分栏

* H1_20：实际5/10/15条件→20，仅最后prediction token；另外两个token不解释成自由rollout。
* H2_15：实际0/5/10→预测15，尚未执行所扰动的raw16–20；核first predictor已认证input descriptor的内容SHA树相等（忽略不同文件名）、15标签PNG identity和预测/观测特征精确零。结果全量列出，不把异常隐去。
* H2_20：只由预测15的visual/proprio滚动→20，不接入真实15。H1/H2改变实际/预测中间条件，不是只改变horizon的独立因果干预；额外原生agent proprio权限保持披露，不能称RGB-only公平跨模型比较。

## 数值诊断，不做互斥“原因”判定

1. 信号幅度：全部原RGB changed pixels/channels、整数max变化/RMS、真实特征响应L2/endpoint L2比、float32单ULP endpoint envelope的连续比。只将精确零设为零，未选epsilon、不设低/高信号阈值。ULP envelope只是表示精度尺度，**不是**模型噪声方差/置信界或渲染量化步长；小信号关联不能证明因果。
2. 局部非线性：取原anchor及每个正负probe，使用实际float32的minus/plus距离（允许不对称），分别算左右斜率cosine、二阶secant范数、左右分歧比。再对全部40对应比较两既定h的中心secant方向/幅度；native7、物体xy/包角度另列各轴secant，不混单位给整体norm解释。有限h差异不能区分dynamics、raster、encoder各自非线性，更不能证实或否证无穷小导数。
3. 预测误差：`dp=pp-pm, da=ap-am, em=pm-am, ep=pp-ap`，精确保存`dp-da`与`ep-em`的算术残差；每端MSE、common error `(em+ep)/2`、differential error `ep-em`以及二者解释endpoint MSE的代数残差。共同offset可很大但不影响响应；方向失配必须看differential error。`dot(dp,da)/dot(da,da)`与正交分量仅为当前响应的代数投影，不拟合Q/readout、不学习校正权重。真实响应为零时ratio/cosine为null。

全部因素表固定列root、condition、step-size、coordinate、global raw step16..20、x/y以及root×condition×h；同时列all80与既定56支持数、符号计数和连续量min/median/max。H1→H2终态方向符号转移矩阵保留正/负/零/null全部16格。不挑最优h、根、坐标、overlay、特征区域或方法；不据分组结果调后续阈值。

原生物体姿态分轴、agent位置/速度和完整7维delta分别保留。object-only变化不能冒充完整PushT任务失败或成功；本分析不重定义native success。

## 输出、边界和工程预算

新独占JSON包含protocol/arguments/bindings、80 rows（各3 phase）、40 cross_scale、完整factor_tables、paired_direction_signs与剩余识别缺口。无新图片/latent文件，不复制原大NPY；按41分支组顺序计算、不建D×D矩阵。600s工程wall、32MiB输出cap、200MiB disk floor，canonical JSON exactly-one-LF、exclusive xb/fsync、不覆盖。超时/缺项报错、不写成完成的科学负结果；不承诺峰值RAM上界。

analyze独立重读所有真正使用的binding；verify以保存arguments重算整个报告逐字段一致，而不是只检查摘要。无训练/新NN/新physics；所有科学门false、historical membership仍null，不签充分原始因果认证。仍缺反事实物理、量化与动力学/encoder非线性独立分离以及纯horizon干预；相关性和代数恒等式不填补这些缺口。

## MAIN完整CLI

从 `/home/gbk/code/LeWorldModel/le-wm-main`，NEW输出必须不存在。此处是真实上游SHA，不预填未来analysis SHA。

```bash
.venv/bin/python -m pytest -q tests/test_agent_p5_nonzero_factors_20261029_v1.py
.venv/bin/python -m experiments.spectral_reachability.agent_p5_nonzero_factors_20261029_v1 analyze \
  --h1-output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p5_raw164_jepa_cuda_20261029_v1_cuda_MAIN01 \
  --h1-report-sha e7ac4ef9eb7491f619de25b04c55f5e56560e2833a840dd5305f3db6207b3a5d \
  --h1-receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p5_raw164_jepa_cuda_20261029_v1_consume_MAIN01.json \
  --h1-receipt-sha 4f6bbed041f077818627e68157b0cfab99ddba2e38100827db10f46c4aa1b759 \
  --h2-output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p5_raw164_jepa_h2_cuda_20261029_v2_cuda_MAIN01 \
  --h2-report-sha e5a561215bfe7b6f1c1c224e7661211e3f237fb16458a6a409f5a1da0fc08579 \
  --h2-receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p5_raw164_jepa_h2_cuda_20261029_v2_consume_MAIN01.json \
  --h2-receipt-sha 783b65f67899791b5697bfe0819ccfb3231d57212b9e3e85aefc02419459c0ea \
  --zero-analysis /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p5_jepa_callback_zero_sets_20261029_v1_MAIN01.json \
  --zero-sha 35b4c01e4ce7ad846af740878b5399ef17247425f22cba44ceb330a55755c3be \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p5_nonzero_factors_20261029_v1_NEW.json
.venv/bin/python -m experiments.spectral_reachability.agent_p5_nonzero_factors_20261029_v1 verify \
  --analysis /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p5_nonzero_factors_20261029_v1_NEW.json \
  --analysis-sha ACTUAL_ANALYSIS_SHA
```

worker已做真实H1/H2 metadata-bundle检查（各164行/80对），没有运行actual analyze、没有读取生产NPY评分或生成生产回执。RAM测试覆盖线性/曲率、量化零与非零幅度、common/differential恒等式、时序/selection/raw拒绝、真实NPY EOF/hash、原native7权限、完整80/56/40编排、factor与writer重复写拒绝；full orchestration用明示小向量/metadata doubles，不是实际科研结果。MAIN仍须亲审代码并执行saved analyze/verify，不把测试通过视为科学通过。
