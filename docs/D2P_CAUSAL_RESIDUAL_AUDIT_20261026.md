# D2P：同动作误差补偿与真实因果执行的分离

## 当前判断

不能因为代码测试和若干接口审计通过，就将剩余负结果宣判为理论失败；也不能
预设一定有某个bug，再用调阈值、挑根或修改目标制造通过。本轮实际检验了一个
可执行的补偿办法：只利用自身分支上一包已经观察到的预测误差，修正下一包
预测，并重新优化动作。该办法不更新官方冻结模型、不拟合adapter、不读取
未执行分支或未来观察。结果未支持把它作为P1修复或默认运行时。

本轮没有发现能够解释剩余反转的新代码根因。**P1未解决，整套研究未闭环**。
这不是D2P理论反证；历史BN污染、目标关节限位、求解不足等已经证实的问题
仍需与本轮尚未解决的预测/执行失配分开报告。

## 预先固定的干预与实现

保留全部16个已知根`202650000..007`及`202651000..007`、三种精确执行器
掩码、三类既有程序、原前缀和原目标。全部使用官方原始checkpoint，权重SHA为
`eb70b1fd5409f8f81875d62f5ee5a20dd220a3128a477de66b5760f475f0f469`。

令第t包的原单步预测为f_t，执行后的真实RGB表示为z_t，固定：

`b_0 = 0; b_t = stop_gradient(z_t - f_t); corrected_f_(t+1) = f_(t+1) + b_t`。

必须减去**原始预测**，不能减去已经补偿过的预测，否则会错误地反复抵消残差。
每条分支单独初始化；不能借用其他候选的物理结果。首包不改变，后两包沿用
四次GN、原action box、阻尼0.01和同一线搜索，仅改变模型预测的偏移。
所有动作在真实执行前写入selection receipt，绑定上一包执行SHA和RGB frontier。

入口`audit_reacher_d2p_masked_feedback --causal-residual`新增显式v2实验合同。
不传开关仍保留原v1三模式。新模式为`causal_residual_goal`；原replay、goal与
waypoint三组对照全部保留。`causal_residual()`负责有限数值、192维形状、dtype、
device、detach和溢出检查。优化器的标签禁用字段仍指未执行物理标签；已经
执行过的RGB通过显式观察历史和上述残差进入下一次计算，不能把它称作无反馈。

## 同动作诊断不等于执行修复

独立NumPy后处理只读上一轮CUDA保存的完整192维向量，先计算当包预测，
再使用当包观测评估。它不产生新神经预测，也不改变原动作。真正不同的
native/greedy程序只有以下三个根/任务；其余相同程序全部保留，不能算成额外
独立排序成功。表中为终态native减greedy的RGB成本：

| 根/任务 | 原真实历史单步预测 | 上一包误差补偿后的预测 | 真实RGB |
|---|---:|---:|---:|
|202650000，仅腕|-0.040465558|+0.027746024|-0.052696885|
|202650001，仅腕|-0.032373879|-0.026567313|-0.069903544|
|202651004，全驱动|-0.000349849|+0.000619255|+0.002648358|

补偿将004改为正确方向，却将000改错。因此单看004不能宣称排序问题已解决。
这些使用各分支实际历史的数值仍是回顾性定位，不能在根时刻同时获得所有
未执行候选的历史，不能冒充可部署的反事实selector。

## 主机CUDA执行结果

权威报告：
`le-wm-main/results/reacher_d2p_masked_causal_residual16_20261026_cuda.json`。

全部16根×三任务×三固定程序×四执行方式，共576条真实15步分支、1728包、
29376个CUDA predict样本；12项门全部通过，参数、buffers、eval状态、权重文件、
输入和归档源码检查通过。没有CPU神经回退、权重训练或P1准入。

独立分析重新验证全部1728组选择/执行链、残差来源、完整动作、RGB/目标成本、
原拓扑关节距离和优化器receipt。额外对照旧反馈报告，原三模式的432条分支、
1296包，其动作、完整192维预测、物理观测和求解receipt全部逐位一致。
因此本轮新增模式没有暗中改变原对照。

全部16根native真实终态关节距离均值（周期肩、有限位腕）：

|任务|原动作重放|目标重规划|原路标跟踪|因果残差+目标重规划|
|---|---:|---:|---:|---:|
|全驱动|0.127644|0.170523|0.113647|0.168352|
|仅肩|0.587835|0.765244|0.591332|0.766736|
|仅腕|0.675340|0.701611|0.674113|0.694775|

残差模式相对同预算目标重规划，三任务根级胜/负分别为4/12、6/10、5/11；
均差的描述性配对bootstrap95%区间分别为[-0.035388,+0.022252]、
[-0.014678,+0.015648]、[-0.019476,+0.002787]，均跨零。
对应RGB成本区间也均跨零。不能以均值小幅改善或少数根显著改善替代稳定收益。
本轮是已知根诊断，不是独立确认或多重比较控制后的效能结论。

004全驱动、同样施加因果残差反馈后的native/greedy：

- 真实终态关节距离：0.020871953 / 0.014420466；
- 真实终态RGB成本：0.003729284 / 0.001057303。

native仍更差。固定动作上一个预测margin修正，不能推出经优化后的新动作真实
更优；新动作改变了模型误差，而上一包偏移不保证迁移。这里不存在代码实测
证明“补偿必然有效”，也没有理论结论要求冻结近似模型的局部排序永远正确。
不调残差增益、不挑根、不将此模式设成生产默认，不据本轮开放P2/P3/P5。

## 复现、图和测试

工作目录`/home/gbk/code/LeWorldModel/le-wm-main`；神经命令只在授权主机CUDA运行，
已有输出不可覆盖，复跑需更换完整输出路径：

```bash
.venv/bin/python -u -m experiments.spectral_reachability.audit_reacher_d2p_masked_feedback \
  --reference results/reacher_d2p_masked_packet_fidelity16_20261026_cuda.json \
  --policy /home/gbk/.stable_worldmodel/checkpoints/reacher/lewm \
  --adapter-artifact results/reacher_d2p_p1_distribution_repair_v1_20261024_cuda/reacher_training_action_stats.json \
  --causal-residual --output results/reacher_d2p_masked_causal_residual16_20261026_cuda.json

.venv/bin/python -m experiments.spectral_reachability.analyze_reacher_d2p_masked_feedback \
  --report results/reacher_d2p_masked_causal_residual16_20261026_cuda.json \
  --control-reference results/reacher_d2p_masked_feedback16_20261026_cuda.json \
  --output-dir results/reacher_d2p_masked_causal_residual_figures16_20261026
```

全根双图包同时保存PNG/PDF、summary、notes、LaTeX及SHA manifest；图已目视
检查，保留大误差根和相同程序的零margin。相关新测试仅为数值/合同fixture，
不作为神经效能证据。本轮新增19项数值/因果合同测试，相关广覆盖1150项通过
（43.75s），`git diff --check`通过。旧实现归档和所有负结果保持不动，无提交、
推送或数据删除。
