# PushT：解除包内常值动作限制的配对编译实验

假设：当前prefix/suffix只用二维atanh命令并重复五步，可能丢失模型支持的包内
时序控制方向。这是实现容量限制，不预先认定为P1失败根因。旧所有源码/产物不改。

新比较保持四个已知校准根、模型/normalizer/goal0/Q、两packet十物理步、每臂十
candidate、stage+own reach选择、谱前两方向和范数尺度一致。constant使用等距
C=tile(I2,5)/sqrt5，full使用I10；参数均在同一十维packet-logit欧氏球中。
半径sqrt5×(.25,.75)，DLS cap=sqrt5×.75，ridge=.002。constant的谱扰动保持
旧重复命令的物理尺度；这不是旧所有超参和无约束suffix的逐位复现。

suffix使用同一半径球上的精确约束ridge解，记录KKT multiplier和实际应用update
对应的线性残差/penalty，不把clip后动作与clip前乐观分数混用。非线性预测只诊断，
不改变原stage+reach排序。两臂额外差别是控制子空间维度；不声称相同总算力，
Jacobian backward向量工作与前向单列。所有神经工作仅主机CUDA。

每根先复算完整192×10 root Jacobian，直接计算constant Jacobian核对链式法则，
并执行十坐标中心FD。每candidate own suffix重新求192×d Jacobian、非线性预测，
以现有独立CUDA adapter从root完整重放两packet，终态必须逐值相同。四根全部
plans封存后，才由另一个物理driver执行全部80候选；本模块不执行环境动作。

此实验并未修复Q的物体/agent失配，也不保证线性化可靠或接触稳定；半径是固定
校准设置，不是已证明的安全/信赖域。P1保持关闭，不按真实结果改变根或参数。

cwd `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
.venv/bin/python -m pytest -q tests/test_pusht_full_packet_compiler_20261027.py
.venv/bin/python -m experiments.spectral_reachability.pusht_full_packet_compiler_20261027 \
  --compilation results/pusht_d2p_metric_program_preflight4_v1_20261026_cuda/report.json \
  --policy /home/gbk/.stable_worldmodel/checkpoints/pusht/lewm \
  --output results/pusht_full_packet_compiler_20261027_cuda_v1
```

输出目录必须全新。计划根Jacobian88次vectorized reverse、16896个scalar output
VJP方向；它们不是GPU kernel调用/FLOPs。模型predict instrumentation记录实际
前向calls/samples，FD/evaluation重放包含其中，不把它们冒充部署预算。实际候选
生成使用两臂各自root响应谱，因而同候选数不意味着同一候选集合或纯optimizer消融。
