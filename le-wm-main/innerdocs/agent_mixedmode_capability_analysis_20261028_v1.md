# Mixedmode真实任务有符号探针响应（saved-only）

新增本stem源码/tests/innerdoc，不改physical/probe/mixedmode/MAIN封存。没有神经、环境、训练、Q拟合或根/方向选择。输入唯一真实physical report SHA
`57ac436ff6cd7fcffcda62e0a165782b719c0383404647f2f87c1fa97ab009c8`，
目录 `results/agent_mixedmode_capability_physical_20261028_v1_MAIN01`。
固定physical source SHA `21835721f0ee360dde66f1b0b754332438660ebac1fed636794fadbf6865a88c`。
执行入口**完整调用 physical.consume**，其成功后读已经验证的原probe.prepare与原生goal/root，报告绑定own三文件SHA、物理/probe/prepare/journalSHA。不能绕过consumer用raw report冒称已验证。

MAIN已做native独立审计：`results/audit_mixedmode_capability_native_MAIN_20261028_v1_MAIN01.json`，SHA `4e90b88ffb135aeafdd63b68cda57b75f6315638bf1f6829a75822d8d07b622e`；本模块不重复NPZ奖励/终止审计，该回执不是替代physical.consume或本轮有符号算术。

## 固定方向与精确量

对每个原slot×condition、每个role(before/after)、每个预封存probe，令 `s_b` 为五步真实endpoint，`s_0`为同角色zero endpoint。主响应 `d_b=s_b-s_0`；before/after使用同一**原root→原goal**方向 `u=(goal-root)/||goal-root||`，报告 `u·d_b` 与after-minus-before。方向不随after residual、已见响应、成功或cost重拟合。原residual恰为0则u和投影null，不挑新方向。全五探针顺序zero,+x,-x,+y,-y保留，无max/最佳方向/阈值筛选。

四个分开的特征空间：joint_xy=原native7前4维（agent+object），agent_xy=前2，object_xy=第2:4，object_angle_wrapped=第4。前三分别是各空间的原residual单位方向，不能直接比较其不同投影的大小作统一指标。角度先atan2(sin,cos)，使用原角度residual的符号，不把2π表示差当物理能力；角度和px不加权混合，也不改native官方角度/终止规则。native7的速度维及完整body velocities保留在start snapshot与原报告，不拟合它们进新Q。

同时逐轴给出中央有限secant `(s_+ - s_-)/(2a)`，a从封存literal float32 probe读取，代表**连续五步保持该raw幅度**的有限响应，非单步/无穷小导数。固定有符号投影逐列报告，不用secant范数或谱大小判好。附±关于zero的非线性不对称项。角度pair跨π时标branch-cut diagnostic。无符号谱字段明确null，本模块不重复旧谱审计。

正投影只表明相对zero朝原任务射线移动，**不等价于欧氏终距改善、native success或P1机制**。例如object不动、agent有利移动，object投影0不能作为agent+object全任务失败。本模块不根据terminal loss重新标成功；last_return原reward/terminated/truncated只原样留作绑定，不改native门槛。报告不做统计显著性或对照强基线排名。

## 分母、时间混杂与失败

固定两个known roots×off/on=四任务；两role=八组；五probe=40分支。不得筛root、condition、probe或把条件重复当fresh独立根。source任务/selection/context/job_digest严格join。各role五个独立重建分支必须有相同实际start state；不拿索引猜身份。

zero endpoint缺失使该角色所有zero-relative有限响应null，不补零；单probe缺失只影响该probe及相关±pair。secant只需±两个真实endpoint，zero缺失时仍可数值给出但不能冒充已取得zero-relative响应。保留40分支status/attempted/completed/error/last_return，早停不当完整终点。consumer拒绝时不产生成功analysis。

每对before/after单列：probe前elapsed public10 vs15、actual start state7、完整start body snapshot、各空间zero动作drift。zero减法**没有**控制额外五步、agent/object位姿、速度、接触和未导出solver缓存；无matched-time控制，`causal_capability_gain=unknown`。不由末态推断全轨迹接触，不把终距或无符号谱代替有符号响应。P1=false、independent_confirmation=false、fresh_roots0，固定库负结果不否定理论。

## 实际运行与限额

saved-only完整consumer会读取原44k完整closure并复算既有数学、校PNG，非模型或物理重跑。独占新JSON输出最大1MiB，磁盘floor200MiB，xb/no-overwrite/fsync，不写共享总账或旧报告。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_agent_mixedmode_capability_analysis_20261028_v1.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
 .venv/bin/python -m experiments.spectral_reachability.agent_mixedmode_capability_analysis_20261028_v1 \
 --physical-output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_mixedmode_capability_physical_20261028_v1_MAIN01 \
 --physical-report-sha 57ac436ff6cd7fcffcda62e0a165782b719c0383404647f2f87c1fa97ab009c8 \
 --destination results/agent_mixedmode_capability_analysis_20261028_v1_MAIN01.json
```

测试仅RAM合成：符号/反号、joint与object-only区别、zero漂移、common vs own方向、缺失null、2π边界、分母和身份拒绝、full consume不可绕过。合成测试通过不等于科学通过。实际saved输出另提供SHA，结果前不预判；下一科学前置仍包含matched-time或其他能识别机制的独立控制、有效目标模型与fresh确认，不由本描述性报告自动准入。
