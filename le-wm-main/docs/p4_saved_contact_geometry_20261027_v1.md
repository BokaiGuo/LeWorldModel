# 保存轨迹的官方形状几何接近标签（不开放 P1/P4）

本任务使用 experiment-code 技能的真实保存数据、来源核验、非神经测试与运行记录
规则；只新增本 basename 模块、测试、本文及独占 output。旧 sealed 文件与 central
progress 不改。没有 GPU、神经推理、训练、环境实例、Space、reset、physics step、
下载或删除。没有根据 outcome、任务成功或 selected 分支表现拟合任何阈值。

## 先固定几何合同，后读取保存轨迹

安装环境官方 `stable_worldmodel/envs/pusht/env.py` 必须精确匹配已审阅 SHA256
`c6d2104adc368ff1c352edc0e35ac74da9f07fda3a748ddbef0ba6b7a2d1daa2`，
且历史 source seals 必须包含该文件及相同 hash；不同版本／缺少真实 shape 绑定
直接 fail closed。不能退回点到物体中心的距离。历史 source/archive/args 和本轮
读取输入末尾重哈希，环境契约变化不能静默继续。

`DEFAULT_VARIATIONS` 只随机 agent/block 起点与 block angle。默认 agent 圆形、
scale40，`add_circle` 半径 `0.375*40=15`；block 为 T，`add_tee` 强制 scale30，
覆盖传入默认40。两个局部 polygon 精确为：

```
[(-60,30),(60,30),(60,0),(-60,0)]
[(-15,30),(-15,120),(15,120),(15,30)]
```

AST 只读取上述尺寸的 literal 算术，不导入官方环境。合并外轮廓去除内部接缝，
保留凹角。`_get_obs` state 是 agent.xy、block.position.xy、block.angle、agent.vxy。
使用不附着任何 Space 的官方 Pymunk Body/Poly 进行静态坐标转换，保留官方两个
polygon 平均 COG，按 snapshot angle 后 position 设置，再调用 `local_to_world`。
不能把 COG 再从保存的 block.position 任意减掉。测试使用官方 Poly.point_query
独立核对外部距离、非零角、刚体不变性、凹形和接缝；不运行动力学。

长度单位是512窗口的环境单位，不是224 RGB像素。agent中心到T外边界的精确
带符号距离（内负）减15得到 signed_surface_clearance；正值是表面间距，负值
是静态几何穿入度量，不是接触力。另输出 nonnegative_separation。

## 固定、非互斥阶段 flags

Pymunk 安装源码 `Space.collision_slop` 说明默认允许重叠0.1。此量作为唯一长度
容差，**不是把正间距0.1称作实际 collision 条件**。`near_contact = clearance<=0.1`
只是显式几何邻近 proxy；`geometric_overlap=clearance<=0` 也不证明力或动态接触。
逐step输出agent/block平移向量和范数、wrapped block角变化、最大对应顶点位移、
clearance变化。approaching/separating 是 clearance变化小于-0.1／大于0.1；
block_sub_slop_motion 是最大顶点位移<=0.1，不称严格静止或噪声统计判断。
near_contact_with_block_motion 只是近接且block移动的共现，不称因果push。
force_contact、dynamic_contact、task_closure 恒为 unknown/null。没有从 goal 或
旧 outcome 推出闭合阈值，也没有采样间扫掠碰撞保证。

## 数据资格与缺失

P4 8分支×10步=80：逐步 state与独立step文件／branch／execution内容对照。
原runner没有保存每分支初始物理state，所以8个首步位移及差分flags必须unknown，
其余跨round连续计算。不能从另一任务的root重放臆造初始state。P4只有每轮终图
保存bytes：核验16张 terminal.npy 与逐步array SHA；其余64步只有历史RGB hash
witness，明确标为 bytes 未核验，不宣称所有80帧可重新检查。

Capability同算法审计123个分支×10步=1230：完整123 plan/branch census、逐分支
JSON、journal SHA、initial state、逐步journal/trace一致、0..10 PNG文件SHA。
保留selected/greedy/zero全部分支，不仅选41个有利例。summary单列步6..10的
间距范围、near-contact次数与最大block顶点单步位移，供解释selected41 suffix
静止与agent分离是否共现；不能把snapshot远离当作不存在任何中间接触的证明。
RGB只做来源核验，本轮无图像分割或从像素估计接触。

## 运行

工作目录 `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p4_saved_contact_geometry_20261027_v1.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.p4_saved_contact_geometry_20261027_v1 --output results/p4_saved_contact_geometry_20261027_v1_main_v1
```

output必须全新且本basename开头；运行前封存新模块/test/doc和纯读取helper及官方
环境／Pymunk转换源码，输出report.json独占/fsync，运行后核验seal。几何算法确定性，
没有当前时间／随机数／新outcome筛选。exit0只指保存几何标签生成成功，exit2保留
integrity_fail。所有输出P1/P4 admission=false。没有用此proxy补足上游准入、独立
根长程效能、真实接触力或完整成本匹配证据。实际数量和距离以新report为准。
