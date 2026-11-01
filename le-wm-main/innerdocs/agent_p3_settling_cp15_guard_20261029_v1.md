# P3 settling：cp15 全证据 guard（未完成 driver）

本件是独立可复用 guard 和整数预算核，不是原生物理 driver、运行入口、saved consumer
或物理运行授权。只读引用已封存的 Stage C prepare/batch/full receipt，三者实际SHA分别为
`8fdecdc9fe5fcb1f00b03c30e9971566da03c2a523ccc652d86065c9b442e95f`、
`b70022a70a32c0c5f743684276e42f9040e228ae8296bb6daeae5fb70ce28c1b`、
`4e7d35ec52dba109c9e2c181ad46d754f44bc0c0ef41a362c30a68776c66f3b1`。
固定旧source SHA `f070b6afe4fe0ebba66e05d366f929467602b98805980f86f27e2344445a8344`。
不重新运行 upstream fullstrict；旧 receipt 是明确的信任起点，不是本件重建的认证。

`load_references()` 只读核上述SHA与十行完整job/report/receipt join。`guard81(job, original,
observations, operations)` 核按原顺序的81观测及旧完成前全部373个 native 操作收据；每项全字段
一致，观测只按旧枚举运输字段映射。验证原25个raw的 float32 位级顺序和最新动作、全部返回、
最后10动作历史、cp15 off/on/off三帧、goal、完整snapshot及 state float64 bits。输出完整映射和
摘要，未记录的RNG/contact-cache仍unknown。guard81本身不验证实际PNG文件内容，也不自签输入
authority：未来driver和consumer须将所有已绑定 PNG 文件hash+解码像素独立验证。

`before_step16` 先完成guard，再调用调用者的 durable sink，再调用动作 callback；sink 必须真正
写入并fsync之后返回，本件的 RAM fixture **不能证明**未来driver实际持久化。已测试sink异常时
绝不派发。consumer仍须按实际事件hashchain重建guard并证明它在step16 intent之前；不可仅接受
receipt中的布尔值。tail全十支固定原生相对raw正零五步，原15步和原科学gate不变。非零goal
agent velocity按完整goal7继承；raw零可有阻尼及接触，不能断言静止或改善reward。

预算每支constructor1/close1/reset2/set_state4/goal_render2/goal_bind1/goal_setter3/
public30/physics304/render96，共444 operations、888 operation事件、96 observations、
3 guards、3 lifecycle = 990 events。十支public300/physics3040/render960。
96张PNG的原通用上界加旧 metadata quotas与final reserve小于64MiB，但**新增事件的 metadata
结构上界还没有完成证明**，代码据此显式 `runtime_ready=false`；不能以此宣称driver已可生产。
真实早停应停止动作并保留缺失cp20；失败attempt不退款，unknown不能补0。

工作目录 `/home/gbk/code/LeWorldModel/le-wm-main`，仅RAM测试：

```bash
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p3_settling_cp15_guard_20261029_v1.py
```

测试覆盖10支、state/snapshot/像素摘要/goal/raw/receipt篡改及guard fsync失败零派发。
另外只读join实际三authority并以各自已存行调用guard检查十支字段形状；这不是物理重放。
实验代码技能仅用于固定预算和失败记录，未训练、调参或生成图。

待完成：自有Store/Meter/instance native hooks与fresh env执行、完整原51/66/81 guard顺序、
96 PNG/完整事件及partial保留、全生命周期publication、全10 own-schema独立saved consumer、
source closure和输出/墙钟完整预算证明、早停/故障/真实临时盘十支回归、MAIN生产CLI。
在这些实现并通过审核以前，不提供运行命令，不得派发真实环境。
