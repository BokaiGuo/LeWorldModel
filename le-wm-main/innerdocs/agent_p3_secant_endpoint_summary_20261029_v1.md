# P3 C 对 Z/P：同时间native endpoint独立saved summary

仅新增本stem source/tests/doc。worker只做RAM/ABI检查；不运行actual build/verify、fullstrict、NN、physics，不读未来probe、不修改原策略/门。

## 两个显式MAIN权威

- C完整receipt：`results/agent_p3_secant_commit_physical_20261029_v1_consume_MAIN01.json`，SHA `d0c5a34b14dc84ffece573d904e3950e54628df307987ee77fcb6265ca91b638`。
- Z/P完整receipt：`results/agent_p3_secant_controls_first5_physical_20261029_v1_consume_MAIN01.json`，SHA `c58ca1e43439841a3e592edb1128f95e2b5b8375af8f686052ab662930e74b50`。

生产入口只允许这两份原文件/哈希，不打开bindings指向的祖先产物、PNG/weights，不调用任何上游consumer。导入MAIN已完成的严格认证，不冒独立fullstrict或动力学验证。源码仅stdlib；四份已读公式/producer源码作为SHA证据闭包，不import执行。旧文件全部不改。

## 固定身份、分母与时间

C顺序root0 H1/H3、root1 H1/H3共4逻辑槽：第0有真实endpoint，其余3为no_witness，状态/动作/标签/计数严格保持缺失，不借Z/P填C。Z/P顺序root0 Z/P、root1 Z/P共4引用；每C槽对同root的Z与P，共8比较槽，全保留，当前设计6个比较因C缺失为null而不是tie。

本工具不根据结果筛选。核两个schema/status、完成report指向、原科学关闭标记、全分母；核Z/P中`planner_authority.authorities.C_receipt`精确指向C整份权威，`logical_C_slots`逐项与C真实checkpoint/null相同。按root核原task/model task/goal/history/provenance/source_slot、prefix raw一致；有真实观测的分支核cp0/cp5状态及像素SHA一致，仅作为共同起点/执行前缀身份门。

唯一endpoint为**control10 / absolute20**：来自checkpoint step10、对应实际response10、final state、snapshot state及完整float64 bits。Z/P自带endpoints逐项核。cp5是control5/absolute15历史，绝不当收益baseline，也不以其代endpoint。native提前停止而未到10则endpoint null；恰在10停止且具完整返回可用。原C完整权威固定3个no_witness，不能靠新增summary生成动作/观察。

## 独立算术与方向

按原顺序`[agent_x,agent_y,block_x,block_y,block_angle_mod2pi,agent_vx,agent_vy]`，原state7−原goal7逐轴Python float64相减/平方、math.fsum求和、sqrt；不复制producer reward/metrics数值当独立算术。输出原状态/goal、7轴有符号/绝对/平方差、joint XY、agent XY、block XY、**agent** velocity、native7范数与负reward，四组平方及求和roundoff。不同归约原numpy norm可能有末位差异，不放宽阈值或伪称逐位等于旧producer距离。

原native成功规则独立重算：`jointXY < 20 AND min(abs(raw angle delta), 2pi-abs(raw angle delta)) < pi/9`。保留其可能为负的native angle term；不modulo、不以wrapped替换。另报告raw angle绝对/平方差与atan2(sin,cos) wrapped误差/诊断成功。成功必须与原receipt原native_success一致；不重定义任务目标或采用object-only成功。

8槽固定方向C−对应Z/P：

- 各非负误差（native7、位置、速度、raw/wrapped angle及逐轴absolute error）：C误差严格大才`C_worse=true`，严格小则`control_worse=true`，相等含±0为tie。
- reward：C严格小才C即时更差，方向不可反用；native成功：C false且control true才C即时更差，反之control更差，相同为tie。
- 直接state坐标差和native min-angle term差仅带符号报告，不把坐标数值高低/负native term当物理改善。
- 任一endpoint缺失则比较项null（非False/0/tie）；各误差独立统计C_worse/control_worse/tie/null并保8分母。无混单位总分或winner；cp5不出收益差。

这是同已知根同即时控制时间的描述性比较，不是完整H3策略、非贪心能力、独立样本或机制证据。P1/P3/C3科学标记false、FLOPs未知。原两阶段physical计数/PNG计数分别导入，不因本地重算追加physics费用。

## MAIN命令

在 `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p3_secant_endpoint_summary_20261029_v1.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p3_secant_endpoint_summary_20261029_v1 build --output results/agent_p3_secant_endpoint_summary_20261029_v1_MAIN01.json
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p3_secant_endpoint_summary_20261029_v1 verify --summary results/agent_p3_secant_endpoint_summary_20261029_v1_MAIN01.json --summary-sha256 <build实际输出SHA>
```

单元测试为fixture，覆盖全轴独立算术、native严格边界/负min-angle、速度与位置反向、8槽/3no_witness、cp5替代拒绝、wrong-time/goal/raw历史权威/endpoint/标签篡改、提前停止、排他写和临时两文件build→verify。无生产结果分析。

上限：单receipt/source128MiB、输出4MiB、free floor200MiB、整次本地读/算/fsync120秒。排他创建、目录/文件fsync、源与两receipt前后SHA核；超时失败保留文件，不覆盖。verify只重算这两个receipt本地摘要，不重跑上游严格链。下一步仅MAIN读测后真实saved build/verify。
