# MAIN-only mixedmode before/after physical capability probes

新独占 source/tests/innerdoc。所有旧源码、结果、probe v1/v2及MAIN实际prepare保持不变。此模块补齐真实执行器，不重标旧post-only数据、不跳过相同旧42候选。只在MAIN审核后由MAIN执行。worker测试全部RAM mock，没有执行模拟、神经、prepare或网络。

## 两层封存与可执行入口

冻结probe **v2** 的实际prepare为
`results/agent_mixedmode_capability_probe_20261028_v2_prepare_MAIN01.json`，
外部SHA `e8e4554ea2448a8c4bcf4e916934800fa8c7afe3eda335174056a551620da3d5`。
它保留44,042个binding，worker此前真实saved-only canonical build为9,315,944 bytes，MAIN实际写出约8.9MiB。旧v1的8MiB容量失败保留，不是科学负结果。v2已封存，本模块不修改之。

本模块另建own prepare，绑定probe prepare/完整closure、自身三文件、公开helper、RgbObservabilityGuard及安装env源码。`material/verify`分别完整调用probe.verify及原source_material，只读source recipe/真实history/goal/PNG/task，不进行模型或环境调用。安装env AST核验constructor在hooks之前不自行reset/step/render/setup；部分constructor失败内部工作仍unknown。probe.contract的execution_ready=false仅表示原合同不授权物理；本模块own contract+显式授权才开启执行。own prepare最大64MiB，不复制上游模型数组。未知/冲突/缺失binding全部拒绝。

从 `/home/gbk/code/LeWorldModel/le-wm-main`，MAIN下一条命令：

```bash
.venv/bin/python -m experiments.spectral_reachability.agent_mixedmode_capability_physical_20261028_v1 prepare \
 --probe-prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_mixedmode_capability_probe_20261028_v2_prepare_MAIN01.json \
 --probe-prepare-sha e8e4554ea2448a8c4bcf4e916934800fa8c7afe3eda335174056a551620da3d5 \
 --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_mixedmode_capability_physical_20261028_v1_prepare_MAIN01.json
```

取得实际own SHA后（不得拿probe SHA替代）：

```bash
.venv/bin/python -m experiments.spectral_reachability.agent_mixedmode_capability_physical_20261028_v1 verify \
 --prepared /ABS/OWN_PREPARE.json --prepare-sha ACTUAL_OWN_PREPARE_SHA
.venv/bin/python -m experiments.spectral_reachability.agent_mixedmode_capability_physical_20261028_v1 run \
 --prepared /ABS/OWN_PREPARE.json --prepare-sha ACTUAL_OWN_PREPARE_SHA \
 --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_mixedmode_capability_physical_20261028_v1_MAIN01 \
 --authorize-physics
.venv/bin/python -m experiments.spectral_reachability.agent_mixedmode_capability_physical_20261028_v1 consume \
 --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_mixedmode_capability_physical_20261028_v1_MAIN01 \
 --report-sha ACTUAL_PHYSICAL_REPORT_SHA
```

run无授权在任何input读取、建目录、factory之前拒绝。gym/stable_worldmodel只在显式run的factory内加载。Python API还需精确AUTH=`MAIN_MIXEDMODE40_BEFORE_AFTER_PROBES_ONLY`。没有run自动重试或恢复旧目录。consume输出stdout，MAIN如需另存须独占新产物；不重写报告。

## 真实因果执行与固定分母

顺序完全沿probe.jobs：slot0/off、slot0/on、slot1/off、slot1/on；各before/after；各zero/+x/-x/+y/-y。每分支fresh env，按原slot.reset_seed→原source.sample_state placement recipe→seed0 placement reset。不reset(savedpoststate)、不从标签选择动作/目标。

同原warmup10（前5[.1,0]后5[0,.1]）及原source same-condition RGB history3逐位校验；核原source全部可用17 warmup参考（off0..10/on0,5,10，off-after严格对同off）。中间on参考历史缺失不伪称已匹配；当前fresh on仍完整保存与off/on/off invariance检验。native7/body float64 bits/velocity/goal/renderer由既有snapshot和invariant完整记录，未导出solver cache不冒称独立证明。bind原真实7维goal和condition goalRGB，goal_pose不改。

before在warmup root立即用固定probe5，after额外派发mixedmode literal selected prefix5后probe5。全程原float32 raw，同候选buffer，无tanh、clip、/100或二次normalization。模型上下文/Q/latent全由probe.prepare及encoded task SHA封存，executor不生成模型候选、不接物理收益重选。

每warmup时刻0..10、goalbind后root0、after-prefix1..5、probe1..5均fresh off/on/off；probe0复用**紧邻**root/prefix5 fresh frame，不额外render。before每支17triplets，after每支22triplets，共780triplets/2340 PNG。初始+5 probe selected-condition frame作为新ABI的frames6，其余全部PNG/全snapshot仍在journal保存，不省略。native render不能插入physics/setter，off-replay像素精确相等；原RoyalBlue/canvas guard照旧。

额外直接调用**未改的** `pusht_rgb_observability_guard.RgbObservabilityGuard`：warmup0..10一条顺序；目标bind后新的task序列0..5（before）或0..10（after），每primitive前必须fresh observe→dispatch，不能复用旧授权。bind后重启guard因context图像已变且原warmup guard仍ready，绝非跳过动作前观测。guard只看RGB，不拿state代替可见性，不是碰撞/安全控制器。每次guard观测留accepted和完整receipt，row保留失败前的授权下界。它不能预防第一次出视野转换。

native terminal/truncated立即停该分支，无padding、替补或下一probe动作；即使发生probe第5步，保守列native_stopped/end_state=null，实际已返回state/reward/flags保存在last_return及journal，不冒称完整可比响应。hook恢复/close错误分列，不盖原error。保持全部40 row，失败/未启动均null。独立环境仍按预定顺序，不按已见结果筛根。全局wall/存储/journal fault阻止后续分支，close唯一可绕过已耗尽的predispatch预算（不准physics/reset/render绕过）。

## 硬预算、原生hooks及容量

完整成功分项上限：constructor40、close40、reset80、public_step700、physics_step7160、set_state160、goal_setter120、goal_render80、goal_bind40、explicit_render2340。
public700=warmup400+selectedprefix100+probe200。physics7160=public700×10+reset内部160，不能将700称总物理。native_hooks为复用公开context manager，仅临时实例方法hook，finally恢复；不monkeypatch任何旧module globals。physics通过实际space.step入口计数，reset两次_set_state与内部步也计入；goal render和explicit render独立。所有call在callback前durable intent，return后才completed；内部异常保留各轴下界。counts将native physics_step显式映射为probe ABI的internal_physics_step，active键仍按原hook接口，无隐藏开销。

墙钟1800s，含run saved verification、I/O及物理；每actual callback前检查。输出总cap512MiB，final reserve32MiB，diskfloor200MiB，启动要求free>=712MiB。journal**解压前即原始JSONL**累计硬cap64MiB（probe v2同上限）；report8MiB。own prepare64MiB。完整PNG最坏约2340×151KiB≈345MiB，返回info数组分700个lossless NPZ（含goal RGB原精度），journal64MiB、branch receipts及report/reserve在512MiB总额内受实测检查；不是实际产物大小保证，任何超限都属于保留失败，不重试/降精度。实际墙钟及字节/census保存于报告。

Store维护成功或部分写入后的fstat实际bytes，追加O(1)，不每次iterdir。flush/fsync异常也计partial字节；no-overwrite、flat path、O_NOFOLLOW，结束全量name/inode/size/hash census验证漂移/未知文件。所有PNG和step返回info原始数组长期保存，不把224RGB展开成JSON。完整info数组不pickle，NPZ metadata含dtype/shape；所有file SHA在最终artifact census绑定。journal每事件index/previous_sha256/event_sha256，fsync逐事件。最终source/hash/journal/report异常分别保留，首错不覆盖；若最终report无法发布，尽力另写terminal_failure.json，不能称完成报告或科学负结果。

## 新ABI及saved consume

报告schema严格为冻结probe v2`.evidence`，另有executor_schema本stem`.report`；probe prepare SHA与executor prepare SHA字段分离，execution_binding文件及报告一致。40rows/job_digest原顺序，causal_replay_returned逐位绑定warmup10+selectedprefix5，probe_attempted/returned固定raw5，raw_return另保实际latest_action位串、state7、完整snapshot、native奖励/终止/原始info lossless数组。fresh_observation含三图及pre/post。counts含所有实际进入的轴，不伪造未进入资源的0事件。

完整报告的own consume先核外部reportSHA、own verify、全部artifact census及两个prepare/source绑定、journal chain、原动作/真实actual位串与停止后无派发、guard完整序列，再调用冻结probe v2.consume；成功仅为saved_probe_responses_not_admitted。对于尚未进入所有资源轴的partial（例如构造前wall耗尽），own consume核实际hashchain/分母/attemptreturn及cap后输出partial_evidence_preserved_not_admitted，contrasts=null，不给旧probe consume伪造全轴数据。source消失或未能完整发布journal时仍可读原错误产物，但严格consume拒绝认证。

## 测试和解释边界

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_agent_mixedmode_capability_physical_20261028_v1.py
```

RAM fake环境只做数组确定性算术，无native模拟/神经。覆盖40全分母、before15/after20 literal动作、全部native-hook轴、2340PNG、新v2.consume实际接口、hashchain、拒绝授权/cap/wall、异常attempt下界、晚发hash错误不盖首错、lossless info、低层追加/部分写/final未知文件拒收。synthetic通过不等于实际physics通过。

本实验两个known roots，前后时间/速度/接触/cache混杂明确保留，零相对响应不是任务有利方向。终距改善、无符号谱变大都不是能力有利改变。P1/独立确认/equal total compute始终false。下一独占分析器只做任务有符号响应与完整分母，不能据固定库负结果否定理论。
