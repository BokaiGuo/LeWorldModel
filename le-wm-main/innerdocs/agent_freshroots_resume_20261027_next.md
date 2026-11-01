# 独立新根关键路径：返回链与实际 metadata 增量

仅新增本stem文件。未派生agent，未改旧封存/dirty文件、MAIN源码或共享文档；未运行神经、CUDA、模拟、物理、训练、下载。已完整读取experiment-code技能及两份references，按用户的metadata-only范围实现解析与RAM mutation tests，不执行通用训练/优化流程。个人goal仍由旧blocked职责占用，未覆盖/伪标complete；本人完成条件是新消费者、实际saved运行、回归及精确缺口交付，不是P1完成。

## 相对上次的实质新增

上次四文件SHA逐一复核相同；旧coverage SHA `00a3fffeaaa8943e851d129b941a7f07feaa1db387117beb98110e4ec95f5eeb`，没有重新包装旧coverage当新证据。新实现只引用其固定8来源/16 refs、10个dispatch call_id、21段patch索引。

这次实际追到原tool返回和异步session poll，严格以call_id→operation index→session_id连接，而非相邻行猜测。已解决5个具体未知状态请求：

|来源/dispatch|新增实际证据|仍不能推出|
|---|---|---|
|B 第1次，原log22791|返回22794有ValueError `too many values to unpack (expected 4)`及producer traceback|reset计数/全factory-env字节|
|B 第2次，22814|返回22817有RuntimeError `permute(sparse_coo)`维度不符及producer traceback|失败前全部reset尝试ledger|
|A，58279|返回58284分配session71462；poll58298→58301明确exit_code0|原run完整source closure|
|E，58307|返回58309分配session29686；poll58316→58319明确exit_code0|原run完整source closure|
|H，58363|返回58367分配session42594；poll58374→58378明确exit_code0|原run完整source closure|

行号均在第一份旧coverage绑定的原会话（Oct23/01a16903-c194…jsonl）。另B最终/C/D/F/G有对应stdout返回，但旧调用只打印r.output，未保存exit_code；不可把外层`Script completed`当内部进程exit0。保存traceback仅取异常行与P0/P1 producer帧，不读取native outcome或成本来选根。

新增13条dispatch/poll收据、15条patch-tool返回收据，匹配无悬空session/arity歧义。patch返回往往只是`{}`，它并不是原加载字节的SHA收据，不能借此消掉producer运行绑定。故**5个原子状态请求解决，不等于5个完整reset blocker解除；原16 refs仍16未认证**。

另实际发现第二份旧会话有append。scope冻结时旧227482902字节前缀SHA仍为 `cad9cf6253c2b466fcc48bddda725368426e796f8337326733de175db8b5ee22`；新增3052250字节单独SHA `77bb8f098e921d97e0b2e517bda87aae27c398f83f8d381ee68683e8beb7b229`，捕获范围终点230535152字节。按固定字节范围读取，后续append不会漂移旧前缀或冒称已处理；前缀改写必拒绝。第一份508228945字节原log完整SHA未变。

## 不是“新目录=新来源”的历史delta

读取任何delta内容投影前，先按纯文件规则冻结scope：literal regular `results/**/*.json`，文件名是report/sourcepack/plan/protocol或含prepare/request；减去原SHA固定的618 paths，排除自身stem。没有按状态、结果、日期字符串、成功率或根难度筛文件。目录名本身不被当成未知origin或认证依据；同名`.json`目录不是JSON文件，不混入文件分母。

实际固定1033 files/949956244 bytes，见 `results/agent_freshroots_resume_20261027_next_scope.json`，SHA `51e4b272109d9e5a74d3f8f2ab211e4e477196bd87d034588966375eadb56cd1`。它是**原清单之外的有界路径差集**，含早期遗漏及后续新产物，不称1033项全部为时间上新增，更不称全球完整历史。新建文件须下一次显式scope delta；不是每次resolve追着live目录扩大。

resolver实际逐文件重SHA并白名单投影schema/环境声明、root/reset seeds、声明身份SHA、reset seed span及literal path+SHA父绑定；不解析Q、reward、success、轨迹或候选分数。已声明预留seed/span保留为`declared_not_observed`，不当实际执行。无origin字段的衍生分析报告标`no_origin_fields_in_whitelist`，不会逐目录永久判origin unknown；但字段缺失也不证明该文件不来自历史根。

gzip_base64请求允许有界单member无尾随解压，再独立投影metadata；保留`/decoded_envelope/…`指针，原文件SHA绑定解压字节。仅解析metadata不冒称执行producer-specific prepare认证。每文件128MiB、原delta总1GiB、JSONL行8MiB、单次inflate16MiB/总256MiB、entries4096、输出8MiB。错误/变化/超限保留原分母和原path/SHA，行标`not_verified`且无认证字段，不静默删项。尚未读取的非本规则文件、其他机器/会话、未保存失败root仍不在认证范围。

## 最小剩余证据：缺哪份资产

A–H的精确原目录/report SHA均继承旧coverage `sources[]`，新输出逐来源再次携带原report SHA。不是要求重跑模型或重新生成旧root：那不能补历史事实。

1. **原运行加载闭包或等价穷尽reset证书**。原8个目录（A/B/C/D/F/G只有report；E另有postrun_implementation，H由后续physical seal引用）缺的是各自**生成该report时**的producer/P0依赖、4层factory和installed PushT env/wrapper加载SHA绑定。不是缺“当前文件”或再要一个Git快照。4层已有候选blob为 `audit_pusht_reachability_horizon.py`、`run_pusht_mode_aware_shared_candidate.py`、`run_pusht_signed_torque_audit.py`、`run_causal_visual_pusht_world_model.py`，原SHA在旧coverage historical_chain。原env候选路径是 `.venv/lib/python3.12/site-packages/stable_worldmodel/envs/pusht/env.py` 及其wrapper/import closure；当前版本/mtime/包名不证明旧版本。需要原prerun/source-load manifest、容器/环境可追溯快照或相同可信强度的原执行记录，不能用后续seal倒灌。
2. **尝试完整性**。B必须涵盖22791/22814/22850三次dispatch，不只是最后报告；其余至少各1次已定位dispatch，另有未进入这两份会话的restart仍未知。需要原reset-attempt journal，或由实际加载闭包推导且独立有scope证据的穷尽有限包络，包含失败/未选/拒绝候选和seed0显式state重建。参数相同/最终root_count不能代替。新返回链无需再索取：A/E/H退出及B两次异常已经有本地原记录。
3. **全局范围**。需要有责任边界的完整历史scope：哪些机器/工作树/线程/时间范围及未写出report的进程在内，旧618之外的捕获delta与剩余文件/会话如何覆盖。本轮新增1033 paths和一段会话append可交MAIN合并检查，但绝不自动变成global complete。
4. **候选身份**。旧预声明四槽的实际state7/wrapped-pose/snapshot/RGB3及历史/批内排重尚不存在；不得用新的seed、goal、velocity、renderer差异代替新物理根。新输出没有进行采集或发放运行授权。

若(1)/(2)确已无法恢复，应明确标记这8个原run的历史执行排除证据不可恢复，而不是用“未找到记录”推新鲜。原认证门槛保持关闭。

## 能否事前随机封存新的采集协议？

可以封存，但它解决的是**事前固定选择/随机性**，不自动解决**历史不重复认证**。原四槽protocol/prereg已经事前固定；另开新协议也必须保留有限seed/proposal/reset span、固定分母、全尝试durable journal、因果RGB3 bootstrap、native earlystop/failure无替补、原物理identity排除及未来主线程执行权。不读取outcome调池/调阈值。

要“充分证明与全部历史根不同”，仍需完整旧身份清单或可证明旧历史域与新域严格不交的历史字节证据。当前缺旧factory/env/全scope，故没有后一个证明。有限精度PRNG的新seed不是物理状态不交证明；不能用连续随机碰撞概率近零替代现行显式freshness门槛。可按单独授权进行**calibration exploration且global freshness unknown**的有界采集，但不得把它计作独立确认/P1；本实现不自动授权此采集。若原记录不可恢复，任何改变确认scope/证据要求必须另行明示研究协议变更，不能偷偷在resolver里放宽。

MAIN最新24suffix结果中teacher_full原生1/8仍是known-root校准，不纳入本工具选根。PushT原成功位置同时包含agent/object，object-only投影非正不能代替完整任务失败；本工具不读取此类outcome作排除决定。

## 实际执行与交接CLI

RAM回归含返回链、字段mutation及无损大输出封装测试；原调用链的实际metadata运行已取得上述5个状态闭包。最终delta实际运行/verify状态见交付回报，不预填全球认证或fresh候选。

第一次全量resolve完成1033文件投影后触及8MiB输出上限，未保存假成功。最终版本保留8MiB落盘上限，超过时仅将完整JSON结果作确定性gzip+base64无损envelope，原semantic SHA/长度全绑定，解压上限64MiB、单member无尾随；verify重算并逐字比较解包语义。不删任何原文件行/绑定/收据来适配大小。此封装没有扩大输入范围或改变认证门槛。

实际大闭包prepare还触及旧8192字段行上限；最终按明确65536行内字段上限完整投影，并以9000条literal绑定RAM回归防止丢弃。其余输入/输出/语义总量上限保持；这不是放宽freshness门槛。最终测试46 passed。

工作目录 `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_freshroots_resume_20261027_next.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_freshroots_resume_20261027_next resolve --scope results/agent_freshroots_resume_20261027_next_scope.json --scope-sha256 51e4b272109d9e5a74d3f8f2ab211e4e477196bd87d034588966375eadb56cd1
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_freshroots_resume_20261027_next verify --scope results/agent_freshroots_resume_20261027_next_scope.json --scope-sha256 51e4b272109d9e5a74d3f8f2ab211e4e477196bd87d034588966375eadb56cd1 --saved results/agent_freshroots_resume_20261027_next_delta.json --sha256 SHA_FROM_HANDOFF
```

capture-scope仅在MAIN明确要收集下一段delta时调用，stdout JSON由MAIN封存；它不覆盖现有scope。resolve/verify只读并输出stdout，metadata进度走stderr。自有source/tests/doc与全部原path/SHA仍绑定；没有新的GPU/physics命令。测试、返回链和metadata增量均不是P1科学准入。
