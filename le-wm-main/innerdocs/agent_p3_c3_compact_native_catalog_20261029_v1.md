# C3 compact native → Catalog：可执行 RAM 接线切片

本 NEW 三件套不修改已冻结 codec、routing、原 tree 或结果。它不是可启动 HOST 的生产 runner，没有生产 prepare/run/consume 命令；`production_ready/full_C3_demonstrated/physical_release/neural_executed` 全 false。MAIN 科学合同的 12 槽、1728 请求、27 轴、4GiB（含 512MiB reserve）、7200s/900s、GPU allocated3GiB/reserved4GiB 均未改。

## 实际实现边界

`NativeCatalog(store, catalog).ingest(slot, root, result, events)` 必须接完整原 slot `io(call,payload)` 序列及原 result，而不是选择过的 call 列表。先核原 resource census、完整连续 call IDs、原序连续 invocation、整个 journal 对 transcript 的逐字节对应。对每个调用使用原 `old.audit_native()`；成功方可登记 nominal/J 以及 replay/nominal boundary 导出的完整 context。parent task 必须是封存 root 的原 `parent_task(root,context)` 完整值，且 context 已有前序 root/native 来源，不能靠相同 shape、自称 hash 或自己的后续输出启动来源。

每个 nominal boundary 都从实际 `latent/history` 原 f32 blob 解码回原 full3/between2 literal context；H3 中间边界也保留。成功 replay 的全部 path contexts 来自原 saved validator 核过的 native result。每项绑定有显式 slot/call/boundary 或 ledger index 来源，不将 nominal context 当真实 RGB feedback。

native transcript Catalog 目标是 frozen codec 的 W view，**不是 generic L 列表**。ledger 是全部实际 return/failure/blocked dispositions 的 W view，展开仍按原 entry index 排序。失败 native 调用仅保留完整 prefix/transcript 和实际费用，不登记为 complete native closure，不可作为 bank source capsule。

source capsule 使用 `/__virtual__/agent_p3_c3_compact_native_catalog_20261029_v1/s{slot}_c{call}.json` 的显式非实体 namespace。指针 SHA 是整个原 transcript 的 canonical SHA，不是磁盘文件 SHA。新 indexed reader 精确核 W indices、原完整 transcript、canonical SHA/length 和 `materialized=false`，不为每个 capsule 扫描所有其它调用，不改旧模块/global，也不制造物理 JSON。

`finish_slot(slot,result,events)` 已实际接 frozen `Routing.tree()`，保留全部 144 jobs、原 cap-refused requests、原 progress/R bank、pre-eligibility row、上下文和全部 native IO。native/resource 先占原 journal 的真实位置，auxiliary 位置暂留私有 RAM 空位；完整 tree 路由后，用显式 `CatalogEncoder` 将已认证且字节相同的子值接入既有 DAG/W refs，填回**全部** auxiliary 位置，再生成原顺序 104-byte hashchain。不会把 projection 冒称完整 journal。

`verify_full_journal()` 精确核全序 index/previous/ref/hash、每个 typed row、所有位置均已完成；重新编码不能新增缺失对象。它使用显式 reader/encoder 适配，不 monkeypatch frozen codec。`reconstruct()` 要求 finish 后从 CAS 实际展开完整 auxiliary/native journal 与 result DAG，逐字节核传入的绑定值，再将这些**展开值**交原 `validate_tree()` 做 saved policy/bank/beam/gate/27-axis/whole-segmented 重建，不仅检查 native projection，也不信调用方另一份 result。这是复用原重建，不是第二独立数学实现。构造时另核冻结 codec/routing/common-cap 三个直接 source SHA；该直接 pin 不冒称完整生产 source/class/weights closure。

`RamExecution.run(tasks[2], injected_factory)` 直接执行冻结原 `execute()`，不是另写简化十二槽循环；其 source callback 使用同一 virtual semantic pointer ABI。它仅接受原 `RAM_FIXTURE` callback，保留 12/1728、runtime failure 后其它槽 blocked、cap/no-witness 不 fallback。**此类明确完整保留原 event/result 在 RAM，不能用于 HOST 生产。**

验收分栏：十二槽测试覆盖的是原全部 controlflow 与原 saved reconstruction；新 typed Catalog/原序 auxiliary transport 在真实维度的完整 R 单槽、progress cap 单槽、失败 prefix 单槽分别验证。尚未把这两层冒称为“十二槽全部已经落盘并完成文件级 consume”。

本轮完整 R 单槽 RAM 实测：39 native invocations，38,334 原 events，其中 38,297 native/resource 投影、37 auxiliary；29 nominal 与20 replay context boundaries。实际 object+48-byte index 9,137,128 bytes、全原序 journal 3,986,736 bytes；field/static metadata318,512 bytes，三者合计13,442,376 bytes（RAM 编码长度，不是生产磁盘结果）。完整结构独立 owner census161,325 bytes/764 objects，最大 parent40,455/candidate5,373/slot20,110，均在既定桶内。static typed+metadata657,605 bytes，`source_entries=[]`，因此不声称已测生产全 source seal。此实测不用于平均外推十二槽 fit。

## 预算与仍未完成的生产接线

本切片不是“整个 4GiB 已证明适配”的交付。冻结 codec 已给出的 live 2,930,867,648 bytes 与 reserve 后 3,467,738,560 bytes 不变；这里没有把新增 native generic headers、外部 metadata、pending journal、文件 manifest 的费用偷偷判 0，也没有把 RAM fixture 小输出乘次数当全形状证明。

尚缺的实际工程部分如下，不能启动生产：

1. 当前 `finish_slot` 是 **offline RAM** 接线，已有完整 result/events 后才路由 auxiliary。尚未实现回调运行中的在线 owner 路由及原序 journal 持久化。RAM 空位没有 durable completion，也不允许发布 saved receipt。
2. 尚未把整个十二槽 typed journal、DAG roots、Catalog origin map、capsule descriptors 接到独占 DiskCAS 格式/完整只读文件 consumer。冻结 DiskCAS 本身已有，但本件没有假称“调用它等于完整生产接通”。
3. 原 typed native rows 和 structural routing 已有边界；**native task/request/result 的附加容器、scope/delta、auxiliary fallback、全部 transport index/manifest/completion、长错误 reserve 仍须做独立 owner census 与整数最坏上界**。本件 `coverage.complete_transport_fit_proved=false`。827,228,736-byte 旧 live 余量不是未经证明的免费费用，也不能扩大 4GiB。
4. 尚无本生产 runner 的实际 source/model/class/precision before/after 与最后 report fsync 后 completion 包装。静态 seal 仍复用上一层；RAM tests 的 `sources={}` 与预先 fixture 字段发现都不是生产 closure。

因此本件解决“真实 native → Catalog → 完整 slot auxiliary/journal → 原 saved validator”的可执行 RAM 路径；没有用条件公式替代这条已实现路径，但也没有对尚未实现的整份 transport 作 fit/生产 ready 声明。失败、未知 wall/GPU 峰值都不能冒充科学负结果或准入。

## MAIN 可运行验收

在 `/home/gbk/code/LeWorldModel/le-wm-main`，仅用模块入口：

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q -s tests/test_agent_p3_c3_compact_native_catalog_20261029_v1.py
.venv/bin/python -m experiments.spectral_reachability.agent_p3_c3_compact_native_catalog_20261029_v1 coverage
```

第一条只跑真实维度的 injected RAM callbacks，不读生产 authority、不运行生产 fullstrict/NN/physics；stdout `RAM_NATIVE_CATALOG_NOT_PRODUCTION` 给实际 static、native projection、object/index、journal 与 owner census。测试覆盖真实 R 全事件/child context、完整原 saved reconstruction、K1 progress cap、失败 prefix、不授权失败 bank source、全 12/1728 正常/no-witness/cap 与 runtime block，以及 coherent J/native task/unknown event/orphan/call identity/journal bytes 篡改。没有 HOST CLI；剩余接口在实现并审读前不虚构命令。

最终自身回归实际终态：上述 `pytest -q -s` 命令 **18 passed in 299.90s**，2026-10-29。此前版本17项270.02s不是冻结最终回归；最终版另加强制从CAS读取完整auxiliary/result的消费者边界及相应篡改测试。三件套在此记录后冻结；后续生产接线用NEW stem，不改本件。

接口设计技能用于显式 Catalog provenance、semantic 与物理身份分离、以及 original validator adapter 边界。未改变原 policy、bank、资格 gate 或科学门。
