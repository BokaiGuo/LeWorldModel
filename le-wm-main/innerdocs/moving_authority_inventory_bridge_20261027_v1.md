# 紧凑 catalog → resolver inventory 桥

## 交付与边界

新增独占文件，旧版 catalog/resolver 保留；不修改共享 progress/dispatch/closure。

- 源码：`experiments/spectral_reachability/moving_authority_inventory_bridge_20261027_v1.py`
- 测试：`tests/test_moving_authority_inventory_bridge_20261027_v1.py`
- 本说明：`innerdocs/moving_authority_inventory_bridge_20261027_v1.md`
- 新生成结果：`results/moving_authority_inventory_bridge_20261027_v1.inventory.json`

输入 schema 为 `moving_historical_authority_inventory_20261027_v1.inventory`；
输出严格只有 `schema/files`，schema 为
`root_exclusion_authority_resolution_20261027_v1.inventory`；每项严格只有
`path/sha256/family`，其中 family 逐字复制 family_hint。
全部 618 项原顺序保留，不按成功、失败、selected、unresolved 或 family 筛选。
`original_four_candidate_scope_unverified` 不重命名为 `original_four`。
family_hint 只是路由提示，不是 coverage 批准、物理 lineage、freshness 或 admission 证据。

## Fail-closed 与读取预算

CLI 强制指定 input SHA256，对同一份有界读取字节验证后才投影。
输入上限 2 MiB，输出上限 2 MiB，条目上限 1024，非空字符串上限 4096 字符，
每个已访问对象上限 128 个键；缺失字段拒绝，不填 0、不猜 family。
输入 schema、容器类型、重复元数据键、尾随内容、SHA 格式和输入摘要均检查。
重复 path 一律拒绝，包括相同重复、不同 hash 或不同 family；相同 hash 的不同 path
合法且全部保留。path 仅做绝对、规范词法及项目内路径检查，不调用 resolve/stat/open。
这意味着不验证 evidence 文件是否存在、符号链接是否安全、实际 SHA 是否匹配；
这些不是该桥的覆盖结论，后续 resolver 仍必须自行检查。

仅打开用户明确提供的 catalog；复用既有纯标准库 lexical skip，白名单只解码
schema 和 evidence_files 下的 path/sha256/family_hint。其他子树仅词法跳过，
不是通用 JSON 内容验证器；outcome 值不解码，也不打开任何 evidence 引用文件。
没有导入/执行 resolver、模型或 physics，没有遍历 results。
输出采用 `xb` 排他创建并 fsync，已有文件、悬空 symlink 和并发同名创建均不能覆盖；
不自动创建目录、不删除或清理。I/O 中断可能留下不可覆盖的部分新文件，应使用另一个新路径。

## 验证及实际生成

工作目录：`/home/gbk/code/LeWorldModel/le-wm-main`。
先写测试并确认因尚无实现而失败，再实现；生成真实 inventory 前测试结果为
**22 passed in 0.15s**。测试仅为软件正确性证据，不是科学证据。

```bash
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_moving_authority_inventory_bridge_20261027_v1.py
.venv/bin/python -m experiments.spectral_reachability.moving_authority_inventory_bridge_20261027_v1 --input innerdocs/moving_historical_authority_inventory_20261027_v1.json --input-sha256 c425052a9a43e362eff77befc21e37c6a72fdc3fa455a6f5ac7c53b1c0487341 --output results/moving_authority_inventory_bridge_20261027_v1.inventory.json
sha256sum results/moving_authority_inventory_bridge_20261027_v1.inventory.json
```

实际生成 618 项、153069 bytes。独立 jq 比较输出 files 与原 catalog 的全量
`map({path,sha256,family:.family_hint})`，以及严格顶层 keys/schema/count，返回 true。

- 输入 SHA256：`c425052a9a43e362eff77befc21e37c6a72fdc3fa455a6f5ac7c53b1c0487341`
- 输出 SHA256：`9fc0218a8c39b194abac363ca59bb7877323bdcf882e8aefd54a1c9effb8bd9a`

未执行 resolver；未读取真实 outcome、未训练/adapters/Q 拟合、未执行 CPU 神经实验、
未新增 model/physics、未删除或发布。任何真实实验仍须 MAIN 审核后 HOST CUDA。
