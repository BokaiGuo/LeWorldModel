# Saved capture 来源—身份元数据关联闭包

## 独占交付与输入

- 源：`experiments/spectral_reachability/moving_authority_identity_link_closure_20261027_v1.py`
- 测试：`tests/test_moving_authority_identity_link_closure_20261027_v1.py`
- 本文：`innerdocs/moving_authority_identity_link_closure_20261027_v1.md`
- 新报告：`results/moving_authority_identity_link_closure_20261027_v1_MAIN01.json`

唯一真实数据输入是 MAIN 已生成的
`results/root_exclusion_authority_resolution_20261027_v1_inventory618_MAIN01.json`。
输入 SHA256：`e2afed12a1d8f788ad16d0223482377b5f54d281e67739766475228b53ac2591`。
不执行 resolver、不扫描目录、不打开报告中的任何 evidence、catalog、inventory 或 protocol 引用。
全部旧文件保留，不改共享 progress/dispatch/closure 文档。

## 字段分析与证据规则

输入 schema：`root_exclusion_authority_resolution_20261027_v1.resolution_request`。
618 个 history_file_requests 均有 path/SHA、projection、root_identity_projections、
input_lineage_links、inherited_refs、blockers、status、hash_authority、inventory_family。
原 capture summary 为 618 captured / 344 metadata projected / 199 direct identity / 123 linked sources。

398 条 input_lineage_links 的 pointer 分布：bindings 319、plans_bindings 16、
protocol/bindings 59、source_bindings 4。
原 resolver 链接依据投影值与任一文件 path 或 SHA 的相等性匹配；不能直接把全部链接当作
同一来源的 path/SHA 双重绑定。本实现仅接受以上四种路径—SHA 字典容器下的单层显式键。

一条有效边要求：

1. 来源已捕获、path 唯一、SHA 合法。
2. 原 link relation 为 `explicit_input_binding_not_physical_root_equivalence`。
3. pointer 在该来源的 projection 中恰好存在一次。
4. RFC6901 解码后的字典键 **等于 target_path**。
5. 投影值 **等于 target_sha256，且等于目标节点已捕获 SHA**。

实际有 370 条满足条件。另 28 条为内容 hash 相同、键 path 不同的别名候选：
标记 `unknown/path_alias_not_explicit_same_path`，不追踪、不把内容别名变成同一来源。
缺目标、缺 captured link、不能证明的绑定保持 unknown；不从文件名、目录邻近、
共同 seed/root hash 或结果相似性补边，不自行建立反向边。
projection 中虽有 path/SHA、但 capture 没有对应 link 的条目只列入 unlinked_bindings。
外部源码/模型绑定也只作为未捕获目标描述，不打开或导入。

## 图、循环、冲突与 unknown

来源节点保留 path/SHA 和 **原文件自己的** direct_identity_projections。
身份字段限定 root_seed/reset_seed/root_sha256/root_goal_sha256/task_id/episode_id/blind_sha256，
必须逐 pointer 与原 projection 一致；不把上游身份字段拷贝到消费者的直接身份里。
每个来源到含身份元数据上游文件的 association，提供 target_path/target_sha256 和
有向 witness_edge_ids；沿这些边可复核原 pointer、path、SHA。每个目标只保存一条
BFS 最短证据链，不枚举指数条数的所有路径。

环检测采用可达子图拓扑检查。自环、互环或可达上游环都会使该来源的 associations
整体为空，状态 blocked_graph；不影响不连通的有效分支。错误 relation、缺投影、
hash 矛盾等 edge conflict 同样阻止该来源及所有能到达它的消费者产出 associations。
缺节点和内容别名为 unknown，不冒充有效边。输入节点 path 重复、投影 pointer 重复、
身份不一致、历史 SHA 矛盾、reset-bound 计数不符等直接拒绝整份转换，不生成输出。
多个上游不同 root/seed 可以合法共存，因为这是多个文件的元数据集合，不是物理等价声明。

原 blockers、inherited_refs、authority_requests、summary 全部保留到对应输出字段；
缺失值不填 0。即使有上游关联，也不撤销无直接身份阻断，不认定 root index/slot 对应，
不认定 reset attempt 总界或覆盖完整。

## 真实生成结果

输出 schema：`moving_authority_identity_link_closure_20261027_v1.metadata_link_closure`。

| 项目 | 数量 |
| --- | ---: |
| 来源 / 原直接身份来源 | 618 / 199 |
| 原链接 / 有效双绑定边 / 别名 unknown | 398 / 370 / 28 |
| conflict edges / graph-blocked sources | 0 / 0 |
| 有上游身份元数据关联的来源 / associations | 120 / 278 |
| 无上游身份关联的来源（可能已有直接身份） | 498 |
| 原 inherited refs 完整保留 | 960 |
| 原缺直接身份 inherited refs（仍未消除） | 614 |
| 上述 614 中有上游关联 / 无上游关联 | 120 / 494 |
| 原 reset-bound 阻断 refs 完整保留 | 16 |

注意：120 个有上游关联的来源，与 120 条有上游关联但无直接身份的 inherited refs，
是两个不同统计口径；不能互换，更不能称为 120 条物理身份已解决。
原直接身份仍为 199，原缺直接身份 refs 仍为 614；16 条旧 reset-bound 阻断仍在。

报告 4425932 bytes，SHA256：
`7c7e1792c5a17986aaf3a38539a280ec46d5910eb4d9351babbe9bf2a70b3576`。

physical_root_equivalence_proved、reset_attempt_bound_certified、MAIN_coverage_approval、
freshness_certified、P1_admission、historical_unknowns_discharged 全为 false。
这里的 closure 仅指已明确输入边上的有向元数据可达性，**不是历史 authority/freshness 闭合**。

## 测试、命令与限制

工作目录：`/home/gbk/code/LeWorldModel/le-wm-main`。

```bash
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_moving_authority_identity_link_closure_20261027_v1.py tests/test_moving_authority_inventory_bridge_20261027_v1.py
.venv/bin/python -m experiments.spectral_reachability.moving_authority_identity_link_closure_20261027_v1 --input results/root_exclusion_authority_resolution_20261027_v1_inventory618_MAIN01.json --input-sha256 e2afed12a1d8f788ad16d0223482377b5f54d281e67739766475228b53ac2591 --output results/moving_authority_identity_link_closure_20261027_v1_MAIN01.json
sha256sum results/moving_authority_identity_link_closure_20261027_v1_MAIN01.json
```

先写测试、确认 RED 再实现，新增 29 tests，联合旧桥 22 tests 为 **51 passed in 0.27s**。
包括 RAM 三节点链/别名/缺边/多身份、自环互环及传播、独立分支、冲突传播、缺字段、
16 阻断保留、旧 authority 保留、输入 SHA、预算、outcome 不解码、重复 JSON key、
已有输出/悬空 symlink 禁止覆盖等。软件测试不是科学证据。
生成后另用独立标准库检查（不导入实现、不打开 evidence）逐条核对 278 条 witness 链，
并比较全部来源的 SHA、直接投影、blockers、960 refs、16 旧阻断及 authority/summary，PASS。

输入上限 8 MiB、输出 16 MiB、来源 1024、边 8192、association 40000；
白名单投影结构访问上限 100000、深度 16、数组 40000、字符串 4096。
只复用已有纯标准库 lexical skipping，未解码非白名单 outcome 子树；
不是对被跳过载荷的完整 JSON 合法性验证。
CLI 强制 input SHA，对单次有界读取的同一份字节验证；输出仅 `xb` 排他新建并 fsync，
不创建目录、不覆盖、不删除。写入中断可能留下部分新文件，需选择其他新路径，不能覆盖重试。
已有 MAIN01 结果存在时上方生成命令将拒绝；复跑必须使用另一个全新输出路径。

未训练、未 adapters/Q 拟合、未 CPU 神经实验、未引入/执行 model/physics；
无删除或发布。任何真实实验仍须 MAIN 审核后 HOST CUDA。
