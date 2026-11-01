# 有符号真实响应与同动作约束 beam 对照

## 真实有符号响应：已经独立复算

分析 `results/agent_mixedmode_capability_analysis_20261028_v1_MAIN01.json` SHA `9f1cf0f84f8199ec5336ff9ef7b4a24293d99534c37296c136b9fe4ec4d2389d`。MAIN启动的完整consume/分析算术正常走完，但排他写入发现该并行产物已存在，抛FileExistsError；未覆盖、未当作科学失败。其来源已向工作者请求核对，来源归属未定前不冒称MAIN首次生成。

MAIN另以不导入分析器的独立源码，从自己先前严格验收的40端点与原native root/goal重算160有符号投影、64有限secant及全部前后差值；全数一致。回执 `results/audit_mixedmode_signed_MAIN_20261028_v1_MAIN01.json` SHA `30ac14f7cce46d4116980b79a88df0c03e6b6f1550e15e4d09f01a2bece7c085`，并核分析源码/tests/doc真实字节与产物绑定相同。

| 原任务条件 | jointXY 有符号 x secant 前后差 | y secant 前后差 |
| --- | ---: | ---: |
| root0 off | -40.76523 | +0.56532 |
| root0 on | -41.73227 | -2.54333 |
| root1 off | +0.64856 | +57.95138 |
| root1 on | -0.59125 | +56.01570 |

单位为px/持续五步raw命令幅度；共同原root→goal方向固定。列符号不是选中最优动作，不用单列大小判所有可达方向优劣，不是无穷小导数、终距改善、成功率或因果能力收益。两个known根、四条件；before10步vsafter15步存在时间/状态混杂，不能称理论负结果。

下一最小可辨识干预已实际派给Poincare：同4任务×5probe的20分支zero5 matched-time control，新stem `agent_mixedmode_zero_time_control_20261028_v1`。真正派发warm10→zero5→probe5，不跳zero、不重置saved poststate；不新增幅度/方向搜索。预期400public/40reset/4080内部physics/1320render，待实现和MAIN审读，不提前宣称完成或科学准入。

## 同动作约束kinodynamic beam：实际HOST CUDA与严格验收完成

MAIN完整审读producer/doc，25项测试通过。实际prepare SHA `de3db628c5bd15f62f700fa56c917582d5e9fa27cf3cb979abf25a181d5e8dc9`。实际HOST CUDA report `results/agent_kinodynamic_matched_domain_20261028_v1_cuda_MAIN01/report.json` SHA `d4976aeca357b5324c74be8b2b7e3bfc254e82b834b3acfe1b0ab90a1ef4614c`，288predict API/sample、errors=[]、模型哈希不变；两根各144，20首层/beam6扩120次/完整与分段重放各2。

MAIN严格consume及280node的独立Q/frontier/argmin复算完成。持久回执 `results/consume_kinodynamic_MAIN_20261028_v1_MAIN01.json` SHA `f4c5b74b6455c62986f23aafa94ef2e3793ff9b8c3c45c6d7c107395a2c13aca`；两根选路径[12,11]/[17,19]，模型终点Q 0.57598809/0.66564188。不是RRT、不是官方论文忠实复现、不是等FLOPs/总资源，当前无这两条动作的物理/RGB证据。相同动作球和predict轴可对齐已有matched sampling；未作真实任务优劣判断。

首次手工consume参数SHA误填导致external hash拒绝，随后依据sha256sum的实际report SHA执行持久严格consumer成功；未改输入/门槛、未重跑模型，此类命令错误不是科学负结果。

Jason已收到新独占物理adapter任务，严格绑定原根recipe/history/goal和新producer arm，不冒用旧whitelist。P2组合执行接线仍需后续交付。

## P3固定5步后续

初次联合回归3失败/179通过，未放行物理。MAIN定位到失败receipt语义测试、RAM prepare缺arguments、source closure显式缺fixed源码；工作者修正后MAIN实际82项physical回归通过。源码/文档完整再审，物理prepare正在session76149，尚无新物理结果。第一次相对output被canonical absolute path保护拒绝、无输出；改用确切绝对路径后继续，未放宽路径检查。

### 实际 prepare 终态与根因（后续主线程核实）

session76149 已终止于 `literal original cold task/contract/blind slot identity`，未写物理 prepare、未执行物理。MAIN逐项复查发现只有 `fixed.digest(e['task_contract'])` 不匹配。原编码器的 `codec.core.digest` 使用 `json.dumps(sort_keys=True,allow_nan=False)` 默认空格，runtime 的 `fixed.digest` 使用紧凑 separators；不能混用两个合同的哈希方言。

两根使用原编码器函数分别精确恢复权威 `3e7a1b9da62918ca73610f222f836b447e7ea1610295f587e5d3ed00e9ef549f`、`c89d987170c5d05340cb4bd4702c78cc387f7df0ee6ff9ed6d06c03e99867252`，错误 runtime 哈希则为 `78fa247c16c8427e27644e0e26bd5780faa4667a71fa593e0d25bcde1f7b8efb`、`5d8f04aa24a8d8d4bd919a7bbea1ec82da0575bf6e80cc77d8273fc563899a03`。这是可复现的代码接线错误，不是任务身份漂移或科学负结果。旧v1冻结；已派独占v2最小修复，仅在原合同边界使用原函数，其他runtime seals不改，须补真实封存合同回归后再实际执行。

原P1/P2/P3/P4/P5科学闭环、独立泛化、强对照物理及完整三实验族仍未完成；goal保持active，不把以上工程与局部模型证据扩大为完成。
