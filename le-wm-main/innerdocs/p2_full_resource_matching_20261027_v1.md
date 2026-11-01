# P2 完整资源向量与有用工作预算接口 v1

仅NEW `p2_full_resource_matching_20261027_v1.py`、对应tests及本文；不修改sealed planner/consumer/结果，不读取物理评分来分配预算，不执行CUDA、physics或CPU神经。模块仅标准库，测试仅算术mock及真实saved resource receipts；没有模型或环境依赖导入。按api-and-interface-design技能固定claim标签、边界校验与unknown语义，避免下游把有限预测计数提升为全算力认证。

## 已实现的接口

- `vector/validate/add`：完整版本化向量，逐轴非负literal int或null；bool/浮点/未知字段拒绝。null绝不转0；zero仅表示声明工作scope内不存在该项。AD按192×20和192×10分开，同时给出数学VJP方向与Jacobian元素数；后两者**不是**backward API次数或FLOPs。SVD按矩阵shape，bounded solve按变量宽度分别计数。peak bytes取max而非相加；其他disjoint work才可相加。
- `sum_disjoint`：共享work_id重复拒绝。shared只入paired unique一次；standalone attribution分别包含shared，不能把两臂standalone相加当实际运行总数。
- `compare/validate_claim`：`predict_frontier_equal`仅检查predict/action_encoder的calls、samples和context_tokens；`full_resource_equal`要求**全部**跟踪轴已知且相等。已知不等或任一null都不能通过；报告仍不认证硬件成本。即使两边unknown相同，也不是相等。168/168误标为full直接抛ValueError。
- `candidate_recipe/allocate`：mandatory root compile＋winner replay＋本臂endpoint/warmup固定项，配上单个有用candidate单位，逐轴整除取最大可行候选数。没有空跑、padding、退款、缩短replay或省略endpoint。conditional SVD/solve单位为positive-radius的安全上界，未来zero-radius需记录实际0，不能把cap当实耗。
- `BudgetLedger.reserve/finish/receipt`：绑定frozen root/task/model/protocol SHA；只接受有用purpose及唯一work_id，超cap在reservation前拒绝。完成量可小于reservation，以保存部分失败；reservation永不退款，重复finish/超额完成拒绝，pending保留。**此处是纯RAM算术适配器，不是已接入GPU的durable runtime guard**；未来runner必须先durable journal再dispatch，并为每项primitive真实调用提供terminal evidence。
- `audit_current`：固定实际planner/physical SHA，重验原prepare/execution/source archive/current source、root row digest、bank/journal哈希；从真实request/terminal重新计数，而不只信任summary。AD计数来自实际operation记录；SVD/solve shape进一步由封存源合同和保存的192×20/192×10 Jacobian绑定，shape不是新增profiler测量。逐root稳定身份保留shared/exclusive/standalone，物理collector在cohort级共享，不武断平均到每个root。

## 真实80/40：168/168究竟遗漏什么

输入planner report SHA `4023a0e3896a90407188d260a45dd334e0802ba17ecd59ef62ea931603eba173`；physical report SHA `9144c3f25f2880af62d94dd88257918fa508b5ddbea51d98e9ccbc3d3b7532e2`。均为已有实际main执行，本模块未重跑。

| 每根standalone归属 | joint80 | conditional40 |
|---|---:|---:|
| predict calls / samples | 168 / 168 | 168 / 168 |
| action_encoder calls / samples | 168 / 168 | 168 / 168 |
| predict / action_encoder context tokens | 504 / 504 | 504 / 504 |
| 192×20 Jacobian calls | 1 | 1 |
| 192×10 Jacobian calls | 0 | 40 |
| 数学scalar-output VJP方向总数 | 192 | 7872 |
| Jacobian标量元素总数（非FLOPs） | 3840 | 80640 |
| SVD 192×20 / 192×10 | 2 / 0 | 2 / 40 |
| bounded solve width20 / width10 | 1 / 0 | 1 / 40 |
| 原Q candidate selection evaluations | 80 | 40 |
| 本臂新终点visual_encoder / projector | 1 / 1 | 1 / 1 |
| 本臂selected / warmup公开step | 10 / 15 | 10 / 15 |
| 本臂warmup reset | 1 | 1 |

shared root compile为每根4predict＋4action_encoder、1次192×20 Jacobian、2次SVD和1次solve。每臂exclusive有164predict＋164action_encoder；conditional额外40次suffix AD/SVD/solve。实际所有160个suffix均positive-radius，不能推广成未来必然160次solve。

四根paired unique是1328predict＋1328action_encoder、4次192×20＋160次192×10 Jacobian、768＋30720=31488数学VJP方向、8＋160=168次SVD、4＋160=164次solve。两臂standalone各672predict；相加1344会重复计shared的16predict。物理执行实际200公开steps=80selected＋120warmup，32collector resets＋8warmup resets；终点visual_encoder/projector各8。cohort collection的32resets全体共享，只记一次，不在root之间假分摊。standalone再分别归属该shared项时也不能求和当unique。

## 为什么严格equal vector没有成立

首先，已知维度就不等：192×10 AD、对应数学VJP／SVD／solve、selection evaluations不同。其次，backward engine API次数、CUDA kernel次数、FLOPs、统一同步device时间、peak GPU bytes、验证CPU成本、历史context构造和physics私有substeps未被这一版本完整分臂测量，均保留null。保存root总wall time包含序列化、共享compile等，不能倒算成每臂公平device预算。

192和7872是数学输出方向，不能声称41倍运行时间／FLOPs；SVD次数也不能无视shape直接折为某个predict倍数。模块没有经验权重或任意换算系数。source hash绑定和实际receipt计数不等于独立硬件profiler证明。

对**当前不变**的候选算法族，设joint候选数j、conditional候选数c。standalone forward为`8+2j`和`8+4c`；相等要求j=2c。但192×10 derivative histogram分别恒0和c，相等又要求c=0。因此没有j,c≥1的严格equal-vector配置；selection evaluation轴还要求j=c，是另一个冲突。接口给出该infeasibility certificate，而不是建议joint空做40个AD/SVD来造相等。这个结论仅限当前family，不排除另立有用gradient baseline的新版本。

## 最小下次协议：共同cap，不冒充实际相等

`--next-protocol`可立即生成纯JSON建议：每根取当前80/40两个有用recipe的componentwise最大值作共同cap。mandatory固定工作各8predict（4compile＋4replay），共同168predict及对应encoder/token cap；192×20 AD上限1，192×10 AD上限40，数学VJP相应上限192/7680，SVD上限2/40，solve上限1/40，selection cap80。每臂完整H2＋15warmup＋1次新encoder/projector，collector为cohort shared另行计数。分配器在该cap下返回joint80、conditional40，joint未用AD额度明确为40，**不补齐、不转借、不声称相等实耗**。

该建议只是可执行资源设计，不授权新实验或method选择。下一runner最小要做：

1. main独立冻结新protocol SHA、root/task/Q/goal/model/raw域/normalizer/H2/backend/hardware、种子与有用算法配置；保持无outcome调参。新baseline若用剩余额度改进候选，须独立新version，不能追认旧joint。
2. 每个forward、derivative（input/output维度、dtype、vectorize/backend）、SVD/solve（shape/radius）、反馈／endpoint编码和公开physics调用逐项predispatch reservation＋durable attempt/terminal；失败和部分完成不退费。共享项按work_id去重，runtime实际量和cap分表。
3. 若要比较完整hardware cost，预先声明同步计时范围／hardware控制及profiler hook，区分model-only和end-to-end；记录backward API、kernel、FLOPs测量方法、peak及CPU传输／验证成本。没有测量的项继续unknown，不因common cap而置0。缓存输入历史成本须共同来源绑定并单列，不能称免费全流程。
4. 只能先称**componentwise-cap-matched useful-work frontier**，报告实际向量与任务后果。若坚持严格相等实耗，应先找到自然产生相同完整向量的有用算法配置；当前80/40 family无解。P1前置、独立确认根与P2机制／效能准入不由本接口替代。

## 测试与命令

tests包含维度混淆、168/168误标、unknown!=zero、SVD/solve/编码／物理各轴差异、shared重复、zero-radius实际0、reservation失败不退费、partial completion、无padding预算分配及真实saved账复算。mock通过只证明接口；saved-data通过只证明已保存计账一致，不是新神经或新physics，更不是P2成功。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_p2_full_resource_matching_20261027_v1.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m experiments.spectral_reachability.p2_full_resource_matching_20261027_v1 --audit-current --output results/p2_full_resource_matching_20261027_v1_audit01.json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m experiments.spectral_reachability.p2_full_resource_matching_20261027_v1 --next-protocol
```

输出文件用exclusive create，存在即拒绝；保留200MiB，不复制上游大报告／权重／图片，不触发后续作业。只需数十KiB新JSON和小源码文档。
