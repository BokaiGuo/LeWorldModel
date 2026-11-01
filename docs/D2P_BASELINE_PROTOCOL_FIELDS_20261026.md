# D2P reported-baseline 精确协议字段建议（2026-10-26）

这是文档合同建议，未实现、未迁移记录、未运行实验。目标是防止表面相同的 H、±、视觉输入或 frozen 标签造成伪可比性，不增加科学通过声明。P1仍未通过；预算路线B0–B3与研究阶段P0–P5独立。

## 当前注册器事实与证据边界

只读核对 `le-wm-main/experiments/spectral_reachability/d2p_reported_baselines.py`：`validate_record`及reference验证要求精确字段集合；`horizon`只接受正整数或 `unknown`，`n_seeds`同样如此；元数据拒绝null、bool、空集合；分类按字面JSON相等，不推断单位换算。仅允许 `paper_reported`，禁止pooling及配对区间。因此不能直接新增顶层字段，也不能把goal offset硬塞进horizon。当前 `direct_comparable`只是元数据分类，不证明复现、配对或等算力。

已核验示例沿用现有文档的一手材料，不新增联网：

- HWM：`D2P_CROSS_MODEL_BASELINE_ASSET_AUDIT_20261026.md`，原文 [2604.03208v2](https://arxiv.org/html/2604.03208v2)，§3.2 Table2的PushT d=25/50/75是起终轨迹间隔；附录Tables11/12另给flat h=15、hier高低层各5。PushT含RGB及agent位置/速度；高层额外学习，低层encoder冻结不代表全系统training-free。PDF页码、评价seed及误差定义仍未核实。
- PlanWAM：`D2P_REPORTED_BASELINES_LEWM_PLANWAM_20261026.md`，[2610.11382v1](https://arxiv.org/html/2610.11382v1)，§4 Tables1–3及附录C。64条候选、每条8waypoints/4秒，不是本地8个world-model tokens。HUGSIM 436场景按难度人数80/157/96/103加权；不是四难度等权均值或436训练seeds。DINOv2基础冻结但LoRA和规划模块训练。
- SAGE表1/18的SD/pooled、AP-CEM/AP-rank的memory/action权限、Frozen HWM的SEM定义：现有专属审计尚未得到足够原文，均只作待核验字段需求，**不是已确认论文事实或数值**。不得用附件填注册器。LeWM Table10也保持未逐项核读状态。

## 建议统一叶子与来源

新扩展字段使用 `status=verified|unknown|unverified|not_applicable`；verified才有可用于比较的value，not_applicable须理由及证据；未报告/未取得不可编码成0或false。每个事实绑定 `source_url/version/locator/source_kind`，locator为已读section/table/page，PDF页未读就标unknown；source_kind区分原文、官方项目页、本地源码、附件线索。冲突保留两条来源并阻断自动匹配，不择优覆盖。实数要求有限且拒绝complex/bool；枚举的unknown不可视作相等已知值。

## 时间、动作与反馈

| 建议字段 | 精确语义/约束 |
| --- | --- |
| `temporal.goal_offset` | value、unit（source_frame/env_step/second）、起终索引定义、轨迹采样间隔；不等于预测长度。范围/分层分别记录，不取均值代替 |
| `temporal.task_distance` | maze最短路径等非时间难度量；unit与计算规则；不得与goal offset合并 |
| `temporal.prediction_levels[]` | 每层level、horizon、unit（model_token/macro_token/second）、每token对应primitive steps及stride、是否重叠、终态定义；多层禁止压成一个H |
| `temporal.execution` | 每次执行primitive steps、replan interval、feedback timestamps/modalities、总步数上限及停止规则；open-loop与chunk-MPC分开 |
| `action_contract` | raw/normalized/atanh坐标、shape/layout、bounds、normalizer来源、packet长度、repeat和实际动作映射。参数box不等于tanh后的物理box |
| `resource_contract` | candidates/update、iterations、elite、initial/nominal重评分；实际predict calls/samples/token-work、encoder、backward、墙钟/硬件/并发及缓存。计划上限和实际测量分开 |

仅当token长度、repeat与时间单位均有依据才计算派生执行长度，并保留公式与来源；不得默认相乘（层级宏动作可能重叠）。HWM的d75和flat h15必须分别记录；PlanWAM4秒轨迹、8点及训练未来帧1/2/3/4秒分别记录，不能据此换算本地Reacher horizon。

## 统计：SD、SEM与pooled不是同一字段

| 建议字段 | 必须保存的内容 |
| --- | --- |
| `statistics.estimand` | metric、unit/scale、direction、成功predicate、分母及失败/超时处理；percentage不自动等于binary success |
| `statistics.point_estimator` | mean_of_seed_means / pooled_success_fraction / scene_weighted_mean等；权重、分子分母、是否舍入；无原始值不反推 |
| `statistics.uncertainty` | kind=SD/SEM/CI/none_reported/unknown；value、scale、center、跨何种unit、n、ddof、CI level/method；none_reported只表示已读表没有报告，不是方差0 |
| `statistics.replication` | 独立单位root/source/manifest/backbone_seed/head_init/query/scene；各层数量、ID可得性、nested/crossed关系及共享数据；训练seed与评估seed分列 |
| `statistics.aggregation` | 先head再backbone或先query后manifest等顺序，权重和排除规则；配对ID及其来源另存，不由同seed数量推断 |

SD→SEM只在独立单位、n、聚合方法均核实且转换合理时另存derived值，保留原报告值；本任务不做转换。3×3嵌套重复不能当9个独立backbones。pooled 150 queries不能套另一个表的SD，相关query也不能自动计算独立二项区间。PlanWAM场景权重为已核验正例；SAGE/冻结HWM的具体SD/SEM定义仍unverified。不跨论文pooling、不从舍入百分比制造配对推断。

## Memory/state与额外训练权限

| 建议字段 | 精确记录与比较规则 |
| --- | --- |
| `access.entries[]` | 每条resource=RGB/history/proprio/physical_state/goal_set/memory_observation/memory_action/language等；phase=train/plan/replan/evaluate；permission=allowed/forbidden/unknown；实际使用与允许访问分列 |
| `access.memory` | 来源dataset/split/trajectory IDs与评价根去重；容量、检索索引/规则、是否读存储动作、是否在线追加、构建/检索成本；没有memory不等于memory字段缺失 |
| `access.state` | 位置/速度/姿态/物理goal来源、调用方、是否仅evaluator可见；额外goal-set数量、生成规则及权限；RGB-only不能默许物理state生成目标 |
| `training.components[]` | component、checkpoint/revision、pretrained来源、frozen phases、额外优化类型、数据/split、teacher权限、训练/适配成本、发布资产/许可核验状态 |
| `training.claim_scope` | 相对哪个输入checkpoint“不额外训练”；基础冻结、LoRA、head、macroencoder分别记。训练时future/state teacher不等于部署oracle，但也不能隐藏额外训练 |

权限差异须输出具体维度，不能仅凭都写visual或frozen判同合同。HWM的proprio和learned高层、PlanWAM的ego特征及LoRA均应显式登记。AP observation retrieval与stored-action retrieval应有不同槽，但当前未核验其论文设置，不预填allowed。D2P的goal-set额外输入单列；不因evaluator读pose就误记planner获物理状态。

## 最小迁移建议（未实施）

1. 保留现有record和classify行为、旧报告与ID不变；先新增独立sidecar合同，由显式record_id关联，含 `schema_version`、`protocol_details` 和逐字段证据。旧验证器只接收原record，不把sidecar喂给精确字段验证器，也不绕过验证。
2. 未来实施时新增一个读入adapter，分别验证legacy record和sidecar；旧调用无需sidecar仍可读取。缺sidecar的记录保留原分类，但新扩展比较返回 `insufficient_metadata`，不沿用旧direct标签作完整合同通过。扩展比较须有显式reference，逐维输出match/mismatch/unknown及证据。
3. 若必须内联，才显式扩展允许的可选字段集合、保留旧required集合，并修订递归validator以支持带status的typed leaf；不能把bool/null直接交当前 `_json_value`。不修改旧horizon正整数语义；未证实为预测horizon的旧H保留原文本/线索，horizon标unknown，不猜测回填。
4. 迁移逐条核实并追加来源，不覆盖旧记录。只有已核验值进入比较；project_reported、attachment_claim不能伪装成当前只支持的paper_reported。新增evidence kind需另立明确合同；未实现前保留文档sidecar、不注册为paper值。
5. 验收应覆盖旧record可读、无扩展不放行、goal_offset与horizon不同、SD/SEM/pooled不同、权限/训练差异阻断、未知值不默认匹配、nested重复不伪独立、引用不可升级为本地结果。验收测试只是接口证据，不替代实际强基线或P1。

扩展协议匹配仍不授予pooling或paired inference；保留现有操作禁令。没有运行收据不能证明等预算，没有独立outcome不能证明机制效益。

## 本轮验证

只读当前注册器、sampling封存接线、合同审查及已有原文核验文档；未联网、未导入模型、未跑测试/GPU。写入后仅检查两份授权文档的空白与必要字段覆盖。采用api-and-interface-design的合同优先、显式类型及兼容扩展原则，故推荐sidecar先行而不是直接破坏精确字段验证器。
