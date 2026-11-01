# 已有 identity/fullgoal 后的窄修复：checkpoint-bound action normalizer

使用 experiment-code 技能；按用户有界范围仅做非神经 tests、prepare freeze。
不新建重复的 goal-consistent compiler，不改任何旧 sealed 文件，不训练，不跑 GPU/physics。

## 为什么不再做 identity/Q 实验

已有真实 `pusht_d2p_nuisance_suffix_bank4x16_v1_20261026_cuda/report.json`：
3968 predict、128物理分支、4根×16前缀×identity/Q，全部 gates true。
`pusht_d2p_consistent_metric_suffix.solve` 在谱初始化、GN refinement、最终成本
使用同一个 projection。另有 full192_one_shot/full192_gn4 对照和原官方任务重标记。
这不是当前六context/full10实验：它是 constant2、observed context、GN4、damping=.01。
不把两者宣称完全等价；这里只按用户“已有实际 identity/fullgoal 则换切入点”的条件分支。
这也不是 identity/Q 的一般负结论。

已有 live-vs-extra/100 对照也完成（64predict、16固定窗口）；不重复。
已有归档 action-only statistics 不同于 public 常数，但未绑定官方训练run，不能替换。

## 具体缺失的契约与本次代码边界

官方本地 checkpoint `config.json` 是模型构造参数，没有 fitted mean/std。
当前 `utils.get_column_normalizer` 从 dataset 计算实际 mean/std，`train.py`
只将其放入 dataset transforms。`save_pretrained` 保存 model.state_dict 和 config，
其调用接口不接收这些 fitted transforms。此为当前源码可见的导出缺口，不断言
官方权重一定由这版 train.py 训练，也不因文件缺失就声称权重内绝无相关数据。

新增 `BoundActionNormalizer` 是可集成的无默认值接口：要求独立审查的 training export
SHA256、精确 weights/config SHA256、训练run/recipe/config/action字段digest、
fitted mean/std、raw divisor、std ddof、frameskip5和time-major xy顺序。
元数据不在 checkpoint model config 中猜，不从 weights 或 outcome 推断；不接受
archive_statistics 作为 training export。公开常数只用于复验原路径，不作为该接口fallback。
`normalize` 使用 float32、完整五命令packet，显式divisor，并支持批次；不产生新模型预测。
`normalize_tensor` 可替换 `pusht_full_packet_compiler_20261027.make_transition` 内的
raw-to-token调用，保留tensor梯度/dtype/device；tests仅用CPU张量加减乘除验证梯度，
不实例化神经模型。正式接入还需恢复元数据并另封完整历史重绑定，不能只换未来token。

重要：用户提供的 trusted hash 是外部信任锚，不是密码学上证明训练事实。
schema正确或 self-authored JSON 不足以认证来源；不提供制造“官方元数据”的命令。
此接口尚未接入旧compiler/consumer，不能据此声称它们被修复。

## prepare 与主线程命令

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_p1_feedback_action_normalization_contract_20261027_v2.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m experiments.spectral_reachability.p1_feedback_action_normalization_contract_20261027_v2 prepare --output results/p1_feedback_action_normalization_contract_20261027_v2_prepare_v1
```

prepare 保存固定六context的12个原raw incoming packets、全部60个既有候选suffix，
不按outcome或赢家删分支。冻结source/protocol/input/witness hashes。没有可信export时
`runtime_gate=blocked_missing_authoritative_export`，bound tokens=null；这是诚实的
待元数据门，不是假通过或CPU neural fallback。所有实际预算（predict/encode/physics/
Jacobian/SVD/train）均0。本模块没有GPU run入口。

取得发布者或原训练run导出的真实元数据并独立核验digest后，先执行：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m experiments.spectral_reachability.p1_feedback_action_normalization_contract_20261027_v2 validate-export --metadata /ABS/PUBLISHER_EXPORT.json --trusted-export-sha256 REVIEWED_64_HEX_DIGEST
```

再用新的output目录 `prepare --metadata ... --trusted-export-sha256 ...` 可生成显式
bound packet tokens。不能重用旧imagined latents并把它们称为新normalizer下生成；
后续任何神经对照需另封协议，重新处理完整历史/imagined-prefix过程，而非只改suffix token。
本轮不运行、也不擅自授权该后续实验。

## 验证记录

2026-10-27：上述非神经测试实际执行，38 passed（0.86s），包含CPU张量归一化
梯度检查；未加载checkpoint tensor或模型。随后进行唯一新目录prepare。

结果仅以本次终端tests和新 prepare_receipt 为准；文档在prepare前冻结。
P1_admission=false、efficacy_verified=false。此交付修复的是新接口的隐式fallback问题，
没有恢复官方训练统计、建立物理效益或排除相机/模型其他代码问题。
