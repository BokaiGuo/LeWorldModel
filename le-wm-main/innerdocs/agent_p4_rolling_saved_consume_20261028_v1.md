# P4 saved rolling/visual seam consumer v1

独占新增 source/tests/innerdoc，stem `agent_p4_rolling_saved_consume_20261028_v1`。
旧 moving_multiboundary v1 三文件和 MAIN prepare/results 完全不改。
此 consumer 只补 MAIN 指定的 rolling/alias 缝隙，不重复已有 full strict / native audit，
不替代 MAIN 对247 PNG、151 native rewards、10040 events、完整编译数组和预算的审读。

## 精确合同

固定8 episode、24 compile slot、2独立任务。逐臂 join sealed blind NPZ 的真实
float32 warm10、warmup_return、control_return、episode executed_raw；每个已到达
boundary b=5/10/15 必须满足 `actualrawhistory10 == (warm10 + executed_raw[:b])[-10:]`。
使用原 mean/std 以 float32 重算 2×10 action tokens；逐值且逐bit比较，无容差，signed zero也区分。
原 prefix5、goal192、diagnostic suffix5及各边界实际执行的新sealed commands均精确绑定。

R 从原 task root_context 出发，每次 shift 加本臂该 boundary 的真实 saved encoding192，
独立重建全部3帧，不仅验证最后一帧。I boundary5使用原 imagined_prefix；后续全部三帧及
action tokens 必须精确等于本臂上一个 response.predicted_successor，绝不注入观察 latent。
response successor 还绑定原 response.row.evaluations[1].context。

对于30个实际 encoding（包括7个terminal20，只作评估，不作第4个反馈），检查实际观测事件、
alias mapping JSON、original/alias PNG原始字节完全相同、解码pixel hash、编码request身份及摘要、
encoding response/独立receipt/row/event、完整float32 native192.npy文件SHA与数组SHA。
把这些独立重建的 observation hashes/index/latent摘要再与 compile_request 精确相等比较。
只解码需要接线的 boundary图，不重新做全247张原生图像审计。
两种旧JSON摘要格式保持原样：source.digest为spaced sorted JSON不加换行；
packet_digest为compact sorted JSON加换行。没有伪造统一摘要格式。

每个输入文件只读、限长、拒绝symlink，JSON拒绝重复key/NaN/Infinity，NPY禁止pickle，
receipt列出所有实际使用文件的绝对路径SHA及consumer三文件SHA。输出仅stdout，无写文件接口。
CLI必须外部提供report/prepare byte SHA；producer source固定2776029c…，执行binding须匹配。
不重新build旧prepare、不调用旧执行器/validator，不导入torch或环境，不加载模型。

## 早停与失败

MAIN02的e01_R在step11 native早停，只有boundary5/10；boundary15保留not_reached/null seal。
其余7臂20步，合计23个sealed boundary，未达第24个不补跑、不删分母。
terminal clean report才可消费；pending/errors/非终态/缺任一达成boundary/缺文件或错SHA一律抛错、CLI非零退出。
支持control阶段原生早停；warmup早停不是当前可认证接线输入，拒绝而不是伪造缺失历史。
unknown计算成本不补零；不宣称等FLOPs、真实模型预测正确或任务胜出。

## 验证与执行命令

从 `/home/gbk/code/LeWorldModel/le-wm-main` 执行。

```bash
env OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/agent_p4_rolling_saved_consume_20261028_v1_tests.py

.venv/bin/python -B -m experiments.spectral_reachability.agent_p4_rolling_saved_consume_20261028_v1 \
  --output results/agent_p4_moving_multiboundary_20261028_v1_MAIN02 \
  --report-sha 60489b7db3d4eff70d1be46d47d92209e6adcf333ae1d0a0e389b259ec1f6084 \
  --prepared results/agent_p4_moving_multiboundary_20261028_v1_prepare_MAIN02.json \
  --prepare-sha 8528e8d7f9b688b1c93be154305ccc35b8ed6218de63e736b1ae567691c432a2
```

已实际RAM测试及saved-only MAIN02消费，23 boundary精确通过。测试覆盖I/R多种早停长度、
缺边界、stale/错臂history、旧response滚动、normalizer/action/goal/diagnostic变异，以及
原PNG/alias/NPY输出/receipt/mapping/row/event/因果顺序变异。测试不是科学证据。

## 未运行与界限

没有运行CUDA、CPU神经或物理，没有网络、训练、阈值修改或补根。
不重算encoder输出（需要模型执行）；此处验证已保存输出的身份及真实下游使用。
不验证全事件链/全预算/原native reward正确性，依赖MAIN已经通过的独立审读，
本receipt明确标记`full_native_or_budget_reaudit=false`，不能单独放行物理或P4。
不重新hash巨大model weights；仅绑定原prepare外部SHA、原modelbinding及saved模型before/after。
`P1_admission/P4_admission/scientific_pass/physical_release`始终false。
P1及fresh-root前置仍未获准；此次saved接线通过不解除这些科学限制。
