# Full30 全14配对三层 saved 算术

仅新source/tests/doc。唯一输入：RGB完整receipt SHA `42164a225310c39ff83e398f55501e36e8db9876d5b52dbc697d031e31468166`及MAIN同时间scalar summary SHA `d9b97b5c351bcee90f1d2bde5598d2d2c6469dd7bcceb09b9a66117cad00fa16`。不调用producer/fullstrict，不读其上游文件、不重编码/预测/physics/训练，不改原Q/selector/成功门。

四branch固定root0/1×transfer/joint；control0/5/10/15/20/25/30共28位置，4anchor/24forecast分栏；rootwise joint−transfer固定14pair（2anchor pair/12forecast pair）。cp0 model_Q/prediction/error永远null，不补0，不把anchor重编码一致当预测成功。早停缺帧保留null及原位置，模型forecast存在时可单独保留；不以终点填缺cp。

逐行验证RGB与scalar summary原physical report/receipt SHA及完整资源向量相同；state/fullsnapshot/PNG/file/pixelSHA、task/goal、plan/time、native成功严格连接。原native全部轴由已认证scalar summary导入，并独立逐轴相减核其14pair。两份输入没有原native goal7，本文不声称再从state独立重算native；不把receipt的NumPy native量冒充MAIN的fsum路线。

独立向量算术：确认192维字面值finite且无损float32，转float64后逐坐标subtract/square，以math.fsum/192计算model-Q、actual-Q、model−actual有符号差/绝对差、pred-vs-actual MSE。不导入producer数学。原receipt采用NumPy mean，另以本地独立NumPy表达式**精确**核其保存值；输出fsum−NumPy差，不能要求两归约逐位相同或用epsilon合并sign。全部sign只用严格>/<和等于，绝不用乘积判异号，防下溢；same/reversal/both_tie/left_tie/right_tie/null各栏，全14/forecast12/anchor2各自保固定分母。

native7距离/平方、joint/agent/object XY、raw/native/wrapped angle及velocity分开；native success差另列。每对保actual/pred向量是否逐位相同；root1相同也不去重或删分母。资源导入包括RGB全模块轴、physical counts、模型完整资源/wall；FLOPs及未知量原样null，不称等预算。跨臂动作不同不称同动作因果；同动作只指每臂自己的forecast与其实际执行路径。后验诊断不签P1/P2/fresh/physical release。

## MAIN CLI

cwd `/home/gbk/code/LeWorldModel/le-wm-main`，所有文件路径canonical绝对：

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p2_full30_three_layer_summary_20261029_v1.py
MOD=experiments.spectral_reachability.agent_p2_full30_three_layer_summary_20261029_v1
.venv/bin/python -B -m "$MOD" analyze \
 --rgb-receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_two_program_full30_rgb_encode_20261029_v1_consume_MAIN01.json \
 --rgb-sha 42164a225310c39ff83e398f55501e36e8db9876d5b52dbc697d031e31468166 \
 --scalar-summary /home/gbk/code/LeWorldModel/le-wm-main/results/main_p2_full30_same_time_summary_20261029_v1_MAIN01.json \
 --scalar-sha d9b97b5c351bcee90f1d2bde5598d2d2c6469dd7bcceb09b9a66117cad00fa16 \
 --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_full30_three_layer_summary_20261029_v1_MAIN01.json
# S为实际新summary绝对路径，SH为其实际SHA。
.venv/bin/python -B -m "$MOD" verify --summary "$S" --summary-sha "$SH"
```

仅排他新文件、16MiBcap、文件及目录fsync；verify整份复算含绑定。worker只RAM/ABI检查，不运行真实analyze/verify或fullstrict。

首次最小回归35 passed/0.38s；补缺anchor、真实非零fsum−NumPy归约差、reader篡改及实际保存字段形状后，最终38 passed/0.41s，exit0。实际ABI测试仅读两份指定SHA的字段/维度/null，不调用summarize/analyze，不产实际结果。RAM test覆盖全14对root1向量相同/native全tie、模型anchor为null、forecast与anchor分母分开、早停保模型但actual/native null、极小异号值避免乘积下溢、真实持久化writer→reader→wholeverify、来源/资源/指标/PNG/state/goal/task篡改、冷import无producer/torch/gym。数值比较不调epsilon，原NumPy与独立fsum分栏；均不是科学收益验证。三件套至此冻结；剩余MAIN全文审读、独立回归及实际saved analyze/verify，本worker没有生产结果。
