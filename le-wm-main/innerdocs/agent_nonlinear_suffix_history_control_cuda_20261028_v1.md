# Hybrid held-old2 suffix history control — fixed4 HOST CUDA

## 固定干预，不是部署或科学准入

消费已封存 `agent_nonlinear_suffix_midpoint_rgb_20261028_v1` 全四槽 root0/off、root0/on、root1/off、root1/on。midpoint实际 report SHA `1639745d33424a28e39a36924b4fe161a00aed2f5a6ae035352c898a9c8e88a0`、strict SHA `127c650eace2c41e43dc0361aa13009981acfc8e9509dfff7a1f924283f87521`；原prepare SHA `6dd4ef991ebf2947388ae7c139a23776592b200e11032fdc106be281ae9cc5e8` 从其report严格绑定。CLI仍由MAIN提供完整真实path/SHA，无未来报告SHA默认值。

同原raw10/后缀5、goal192、identityQ、原normalizer、原冻结模型与真实midpoint PNG来源，仅将预测输入改为显式 **hybrid latent3 = [original_root_context.latent_history[-2:], actual_full3[-1]]**。action_history仍是真实warmup6..10、plan1..5归一化tokens；附上原plan6..10 token进行一次预测。它不是完整实际RGB history、不是deploy/recompile，不把原root信号冒充child；不重编码、不训练、不拟合、不选择动作/根/模式。old2差异再小也不直接宣称无因果贡献，无相等阈值、无事后容差调节。

四槽必须都有完整真实midpoint/terminal编码和上次预测；缺一在prepare明确failclosed，不能筛行、填0、复制帧或补模型调用。run一旦失败保留四槽初始分母、真实已付attempt/return及partial，不重试；不把未运行项或工程失败当科学负结果。没有打开四reserved roots。

## 来源闭包与真实 native 接线

build真正调用冻结 midpoint.consume，完整重建比对持久 strict receipt；该链继承terminal/physical/suffix v2全168 consume/verify/build与原PNG、goal、task、rawbits、snapshot时间/历史身份、模型/Q来源。新prepare继承整个bindings，再绑定实际midpoint report/prepare/receipt和全部arrays/events、自身source/tests/doc。原suffix compiled slots task与original_root_context/model_task_binding逐值绑定，不重写旧schema或伪造authority。

取实际midpoint report.rows.actual_full3、prediction及feedback.inputs，用已冻结midpoint.load_saved_array读取/校验全精度原float32数组；真实incoming2/outgoing3/raw_packets/mean/std均按原动作重算逐字节相同。prepare保存原task、root_context、entire observed context/prediction3、actual terminal192、hybrid context、原后缀5与request SHA及source_array引用。源文件/实际已安装class源码闭包继承suffix v2所补齐的torch/transformers module sources，不以torch版本号替代源码SHA。

runner直接复用原suffix v2 `Backend.step` → NativeProgram/native ledger，**每行一次**原生后缀5步预测。task只含原模型可见context/goal/Q/normalizer；传入step只有hybrid latent3/action2及原raw5。原native参数与buffers前后frozen/tensorhash精确相同、绑定源前后不变，绝不修改旧module globals。NativeSignal.observe的临时wrapper只作用于本次新模型实例并恢复；禁止encoder/projector，资源cap为0；额外线程profile禁止AD，不替换torch globals。

全部原精度buffer即时写入fsync/hashchain journal：program_packet的latent `[1,3,192]`、history `[1,2,10]`、raw `[5,2]`、mean/std、token `[10]`、outgoing `[1,3,10]`；AE输入/输出、public predict两输入/完整输出、predictor两输入/输出、pred_proj `[3,192]`输入/输出、program_boundary及forward start/finish。native `[1,3,192]`输出不是shifted history，只有其末latent为未来5步终态；boundary按原算法shift。consumer精确复算各数组接线/shape/dtype/hash、scope/operation link、entry/IO/return嵌套及边界，全量拒unknown events/candidate/AD事件，不以过滤器抹去异常。

精度：CUDA only、MATH SDP、matmul/cudnn TF32 off、autocast off、inference/no_grad，before/after守卫持久且严格匹配原contract，每步检查。模型前后hash/source、prepare SHA、文件/目录全量census、绑定/初始四槽、native seal、逐行结果与journal SHA/tail皆须一致。

## 资源账与输出诊断

恰4 predict API/4 samples/12context tokens，4 AE（4samples/12tokens）、4predictor（4samples/12tokens）、4pred_proj（12rows）。0encoder/projector/AD/SVD/solve/physics/训练。逐行ledger实际attempt/return、所有core资源轴和nested inclusive wall保留，consumer独立总计等于4真实预测；不是按cap填实际数。继承midpoint编码/预测与更早物理/优化费用独立保留。本次0新物理不表示原数据从未执行物理。CUDA allocated3GiB/reserved4GiB、总输出64MiB（含final8MiB）、floor200MiB、wall600s；新排他路径、不覆盖不删旧结果；FLOPs/cuda_ns/engine backward unknown=null，不称等FLOPs。

每行同时输出hybrid和entire-observed终态192、actual终态、两组endpoint MSE（primary）及goalQ绝对误差（secondary），差值方向均hybrid minus observed；保留完整terminal_response向量/MSE/maxabs、old2每个语义角色warmup5/plan0漂移MSE/maxabs。保留endpoint误差变化的cross term与response squared，不删除抵消。没有因果泛化、baseline排序修复或物理收益结论；这仅检验本固定预测中old2重编码漂移是否改变终态输出。P1/freshness/scientific_admission/physical_release始终false，两个known roots不提供独立确认。

## MAIN精确CLI（inner根目录）

```bash
M=experiments.spectral_reachability.agent_nonlinear_suffix_history_control_cuda_20261028_v1
HISTORY_PREPARED="$PWD/results/agent_nonlinear_suffix_history_control_cuda_20261028_v1_prepare_MAIN01.json"
HISTORY_OUTPUT="$PWD/results/agent_nonlinear_suffix_history_control_cuda_20261028_v1_cuda_MAIN01"
.venv/bin/python -B -m "$M" prepare --output "$HISTORY_PREPARED" \
  --midpoint-output "$PWD/results/agent_nonlinear_suffix_midpoint_rgb_20261028_v1_cuda_MAIN01" \
  --midpoint-report-sha 1639745d33424a28e39a36924b4fe161a00aed2f5a6ae035352c898a9c8e88a0 \
  --midpoint-receipt "$PWD/results/agent_nonlinear_suffix_midpoint_rgb_20261028_v1_consume_MAIN01.json" \
  --midpoint-receipt-sha 127c650eace2c41e43dc0361aa13009981acfc8e9509dfff7a1f924283f87521
# HISTORY_PREPARE_SHA取上条真实stdout
.venv/bin/python -B -m "$M" verify --prepared "$HISTORY_PREPARED" --prepare-sha "$HISTORY_PREPARE_SHA"
.venv/bin/python -B -m "$M" run --prepared "$HISTORY_PREPARED" --prepare-sha "$HISTORY_PREPARE_SHA" \
  --output "$HISTORY_OUTPUT" --authorize-host-cuda
# HISTORY_REPORT_SHA取真实report.json，不预填
.venv/bin/python -B -m "$M" consume --output "$HISTORY_OUTPUT" --report-sha "$HISTORY_REPORT_SHA" \
  --receipt "$PWD/results/agent_nonlinear_suffix_history_control_cuda_20261028_v1_consume_MAIN01.json"
```

## RAM测试、未执行项与冻结

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider \
  tests/test_agent_nonlinear_suffix_history_control_cuda_20261028_v1.py \
  tests/test_agent_nonlinear_suffix_midpoint_rgb_20261028_v1.py
```

测试仅RAM native journal/array metadata、临时saved artifact与cold subprocess import guard，绝不调用模型/环境。首轮21pass/4fail/19error：上游RAM report fixture缺status，导致多个依赖fixture连锁失败；补齐真实ABI后44passed/2.56s，生产校验未降。扩展联合123pass/1fail：重封unknown-event的测试fixture使用已有尾换行的metadata raw再加换行，导致journal空行解析提前拒绝；已改用真实NativeJournal的transport.raw framing，保留精确拒绝unknown事件的断言。最终实际结果随SHA交付。worker没有生产prepare/verify/consume、GPU、neural CPU或physics；MAIN审读、真实prepare和主机CUDA/strict尚待执行，科学效果unknown。三件套交付冻结后任何修复只新版本。
