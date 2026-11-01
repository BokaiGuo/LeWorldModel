# P3 MPC20 saved consumer v2（工程加固，不是准入）

只增加独占v2 source/tests/本说明；不修改已冻结v1、原prepare或任何结果。无prepare/run/neural/physics入口，不导入torch/torchvision/gym/pymunk/stable_worldmodel；NumPy/PIL仅处理saved值。已有v1完整模型数组、same-feedback历史、动作/原Q、各臂分母和数值守卫校验仍调用冻结`base.consume`，不降低任何门槛。

## 两项具体实现

1. physical JSONL逐行拒绝未知kind/多余字段/坏链。明确13种正常事件及`invalid_render`失败证据；后者不能出现在clean结果。只允许当前有绑定源码的encoder/projector原生handler事件，未知内层kind也拒绝，partial不例外。固定四臂顺序、唯一run bookend、精确episode close、连续operation ID、LIFO attempt/return、pending、所有12资源attempted/completed/cap逐项核对；clean再按每臂/region独立复算constructor/reset/setter/goal_render/goalbind/public/physics/render/close/encoder/projector。不是只相信report同名计数。逐项核对128 observation+128 guard、40 warmup_return、80 dispatch+80 control_return、20 encode、16 seal、80 handler、4 close及所有operation事件（上述数字仅全20实际分母，不用于填补native-stop）。已绑定native constructor无隐式reset；每臂两次reset分别带2 setter、1 goal_render、2 physics子步，额外调用不借工程cap冒充合法调用。
2. 原PNG/别名/request/encoding digest/原生latent namespace/response receipt绑定，读取唯一`<episode>_b<boundary>_encode/preprocessed_pixels.npy`。从source SHA闭包中的原`image_transform` AST确定ToDtype(float32,scale=True)→原ImageNet Normalize→Resize224；AST读取原统计，不import外来模块、不猜默认值。原图必须原生224×224 RGB PNG，因此Resize是identity。NumPy独立逐次float32 cast、乘float32(1/255)、减mean、除std；逐值/逐bit比较全部[1,1,3,224,224]，无容差。拒绝ULP偏移、float64、坏shape/NaN/额外NPY字节/通道互换/错PNG/坏receipt/symlink。

v1未在encoding receipt中保存`preprocessed_pixels.npy`文件SHA，本版本如实输出`original_preprocessed_file_SHA_in_v1_receipt=false`；v2将当前文件hash和精确值重新绑定到原PNG/encoding receipt，不追授生产时文件attestation，不声称仅凭saved文件证明GPU执行。所有P1/P3/freshness/scientific_admission仍false。完整native-stop保持实际步数；partial只返回未认证完整census，不padding、不改动作/计划/评分。

## 实际验证与范围

RAM回归覆盖完整20与5/10/15/20 native-stop、未知与重复/删除事件、完整资源与合法region、原始ABI/source drift、全部uint8值的float32操作、NPY/PNG变异、冷import隔离、真实RAM native-array journal到v2消费闭包（4节点）。首轮61pass/4fail为测试fixture错误：重链缺JSONL换行及fake env fault名`success`未触发；已改为带换行及真实fixture的`terminal`，未删除负向断言。修正后65 passed in28.44s；另增正向重链回归，防止变异测试仅因坏传输格式而假通过。

MAIN明确要求的实际saved stdout自测已执行一次，exit0，未指定`--receipt`且未写原目录：report `results/agent_p3_same_feedback_physical20_20261028_v1_MAIN01/report.json`，SHA `cdf46362e7e446105edf15f8f13f3113b9a7481b93acbd81338e43f2c5640157`；prepare SHA `77b0872ef13a6f3447226c60024109f785d6508fba882aa68ee42107265bfc8a`。report实际59963734 bytes，v2和v1使用原final_reserve=64MiB（67108864bytes）显式cap，不是2MiB；实际stdout status `saved_v2_hardened_NOT_admitted`，20张预处理数组每张150528值逐bit相等，全部census通过。工具显示stdout被截断仅是展示限额，不是进程失败/receipt；未重复执行模型、神经或物理。测试及saved算术成功不是科学通过。之后不重复全量consume，由MAIN持久化exclusive新receipt。

最终联合回归实际120 passed in79.53s：本版本66项（含最后新增的正向重链7行测试）及冻结v1的54项。source365行/tests257行/doc29行，三件套现在冻结；原v1三SHA逐项复核未变。

## MAIN精确命令

在`/home/gbk/code/LeWorldModel/le-wm-main`，无需新prepare、审批manifest或GPU：

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/agent_p3_same_feedback_physical20_20261028_v2_tests.py tests/agent_p3_same_feedback_physical20_20261028_v1_tests.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p3_same_feedback_physical20_20261028_v2 consume --output results/agent_p3_same_feedback_physical20_20261028_v1_MAIN01 --report-sha cdf46362e7e446105edf15f8f13f3113b9a7481b93acbd81338e43f2c5640157 --prepared results/agent_p3_same_feedback_physical20_20261028_v1_prepare_MAIN01.json --prepare-sha 77b0872ef13a6f3447226c60024109f785d6508fba882aa68ee42107265bfc8a
```

`verify`与`consume`执行相同只读完整核查；可选`--receipt results/agent_p3_same_feedback_physical20_20261028_v2_consume_MAIN01.json`仅允许新的独占results平铺路径、`xb`拒绝覆盖，worker未落此receipt。封存后不再修改本版本；新改动必须新版本。仍依赖原prepare/source/weights/known2 saved闭包可读且SHA匹配；不打开reserved4、不扩大根集、不以此实现原full-C3树或科学收益。
