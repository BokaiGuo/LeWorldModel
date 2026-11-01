# P3 fixed5 physical adapter v2（仅修复跨schema合同hash）

## v2最小修复与saved-only回归

MAIN确认v1实际prepare失败为序列化命名空间接线错误，不是身份漂移或理论负结果。原导出codec.core即p1_conditional_task_admission，canonical为`json.dumps(sort_keys=True,allow_nan=False)`（默认空格）；runtime fixed.digest使用compact `separators=(',',':')`。v2仅把bridge中原导出`task_contract`哈希改为`codec.core.digest`，以及自身NAME/文件/schema版本改为v2。没有兼容fallback、不接受两种hash、不修改任何已绑定SHA；context/binding/request/commands/fullplan/journal哈希均原样保留。测试以冻结v1源码归一化版本名后逐字比较，锁定此最小差异。

真实saved-only回归先验证实际CUDA prepare SHA，再沿其sources绑定cold manifest与明确task_export文件SHA；调用原codec.task_interfaces检查已封存导出，不运行producer.consume/prepare/模型/物理。两根真实合同按导出owner canonical精确匹配：

- slot0：`3e7a1b9da62918ca73610f222f836b447e7ea1610295f587e5d3ed00e9ef549f`
- slot1：`c89d987170c5d05340cb4bd4702c78cc387f7df0ee6ff9ed6d06c03e99867252`

测试同时证明旧compact hash不等于权威。将真实合同及其原权威用于明确标注的RAM bridge上下文，证明v1拒绝/v2接受；metric、pixels、slot、normalization或额外字段变化且不修改权威时，v2仍拒绝。RAM上下文不是实际native任务认证，不声称已通过真实prepare。全部旧82项RAM回归也保留在v2测试；旧v1源码/测试/doc SHA逐项核对保持冻结。

实际本地测试：v2完整 **96 passed in 21.62s**，其中新增saved合同/最小差异回归14项（单独运行 **14 passed, 82 deselected**）。未运行真实prepare/verify/producer.consume/物理/神经，未写MAIN结果。工程测试不是物理或科学通过证据。

独占新增源码 `/home/gbk/code/LeWorldModel/le-wm-main/experiments/spectral_reachability/agent_p3_fixed_execution_physical_20261028_v2.py`、测试 `/home/gbk/code/LeWorldModel/le-wm-main/tests/agent_p3_fixed_execution_physical_20261028_v2_test.py` 和本文。旧 producer、物理 helper、报告和共享文档均不改。实际运行仅 MAIN；本实现者只运行 RAM 数组/状态/图像及小型临时元数据单测，未执行真实 prepare、producer.consume、神经、物理或网络。

## 实际输入权威

固定 producer 为 `agent_p3_fixed_execution_cuda_20261028_v1`，源码 SHA `a307615d0366a475f876c2cdf0676a03b202f7470a5a96f5d2b4988e9ff67a31`。固定合同源码 SHA `7cb1cb3a89895755d2bb92cc9dd62de617e4a48471aa388f6fff4eab48a3aae9`。完整复用冻结 allmode 公共 placement/goal/render/hooks/Store 协议，helper 源 SHA `c0842f63685718e3d1d562fbbe5e2e3fdf9103954804f4276a1186d8db91efe4`；不调用它的旧 schema consumer/execute、不投影伪造旧 schema、不 monkeypatch 旧 globals。

MAIN 实际 CUDA prepare 为 `/home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_fixed_execution_cuda_20261028_v1_prepare_MAIN01.json`，SHA `2e680c6ff62e3a93b9c8cb7eef13d96510fe6d0b699561aedd84b616490bbdf9`。实际 terminal report 为 `/home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_fixed_execution_cuda_20261028_v1_cuda_MAIN01/report.json`，SHA `8067a743e3893b120ac8d57582c8552cb5b1dec182f7ca4bd0c6f70ffacaeeb5`。这些是已存在产物的绑定，不是未来 SHA。仅对 terminal 顶层字段做过只读 schema smoke；这不代表 MAIN strict 验收。

prepare/verify/consume 都必须完整调用真实 producer.consume；其错误不吞、不降级为信任 report 字段。保留整个返回对象；不重复 MAIN `consume_p3_fixed_execution_MAIN_20261028_v1` 的独立 Q、stable tie、raw5 复算。输入 hash 全闭包包括 producer source/prepare/report/所有产物、原 panel sources、cold manifest、明确 task_export、唯一 SHA-bound blind、旧物理源及新三文件。最终再验输入漂移。源码改变需 MAIN 重新审读；不能直接换另一个 batch SHA。

## 分母、上下文与动作

固定顺序为 `(slot0,fixed),(slot0,adaptive),(slot1,fixed),(slot1,adaptive)`，四项失败/未开始仍保留，已知根仅2，不是4独立根。条件均为原 P3 **off** task，不混 mixedmode/observed4。原同根 reset_seed/root_seed/placement recipe、warm10 非零 float32（前5 `[.1,0]`，后5 `[0,.1]`）、history3、实际 native7 root/goal、goalRGB/goal_pose/renderer 必须精确绑定。通过 cold manifest 的明确 task 导出和 SHA 输入边连原 source；不按文件名推断。

H1/H3 是模型规划包数，计划 raw5/raw15，但物理都只执行 winner 的 **前5个 raw float32 动作一次**，不执行 H3 后10、不延长 H1、不 child replanning、不读 child signal。禁止 tanh/clip/补零/缩放/拼接；native 实际 latest_action 必须 dtype/shape/bytes 一致。物理库内部原动作尺度100与 relative 控制不变。

warmup native terminal/truncation 不补历史、不重置认证新根，任务结果 unknown；认证根已 native terminal 时允许0步合法早停。实际 plan 原生早停立即结束，不补5。attempt、callback return、durable returned_transition、后续已校验 response 分开：step已返回而 render/校验失败，仍记真实 return，不填0；无法证明实际命令/状态时 physical_receipt=None。raw substitution、dtype错误、构造失败保留日志及四分母，不造任务结果。

每帧记录 PNG 文件SHA/像素SHA、native7 state/snapshot、pre/post 不变量；warm11 + plan0..5 的 off/on/off 完整桥最多51帧/分支。选定 off checkpoint仅0/5且仅在实际完成该capture时记录；早停缺失原样保留。原 source 中间8个 on 身份未保存，明确 `source_unpaired_on_steps`，不补旧证明；有明确原 frame 的位置逐项检查。

## 指标和资源

原生成功是 joint XY（前4维联合欧氏误差）<20 **且**原 native angle `min(abs(delta_angle),2*pi-abs(delta_angle)) < pi/9`。保留原生多圈角实现语义；wrapped angle 仅额外诊断。reward 是负7维欧氏距离（包括velocity），不能由 reward 阈值替代 XY+angle 成功。saved consumer 独立复算 native reward/success、XY、native/wrapped angle、velocity、delta；校验失败的 endpoint 指标保持 unknown。

正常四项上限：constructor/close4、reset8、public_step60（warm40+own20）、physics_step616、set_state16、goal_render8、goal_setter12、goal_bind4、explicit_render204。嵌套 axes **不相加**。计账来自真实 dispatch intent/entry/return，无退款；constructor 未返回则内部 work unknown。output32MiB、最终report预留4MiB、floor200MiB、wall900s（**execute scope，不含此前完整verify，不是whole-job**）；prepare≤64MiB。独占新目录/文件，重入拒绝；先验证 MAIN token 才读取/建目录/导入 gym。saved consumer 不创建环境或模型。run CLI只打印compact summary（状态/路径/实际SHA/计数/错误/wall scope）；完整report仍仅落盘，prepare/verify不算实际物理结果。

所有 P1/P2/P3/scientific/freshness/physical_release/full_C3/compute优势等 FLAGS 保持 False。`artifact_authenticity_verified=True` 仅表示此封存产物通过本consumer；上游 `audit_physical_receipt` 原封保留其 `artifact_authenticity_verified=False`，不冒充它认证物理。equal_requested_execution_horizon=True不代表actual相等；actual长度从实际返回且工程完整的四分母派生，否则None。测试通过、prepare、终态或consume成功均不构成P1/P3通过。

## MAIN 精确命令

以下从 `/home/gbk/code/LeWorldModel/le-wm-main` 执行。仅 MAIN 审读验收后运行；本实现者未执行。目录须不存在。未来 physical prepare/report SHA 只能在对应真实文件存在后计算，不预填。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -B -m pytest -q -p no:cacheprovider /home/gbk/code/LeWorldModel/le-wm-main/tests/agent_p3_fixed_execution_physical_20261028_v2_test.py

/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -B -m experiments.spectral_reachability.agent_p3_fixed_execution_physical_20261028_v2 prepare \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_fixed_execution_physical_20261028_v2_prepare_MAIN01.json \
  --producer-output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_fixed_execution_cuda_20261028_v1_cuda_MAIN01 \
  --producer-source-sha a307615d0366a475f876c2cdf0676a03b202f7470a5a96f5d2b4988e9ff67a31 \
  --producer-report-sha 8067a743e3893b120ac8d57582c8552cb5b1dec182f7ca4bd0c6f70ffacaeeb5 \
  --producer-prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_fixed_execution_cuda_20261028_v1_prepare_MAIN01.json \
  --producer-prepare-sha 2e680c6ff62e3a93b9c8cb7eef13d96510fe6d0b699561aedd84b616490bbdf9

P3_PHYS_PREP_SHA="$(sha256sum /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_fixed_execution_physical_20261028_v2_prepare_MAIN01.json | cut -d ' ' -f1)"
/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -B -m experiments.spectral_reachability.agent_p3_fixed_execution_physical_20261028_v2 verify \
  --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_fixed_execution_physical_20261028_v2_prepare_MAIN01.json --prepare-sha "$P3_PHYS_PREP_SHA"

/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -B -m experiments.spectral_reachability.agent_p3_fixed_execution_physical_20261028_v2 run \
  --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_fixed_execution_physical_20261028_v2_prepare_MAIN01.json --prepare-sha "$P3_PHYS_PREP_SHA" \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_fixed_execution_physical_20261028_v2_MAIN01 \
  --authorization MAIN_P3_FIXED5_PHYSICS_ONLY

P3_PHYS_REPORT_SHA="$(sha256sum /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_fixed_execution_physical_20261028_v2_MAIN01/report.json | cut -d ' ' -f1)"
/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -B -m experiments.spectral_reachability.agent_p3_fixed_execution_physical_20261028_v2 consume \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_fixed_execution_physical_20261028_v2_MAIN01 --report-sha "$P3_PHYS_REPORT_SHA"
```

consume 输出完整 JSON 到 stdout；MAIN 如需持久独立 receipt，请先确保目标不存在，用 shell noclobber 或 MAIN 自己的排他写封装。不要覆盖旧产物。实际 native 环境依赖/placement复现、预算足够及 PNG 源一致性仍须 MAIN 真运行确认；RAM 单测不是物理、性能或科学证据。

