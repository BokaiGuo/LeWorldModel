# 已存actual cp5任务构造器

仅saved文件join，面向MAIN/Bohr未来own-schema prepare的planner输入端。不是R runner、
不是重新编码、不是执行端恢复证明；不改旧R任务/结果，不调用任何producer或fullstrict。
已见两根及其四重复分支不是独立新根，所有科学/物理放行标志均false。

## 接口

模块：`experiments.spectral_reachability.agent_p3_actual_cp5_task_builder_20261029_v1`。

`resolve_actual_cp5_tasks(authorities) -> dict(tasks=[task0,task1], provenance=[...], ...)`。
`authorities`严格包含`prepared/report/receipt`，每项为`{"path":绝对路径,"sha256":实际SHA}`。
三个输入必须属于`agent_p3_shared_panel_20261029_v1`真实schema；receipt必须complete，
且精确连接report SHA、prepare SHA和prepare路径。其完整strict由MAIN先前执行，本模块明确
`upstream_fullstrict_rebuilt_here=False`，不假称重新消费全部native/RGB或神经parity。
API不猜默认来源；未来runner显式选用返回tasks，保留本输出的SHA与provenance。

原task来自prepare.tasks[0:2]，原warmup raw10和原始观测标签来自prepare.layers绑定的
gzip task export，压缩/解压均有界并核实际hash。不从normalized tokens逆推raw动作。
每根必须有四个b5 sealed节点：fixedH1/endpoints、fixedH1/midpoints、adaptive/endpoints、
adaptive/midpoints；全部节点须有对应strict audited条目和完整branch row，不能取子集。
构造器逐一重建原request.local_task绑定公式并校验原goal、identity Q、model/root/blind/domain、
normalizer；核对固定prefix bits、原selection hash、raw10、once-normalized action tokens、
latent前两帧移位、实际observation chronology与原LF packet digest。
四分支完整context/goal/projection/normalizer必须字节一致后才构造本模块local task。
不按Q、成功或终态选择代表；固定顺序只用于已经验证一致的数据序列化。

## 时间语义与身份

| 数据 | 绝对warmup起点步 | control相对步 |
|---|---|---|
| 原root三帧 | 0,5,10 | -10,-5,0 |
| actual cp5三帧 | 5,10,15 | -5,0,5 |

cp5 latent_history = 原latent_history后两帧 + 已保存的actual cp5 RGB编码。
cp5 raw10 = 原warmup末5 + 已实际执行原prefix5；归一化为原float32减/除并reshape(2,10)。
不是imagined prefix；没有新增encoder调用，也不把原observed10误称control10。
原task完整保存在provenance.original_task；全部原分支local binding与observation逐项保留。
新task保留原root/model/goal/Q/normalizer/domain等身份，但有本模块独立的local task_id、
task_sha256、task_contract_sha256和actual context_sha256，不能冒充原control0任务。
local identity绑定三输入path/SHA和全部4个request digest，不受终态结果影响。

## 边界、成本和失败

0 encode/projector/predict/AD/physics；无torch、CPU神经或训练入口。
prepare<=16MiB、report<=256MiB（兼容原132MiB报告）、receipt<=32MiB，
每个gzip export压缩/解压<=8MiB，独占输出<=16MiB；不扩大原planner report reader合同。
验证前后重hash三authority、两export及本三件套，输出保存这些bindings。
上游全部源、模型权重、PNG、journal的严格验收范围继承MAIN receipt，
本模块不重新遍历/重签这些上游资产。future runner仍须封存自己的模型及完整source闭包。
不一致/缺项/重复/bool/nonfinite/非literal f32/SHA变动均拒绝；不输出部分task替补，
未声明零成本复用尚未认证的IO。执行端独立reset/warm10/prefix5及exact cp5 gate
由Pasteur physical_commit_preconditions/未来物理driver负责，本构造器不授予physical_release。

写入使用exclusive xb、文件内flush/fsync后fsync目录；不覆盖已有结果。
纯`join(...)`只作为RAM fixture数值/ABI seam；生产应调用验证文件身份的resolver。

## MAIN命令

从`/home/gbk/code/LeWorldModel/le-wm-main`执行；worker未执行以下生产prepare/verify。

```bash
.venv/bin/python -m pytest -q tests/test_agent_p3_actual_cp5_task_builder_20261029_v1.py

.venv/bin/python -m experiments.spectral_reachability.agent_p3_actual_cp5_task_builder_20261029_v1 prepare \
  --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_shared_panel_20261029_v1_prepare_MAIN01.json \
  --prepared-sha 7ca85050eeab4e47ed35cf84c284ed7e08fc64bce8064d7db9450074045b643a \
  --report /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_shared_panel_20261029_v1_MAIN01/report.json \
  --report-sha 668db35d79eb650419e4efa42dfc21a15cdf91725700b2f202e2a2ec975bcd6a \
  --receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_shared_panel_20261029_v1_consume_MAIN01.json \
  --receipt-sha d23ffa48bc954306bc27a15f924da8daafffc9b4bc76a91e59aa949ab1abbafd \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_actual_cp5_task_builder_20261029_v1_prepare_MAIN01.json
```

验证（将`ACTUAL_NEW_OUTPUT_SHA`替换为上一命令实际打印的SHA，非未来预填值）：

```bash
.venv/bin/python -m experiments.spectral_reachability.agent_p3_actual_cp5_task_builder_20261029_v1 verify \
  --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_shared_panel_20261029_v1_prepare_MAIN01.json \
  --prepared-sha 7ca85050eeab4e47ed35cf84c284ed7e08fc64bce8064d7db9450074045b643a \
  --report /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_shared_panel_20261029_v1_MAIN01/report.json \
  --report-sha 668db35d79eb650419e4efa42dfc21a15cdf91725700b2f202e2a2ec975bcd6a \
  --receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_shared_panel_20261029_v1_consume_MAIN01.json \
  --receipt-sha d23ffa48bc954306bc27a15f924da8daafffc9b4bc76a91e59aa949ab1abbafd \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_actual_cp5_task_builder_20261029_v1_prepare_MAIN01.json \
  --output-sha ACTUAL_NEW_OUTPUT_SHA
```

重建本模块完整输出，不重建上游fullstrict。
调用方可以直接使用Python API，不必先写中间manifest；不修改Bohr编译器。

## 验证范围

仅合成RAM/临时文件ABI测试：完整known2/8branches、精确时间、原绑定重建、
同分支重封错误cp5仍拒绝、missing/duplicate/bool/nonfinite/goal/raw/normalizer/
prefix/observation/receipt/SHA/schema拒绝、真实gzip/磁盘reader/CLI roundtrip及exclusive写。
无本模块生产输出、神经或物理结果；MAIN审读后承担实际saved调用。
