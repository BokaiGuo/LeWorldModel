# P2 saved program linker：实际保存动作接缝

独占新增本stem source/tests/本文，不修改任何封存planner、Pauli tracker、physical runner、
结果或科学gate。不加载torch/模型、编码器或物理；不调用producer.consume/verify/build或fullstrict。
worker只运行合成RAM、临时磁盘ABI测试；以下实际保存分析由MAIN执行。

## 已有实现与本工具的非重复边界

原`agent_p2_three_program_20261028_v1`已执行两已知根off、joint/three-program各42候选，
每根1068predict/85J，并有整段/分段winner重放和实际MAIN完整receipt。
`agent_p2_three_program_model_mechanism_20261029_v1`已做全库成本与排序，不重做。
`d2p_composition_witness_v2.compose`需replay callback；本工具无任何模型callback，
只把真实保存IO解析为可引用程序capsule，再链接具备完整因果转移见证的程序。
不是新优化器、不选择winner、不分析outcome、不产生任意latent目标、不实现tracker。

正式输入为真实producer prepare/report/完整receipt的三个显式path+SHA，以及report绑定的
`root0.json.gz/root1.json.gz`和所引用170个原J NPY。完整42候选在两个root文件中，
不能将receipt的4个selected plans当成完整候选库。receipt的4个plan只作原task/model/
goal/normalizer/程序数权威join，不按它们选择候选。

本工具继承MAIN完整receipt的源/权重/全事件/全候选计算验收，明确
`upstream_fullstrict_rebuilt_here=False`。独立核所引用forward原raw/blob/request、
action_encoder→predictor→pred_proj完整保存IO、float32 once-normalization和完整history roll；
核compiler request的context、原incumbent/raw/logit/coordinates、completed、J文件真实SHA/
shape/dtype/finite/EOF及AD forward路径。不重新计算AD、SVD、原候选评分或完整事件流。
不是独立神经数值parity或重新签发原fullstrict。

## 程序身份、时序与合同

固定known2×2臂×42=168条原候选链，336个程序capsule；全分母不删候选、不按结果挑根。

| 原臂 | 程序 | 规划时域 | 承诺动作区间 | 编译anchor |
|---|---|---|---|---|
| joint | p0 | 15步 | control1..15 | 原root_J |
| three-program | p0 | 原root H3的15步 | control1..5 | 同原root_J |
| three-program | p1 | cp5 H2的10步 | control6..10 | 原edge2/J完整cp5 context |
| three-program | p2 | cp10 H1的5步 | control11..15 | 原edge3/J完整cp10 context |

joint虽然有3个packet转移，仍只有1个program。three-program的p1只承诺其H2计划前5步，
最后5步随后被p2替代；不能将H2的未执行末段误拼入full15。所有时间相对原root/control0，
不是actual cp5新起点，也不是把imagined context升级为真实反馈。
root raw10从prepare.warm_raw10与receipt.root_warm_raw10逐位核对，核原root action tokens。
后续每个packet核完整3latent/2action-token context。接缝比较完整context，不只末latent。
compiler anchor属于incumbent线性化输入，不冒称在最终选中动作处再次求J。

原raw域[-1,1]、joint30 logit球cap=.75√15，原NumPy tanh(float64)→float32字面映射；
不clip、不逆atanh、不投影修复、不加epsilon。所有连接后的committed logits也核同一原球。
模型/Q/goal/task/blind/normalizer/root/source authority必须完全一致。固定原root identity，
相同终latent但不同前history或动作token不能连接；同动作但缺确切保存转移也不推断。

## Python与文件ABI

`analyze(authorities, requests=None)`是已保存文件认证入口，返回本`.catalog`。
`authorities`严格为prepared/report/receipt，每个`{"path":...,"sha256":...}`。
`extract_programs(...)`仅纯RAM seam，不是省略文件认证的生产入口。

输出包含：

- `catalog[336]`：固定程序id、task/model/goal/domain scope、完整input/context路径、原logits/
  raw commands/rawbits、原compiler anchor、实际forward索引/tag/hash、planned/committed时域。
- `candidates[168]`：每个原root/arm/index完整原链的link结果，全部ties/原候选保留，不读成本。
- 成功link：真实拼接rawbits、逐步1..15程序与保存forward来源、完整terminal context。
- blocked：`commands/raw_float32_bits/terminal_context`均null，保留refs和精确原因；不替补。

`link_programs(catalog, refs, expected_steps=15)`支持1..3个固定程序引用及5/10/15时域，
例如同候选p0+p1可输出有保存转移的10步前缀。id格式：
`r0/three_program_imagined_recompile/c00/p1`。输入catalog必须来自认证入口/其绑定输出，
API重核capsule digest不是外部签名，不能把任意自造capsule称为认证模型响应。
显式requests文件是<=256项的列表，各项严格为
`{"id":"fixed-name","refs":["..."],"expected_steps":15}`，另传该文件path+SHA。
只处理调用者事先给定refs，不搜索/排序/择优；未知ref或未见过的接缝blocked。
对不同候选同scope仅在完整context确实逐字相同且各程序都有该输入转移时可连接，
不能把某candidate的输出替换为另一个输入后伪称原转移仍有效。

缺少所需tag/编译转移时保留原候选并blocked；保存字段互相矛盾、SHA/类型/shape/
原task身份篡改则拒绝整个输入，不把损坏数据当科学负结果。输入缺整根/候选也拒绝，不能取子集。

## 有界读取、保存与成本

prepare/report<=128MiB、receipt<=32MiB；root gzip压缩与解压各<=256MiB；每个J<=1MiB。
requests<=1MiB；输出<=32MiB、磁盘floor200MiB；文件xb排他写，文件仍打开时flush/fsync，
随后fsync目录，不覆写历史失败。绑定输入三authority、两root文件、所用J、冻结producer source及
本三件套，分析结束重hash。未知上游源/权重验收继承原receipt，不假称本轮重建全闭包。
全工具0新predict/encode/projector/AD/physics；纯CPU保存数组核算，不是CPU神经推理。
不声明省算力、等FLOPs、非贪心能力、P1/P2成立、fresh或physical release。

## 固定实际authority与MAIN命令

以下真实输入已存在；不是未来SHA。producer source SHA固定
`3111312477ff3cd9a764dc7d063861c1becaaf58885dc0fb25c2f3ec73d15fd5`。

从`/home/gbk/code/LeWorldModel/le-wm-main`执行：

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q tests/test_agent_p2_saved_program_linker_20261029_v1.py

OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p2_saved_program_linker_20261029_v1 analyze \
  --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_three_program_20261028_v1_prepare_MAIN01.json \
  --prepared-sha 86af2d24f73e769fb3b91fa8a571f8ca265df95c93af780d3924ea644585d461 \
  --report /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_three_program_20261028_v1_cuda_MAIN01/report.json \
  --report-sha 56e8b4f2df522f7a0c4cb9bc37ad74bcbc5d4f89e2c5cd601d42cd0eb071d8ef \
  --receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_three_program_20261028_v1_consume_MAIN01.json \
  --receipt-sha c491555fda5bb543af69f70987541edb0abb99fa32c559f3aa6190901393017f \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_saved_program_linker_20261029_v1_MAIN01.json

OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p2_saved_program_linker_20261029_v1 verify \
  --analysis /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_saved_program_linker_20261029_v1_MAIN01.json \
  --analysis-sha ACTUAL_LINKER_OUTPUT_SHA
```

`ACTUAL_LINKER_OUTPUT_SHA`必须用analyze实际打印值；默认不添加跨候选请求。
MAIN如需固定请求，analyze追加`--requests /absolute/jobs.json --requests-sha ACTUAL_JOB_SHA`；
verify从输出arguments重建同一请求列表，不擅自添加新组合。

## Fixture验证与未完成项

最终worker回归：**33 passed in 11.41s**，命令即上方单文件pytest。
开发首轮fixture的`/fixture/prepare.json`与descriptor中`prepared.json`不一致，
authority join正确拒绝；仅修新fixture路径，未放宽生产校验，亦非历史P2失败原因。

测试使用完整known2/42/168/336 synthetic native schema、1068/85原分母、170个临时J及两gzip root，
覆盖真实保存reader→extract→link→exclusive writer→verify，而非只有局部数值函数。
另拒绝错context/action-token但terminal相同、错误J anchor、缺转移、跨根、跨候选错接、
时段重叠、模块IO/标量类型/bytes/SHA/schema篡改。fixture不是原神经值或实际程序证据。
窄实际ABI smoke已只读核对真实root0的`joint/0/0`原生保存IO和共享root_J的3包AD forward/
192×30 J文件身份，退出0；未执行生产analyze/verify或上游fullstrict，无新保存产物。
实际原保存报告的全库提取/输出大小/linked与blocked计数仍由MAIN运行上述命令确认。
任意新context下的程序可执行性、未保存的跨程序转移与原生物理收益均不在本工具认证范围。
