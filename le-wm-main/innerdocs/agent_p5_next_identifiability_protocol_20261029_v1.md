# P5 下一可识别对照：固定最新槽、真正改变早期历史

状态：NEW prospective protocol proposal，等待MAIN批准实现/生产。本文件写于已读本次结果之后，不是本次比较的预注册，也不将下一轮内部干预称物理反事实、科学修复或控制收益。仅新增此文档；不改已有source/tests/results，不运行NN/encoder/physics/fullstrict。P1/P5 admission继续false。

## 1. 实际依据及当前不能推出的结论

项目绝对根目录ROOT=`/home/gbk/code/LeWorldModel/le-wm-main`。以下相对定位均相对ROOT，不搜索替代run。

- 比较：`results/agent_p5_short_start_comparison_20261029_v1_MAIN01.json`，实际SHA `bbafa5ec58dde0f0a12d01aded1f2d49a6a869bfe1c73e0d638eb12a6c57ff24`，status=`all164_and80_common_H2_label_saved_rescored`。MAIN已完成wholeverify；本worker只读报告并重核文件SHA，不声称再次完成wholeverify。
- short：`results/agent_p5_short_start_full_roll_20261029_v1_cuda_MAIN01/report.json`，SHA `1b014fc5c2e9868ee5e08e0dc05b602b5e9708a342c9c842698183f3430297c4`；strict=`results/agent_p5_short_start_full_roll_20261029_v1_consume_MAIN01.json`，SHA `3a20eb780028ddd40a88d11c59b8ada9dcbc0c7147bf1df9aa9acde332aa26fd`。
- old H2：`results/agent_p5_raw164_jepa_h2_cuda_20261029_v2_cuda_MAIN01/report.json`，SHA `e5a561215bfe7b6f1c1c224e7661211e3f237fb16458a6a409f5a1da0fc08579`；strict=`results/agent_p5_raw164_jepa_h2_cuda_20261029_v2_consume_MAIN01.json`，SHA `783b65f67899791b5697bfe0819ccfb3231d57212b9e3e85aefc02419459c0ea`。原H1所有pins继承比较报告`/evidence/old_fixed/H1`与其绑定文件，不凭stem猜身份。

比较报告全164共同H2-label MSE：H1@20=0.5530512516676429，H2@20=0.5464820364799445，short@20=0.56454177042087。全80对的negative/positive/null分别为H1 25/31/24、H2 22/34/24、short 19/37/24；零cosine数量均0。不能因short负向数较少而称其预测更准：全量MSE反而较高；也不把MSE差视作独立因果效应。

H2@15与short@5/10/15的全80对observed/predicted response均exact zero：这些时点在此固定动作面板尚无pair差异，不能以null宣称方向正确。终点仍保留全部80及24个observed-zero，不只分析56个非零。

必须分开的混杂：

1. H1为actual5/10/15→20；H2为actual0/5/10→pred15→20；short为actual0→pred5→pred10→pred15→20。起滚时点、真实反馈、早期输入、T1/T2/T3调用序列及最新pred15生成方式同时不同。short−H2不是单独的“早期历史效应”或“horizon效应”。
2. H1 frame10取goal bind后plan0，H2取bind前warmup10。当前报告164/164的frame10 pixel SHA相同，但source records全不相同；不能据命名认定像素变化，也不能据像素相同抹去来源/编码batch的差别。
3. H1 label batch1与共同H2 label batch2有浮点接缝；比较已统一后者，不能把接缝当物理变化。actual5/10的bank为batch3，actual15/20为batch2，跨时点MSE曲线仍不是同batch纯累计误差实验。
4. old wholehistory/latest实际前两槽输入逐位相同，已由封存cellwise检查证明，不能再用它们作早期history对照。新short终于有真实causal pred5/10，但“有可用差异”还不证明它对输出有效。
5. 只有两known roots，off/on不是新环境样本；proprio特权、冻结JEPA模型、渲染/量化及非独立扰动面板保持。不能跨架构比较MSE大小，不能解释为物理正确反事实或控制成功。

## 2. 下一轮唯一主问题及固定四cell

问题：**在同一short pred15、同一原动作与同一final20 label下，替换早期5/10视觉和proprio历史，是否改变final20预测与原80对方向？** 这是确定的模型输入干预，不是恢复自然轨迹；最新槽不随替换重算。

每branch i=0..163，时间槽固定[5,10,15]。A=原H2 actual5/10；R=short causal pred5/10。每个因素成块替换两个早期槽，不假称单帧干预。固定所有cell，不挑root/off/on/h/坐标：

| cell | visual槽5/10 | proprio槽5/10 | visual及proprio槽15 | 预测20来源 |
| --- | --- | --- | --- | --- |
| AA | A | A | 固定short pred15 | 新1call/branch |
| AR | A | R | 固定short pred15 | 新1call/branch |
| RA | R | A | 固定short pred15 | 新1call/branch |
| RR | R | R | 固定short pred15 | 复用short最后一call |

这里AA不是原H1，RR才是short已执行的完整末次输入；不能把旧H2 final20冒充本AA，因为旧H2 latest pred15来自不同history。AA/AR/RA同样属于人为拼接的内部history，不认证可达性。

## 3. 可直接实现的ABI及精确来源

复用的源码只作正常import/封装，不monkeypatch旧guard/globals：

- `experiments/spectral_reachability/agent_p5_short_start_full_roll_20261029_v1.py`的`next_inputs`、原Trace/CudaBackend和saved codec；source SHA `6b62eead489a3b2c89f6d478cee77c9b3ec9d25253c27b9e293d01beffb56ece`。新runner必须自有3call/branch计量guard，不谎称满足旧4-stage guard。
- `experiments/spectral_reachability/agent_p5_raw164_jepa_h2_cuda_20261029_v2.py`的`rollover`、`CudaBackend.predict`及consume链接：第2次predictor真实输入已经提供actual5/10，只取前两槽，绝不拿其H2-pred15作本次最新槽。
- mask证据为`external/jepa-wms/src/models/utils/modules.py::build_action_block_causal_attention_mask`与`external/jepa-wms/app/plan_common/models/AdaLN_vit.py`实际forward路径；已封存cellwise source evidence表明T3允许矩阵为`[[1,0,0],[1,1,0],[1,1,1]]`，最后槽允许早期历史。允许边不证明有效影响。新prepare重新绑定实际loader/source/model配置，不能只抄此矩阵当当前加载证书。

统一文件证据结构建议：`{directory, report_sha256, json_pointer, array_descriptor, slice, raw_slice_sha256}`；descriptor继承原file/SHA/shape/dtype/nbytes/file_bytes，验证完整单NPY/no trailer后作slice。禁止把文件名规律当权威pointer；先核report call index/module/branch/phase/row交叉引用。

精确JSON pointer与切片规则（`$`为原JSON键，不作shell展开）：

1. short branch i的RR末次native args：`/calls/{4*i+3}/inputs/$tuple/0/$tuple/{k}/$array`，k=0 visual、1 action、2 proprio。完整native outputs：`/calls/{4*i+3}/outputs/$tuple/0/$array`及`/outputs/$tuple/2/$array`。与`/rows/{i}/stages/3/prediction`逐字节交叉核验。
2. R早期visual/proprio直接取上述末次args前两槽，另要求它们分别精确等于short stages0/1的**最后**输出槽：`/calls/{4*i+j}/outputs/$tuple/{0或2}/$array`，j=0,1。最新槽精确等于stage2的最后输出，禁止取full3早输出冒充pred5/10。
3. A早期来源：old H2 `/calls/{4*i+3}/inputs/$tuple/0/$tuple/0/$array`前两槽visual，以及同prefix下`/2/$array`前两槽proprio。必须用H2 consume原链接证明它们是actual5/10，且实际position/velocity及normalizer来源相同；不从raw state自行重编码，不用label bank当predictor proprio。
4. 固定动作取short RR完整action buffer，shape=(1,3,10)，原packet indices[1,2,3]，对应raw6..10/raw11..15/raw16..20。核old H2第2call相同buffer，历史raw1..20及normalizer不变；不做zero/padding/重排/重归一化。
5. 新`assemble(i, early_visual_source, early_proprio_source)`返回三个float32 contiguous数组，shapes分别(1,3,1,16,16,384)、(1,3,10)、(1,3,256,16)。只覆盖前两槽；所有cells最新槽与action全byte等于RR。模型输出full3 visual=(1,3,256,384)、None、proprio=(1,3,256,16)全部保存，final20只取最后槽。
6. label只用H2 `/rows/{i}/label_patch`的batch2 slot1，且与short `/rows/{i}/labels/1`全byte相同；需完整seal后才允许数值读取。actual5/10输入来自已公开context，不是未来20 label泄漏。

## 4. 事前计量、最少预算与不重跑边界

主方案固定**492新predictor API=492 batch samples=1476 frame samples**（164×AA/AR/RA，各T3）；RR复用旧164call，不重新付费。新encoder/proprio_encoder/AD/physics/training全部0。全量四cell比较存在656条branch-cell证据，但不得报告为656新预测。旧short656及H2原编码/预测成本按来源单列，不重复加进新预算，不称等FLOPs。

每新call full native IO为2,457,720 tensor bytes，492共1,209,198,240B；本数不含NPY header、journal、可选输入bank、label副本及load GPU memory。建议新实现事前固定output cap=3GiB、tensor cap=2GiB、wall=900s、floor=200MiB、final reserve=64MiB；不要复制整个旧目录或全部四cell输入bank：prepare仅保存source pointers/slices及输入摘要，run逐branch合成并由native Trace写入真实IO；共同label最多复制一个batch2/branch(128,974,848B)，合计1,338,173,088B tensor，小于2GiB。RSS/GPU峰值是实际另列，非上述IO上界。若实现额外复制需先重新证明上界，不静默降精度/缩分母。

若只问“joint早期替换有无影响”，逻辑最小额外预算为AA的164predict/492frames/0encoder，比较AA与旧RR；**不能据此区分visual/proprio/交互**。本建议主方案仍固定492，不依据先跑AA结果决定是否补AR/RA。164方案是另一个须事前批准的较弱问题，不是本轮自适应停机选项。

复用RR有跨job数值环境限制：相同模型/来源/请求MATH/TF32off不等于实际kernel逐位相同证书。若MAIN要求当次baseline重放以消除此层疑虑，须在prepare前批准**另一完整656call/1968frames方案**，四cells全部新跑、RR所有native IO/full3与旧short逐位核验；新tensor IO=1,612,264,320B，加同一label副本=1,741,239,168B，仍可在上述caps内。不得执行492后因结果不利才补164；预算模式必须先封，partial不得重试。

## 5. prepare → run → consume 的前置、拒绝与可识别读数

这是待实现接口合同，不冒充已有CLI。最小新runner ABI：

`prepare(comparison_path, comparison_sha, short_report/strict路径SHA, h2_report/strict路径SHA, budget_mode='reuse_RR_492', output)` → 无label数值解码的pointer/slice身份表、164×4分母、输入difference census、协议/source/model/旧strict bindings。

`verify(prepared, sha)` → 重建上述来源与规则，不调用NN/fullstrict；`run(prepared, sha, HOSTCUDA_authority, output_dir)` → only CUDA冻结加载，同buffer模型loaded-bytes/strict_load/source/config/model-before/after证据；`consume(...)` → 重算所有输入/输出/动作/shape/counter/journal/seal/artifact/模型绑定与saved scalar，不重跑NN。

拒绝规则：缺任一完整来源/strict/文件、time/branch/task/condition/selection/action/normalizer/model不一致，fail closed；不得新采样或换root。prepare必须给全164早期A/R bitwise/maxabs/L2差异清单；某branch输入相同保留为零干预，不删。若整个因素全164相同，则该因素不可识别并阻止将其作为新contrast生产；不能再给同输入起新名字。不同history内部不可达不作为工程失败，但显式标注。CUDA不可用无CPU fallback；阻止encoder/额外predict/retry；保留所有attempt/return/失败分母。观察到fused attention按原guard拒绝，requestedMATH仍只记请求不冒充执行证明。

全部492新full3返回且durable prediction seal完成后才解共同20 label。RR旧输出必须先绑定，任何新label数值不得进predictor。保存full native module IO及每slot来源摘要，不只保final。新strict必须独立完整核验，不把旧strict status当新成功；任何失败生成partial，不release科学门。

固定读数：每cell每164的float64 C-order全256×384 MSE；每80pair的response L2/dot/cosine/error与/2h central difference，exact zero→null，不加epsilon。所有统计同时给全体及原四root×condition；56非零/24零由共同label定义并与全80并列，不作只报非零的筛选。

预先固定MSE效应：visual=`((RA-AA)+(RR-AR))/2`；proprio=`((AR-AA)+(RR-RA))/2`；interaction=`RR-RA-AR+AA`；joint=`RR-AA`，同样逐branch保存。对预测向量也给同形式的完整差异/范数，但不把非线性cosine作可加分解；cosine报告全cell符号转移矩阵与null。固定报告所有正负零，不做显著性、调权或最优cell选择。

可识别范围：改变早期视觉/proprio块在固定latest/action下对模型输出和误差的条件影响及交互。若变化为0，只能说本panel此干预无数值影响；不能证明早期槽总被屏蔽。若变化非0，也不能定位究竟frame5还是10、视觉哪个patch、训练缺陷或物理原因；需要另一个事前固定的时槽实验才能分解，不在此追加。

仍未解决：short pred15质量、latest与early分布一致性、自然滚动的多步累计、H1/H2编码batch差、proprio权限、renderer量化及task success语义。不能以本内部分解宣布P1/P5通过或选择控制器。目标是取得一个真正不同且可解释的history输入对照，而不是给当前较少反转寻找胜出配置。

## 6. 本次交付与下一步

本次只有该新doc。实际检查：报告SHA/status/全量summaries与164个frame10 pixel身份标志的saved只读提取；阅读旧comparison/full-roll/H2/cellwise及mask源码。没有重跑分析、consume、编码、预测或物理，也没有新fixture源码需测试。

下一可执行工作：MAIN批准492复用RR或事前656全新模式后，实现独占runner及RAM测试。必测：A/R完全相同拒作新factor、混错branch/slot/packet/label拒绝、仅前两槽改变/最新槽与动作exact、全492seal先labels、真实Store→journal回归、partial/cap与guard拒绝。当前没有新生产CLI，不能据本doc自行启动；旧三件套与原失败/负结果继续冻结。
