# P4 known-root saved-source prepare bridge

仅本 NEW stem；不改冻结 runtime v1、旧 producer、旧结果或共享进度文档。仅 saved-data/NumPy/PIL，不能执行 CUDA/physics。mock 仅测试接口，不是来源真实性证据。

## 可重建与不能重建

从 P3 四根 sealed scheduler records、完整物理计划/逐步 journal、NEW17 prefix PNG/native192 NPY、原 goal/Q/model bindings 可重建参数化 root+source_arm 的真实 step5 context。imagined 使用原分段预测，actual 使用同根同动作真实 prefix；保留两条原 action token 和两个旧 latent。与实际 teacherforcing 的输入 context 逐字核对，不能用 teacher 输出充当新选优结果。

四根都是 known calibration roots，16分支/17帧不是17独立样本；重复来源不去重为新批。root 必须不在显式 prior-root 表，且表须与固定SHA的真实首批完整两臂20步报告一致，不能用任意无关排除表绕过首批root。额外前瞻排除/实验roster仍须MAIN协议，既有排除不是独立P1或heldout机制。本工具不复制大 protocol/checkpoint，原编译数组由旧 SHA/source closure 引用并核验H2/H3 Jacobian形状及内容SHA，禁止复用初始 Jacobian 伪称 prefix-specific response。

不能由这些 saved bytes 生出旧 capability 要求的123项真实扰动物理见证。因此本 NEW schema明确叫 P3_saved_compilation_and_observed_prefix，不伪造旧 `diagnostic_recorded`/qualification，不可传给冻结 runtime v1。`runtime_admissible=false,response_completed=false` 永久保留在本准备产物中；将来实际完成也须产生新 response/driver 产物，不回写 prepare。

NEW17旧编码已完成，teacherforcing真实17predict仅固定动作诊断：MSE改善不等于动作排序修复；root1旧Q排序仍反。不能只替换 latent 就声称闭环收益。未来 driver 必须消费真实 prefix RGB/raw-history，重新CUDA AD编译、十候选选优、独立重放、先封存选择再真实执行；step10/15使用自身轨迹的实际新反馈，禁止借用旧P3未来图。step20只能评价。

## 明确预算与缺口

本轮 actual_new_calls 全0。拟议新编译是H1、native3上下文、192×10 Jacobian；每次1 nominal+1 AD forward+10 candidates+1独立 replay，共13 predict/13action_encoder，192数学VJP方向。两臂离线step5响应上限26predict/26action_encoder、2 Jacobian API、384VJP；encoder/projector/physics=0，因prefix已真实编码。

未来两臂×三轮上限78predict/78action_encoder、6 Jacobian API、1152数学VJP；处理context tokens234。8encoder+8existingprojector调用，20图（6次三帧真实反馈+2终点评价），两臂各40步总额中的20步：全局physical cap40含两次5步prefix。无FD/训练/重试/CPU神经回退。输出cap4MiB、磁盘保留200MiB。缓存来源成本不视为免费。

这些是新H1设计上限，不是原P3 H2/H3资源等价比较，更不是已实现kernel计量：backward-engine调用数与walltime/peakVRAM上限尚须MAIN preflight及NEW driver真实hooks固定；当前为null并列入launch_blockers，不能启动。候选生成/约束/AD/compiler与真实逐步guard的实现和source closure也缺失。本工具仅保存准备请求，不实现假编译器或自动run命令。

## 命令

在 `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p4_feedback_root_sourcepack_20261027_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.p4_feedback_root_sourcepack_20261027_v1 --help
```

`audit/prepare --root ROOT --source-arm old_Q__uniform --metric old_Q --prior-roots PRIOR` 读取固定真实来源；prepare另给全新`--output`。`verify --output NEW_DIR --prepare-sha EXTERNAL_SHA` 重建所有源并验证seal；无GPU或物理入口。Python API允许显式参数化来源位置与外部SHA，但真实NEW17/teacher读取会被原producer的封存校验再次约束；不能通过改路径解除原来源闭包。

源码封存包括本三文件与实际使用的只读helper；旧大协议仅引用。缺包先失败，不创建占位。保存协议不是实际response，也不是科学准入；P1/P4始终false。

## 本轮实审与复现

新增35项mock测试；联合冻结runtime54项及second-batch preconditions32项，共121项。真实saved-data读取重建四根/16分支/17帧来源，对root0/2/3的四种原metric×scheduler分支均能生成双臂step5请求（12个来源选择，不是12次新批实验）。root1由实际首批报告排除。只封存root0、old_Q__uniform这一项作为NEW prepare示例；按原root顺序选择，不依据teacher结果排序。

实际NEW prepare目录：`results/p4_feedback_root_sourcepack_20261027_v1_root0_prepare01`。状态始终`prepared_missing_actual_response`，无actual response/selection/physical结果。

```bash
.venv/bin/python -B -m experiments.spectral_reachability.p4_feedback_root_sourcepack_20261027_v1 prepare --root 9173218f71898716eac8fe0f3946b8e2cf20527ac71d84555c20c4e1b7fbf333 --source-arm old_Q__uniform --metric old_Q --prior-roots 23b8de7837c39fb4e0fcffe7678732a9d840c57de7a0c8376e22076770ef6050 --output results/p4_feedback_root_sourcepack_20261027_v1_root0_prepare01
```

该prepare命令只可运行一次；目录存在后用verify及MAIN留存的prepare SHA，不覆盖或重跑。CLI故意没有run入口。
