# 固定222 mixed-mode prefix：MAIN-only CUDA实现

只新增同名source/tests/本文。冻结预注册为外层`docs/D2P_P1_NEXT_CANDIDATE_POINCARE_20261028_v1.md`，SHA `9a48dc236024971c2aa65368dc504ea716b9aa5bd6672b8385b0c8a623c6c68f`。两known roots×off/on四项全保留，P1/P3=false，不是独立确认，不根据旧物理结果选方向/权重/root。

## 数学与复用

原allmode v2的42项精确保持；附加180项按i<j、si/sj各-1/+1，使用原外半径R的`R*(si*vi+sj*vj)/sqrt(2)`。复用原all_mode_bank所给完整右基，不删null modes，不因重复raw少计候选。混合项用已封存capture.project的逐行float64 L2球screen（批16复制只为满足其接口），保存投影前后logit/norm、i/j/sign/radius和basis SHA；原42项不再投影，保证精确保持。

所有prefix/suffix只生成一次float32 NumPy tanh权威raw，模型/whole/segmented replay/派发同buffer；不再CUDA tanh（zero点AD链例外）、不二次/100。每个prefix在自身完整预测context重算nominal/J与原`.002` bounded ridge，β1 score=`stage+own_reach`，稳定index argmin，不按非线性终点或真实标签重选。保留S/V/T、全部candidate、initial/greedy/zero/old42 winner/Pareto描述。

不复制完整旧运行框架：直接复用冻结transition、Backend.__init__/step/nominal/jacobian/replay、Modules生命周期/post/array/counts、期望forward schedule构造、kernel数学与history校验、Store事件压缩/put/close、math_resources。新显式candidate循环、上限适配forward/pre与MAIN orchestration为独立函数，不改旧globals、不通过exec/改函数global字典注入。own prepare在base.build完整来源闭包上绑定自己的三文件和冻结prereg；仅saved输入，不读取旧物理标签。

## 固定资源与存储

每项1338 predict/AE/pred_proj，223J（192×10），42816数学VJP方向，224成功helper内部SVD、222 suffix solves；四项5352 predict/AE/pred_proj、892J、171264方向、896成功helperSVD、888suffix solves。ADforward已计入predict。failed helper内部SVD及engine backward API不可见为unknown/null，不把VJP方向当engine调用数。共享历史16编码/source/父规划/v1失败成本不为0。不是等总预算/FLOPs。

prepare2MiB；总output2GiB、final reserve1GiB、diskfloor200MiB；启动须free≥2GiB+200MiB。**journal另有累计解压原文字节2GiB cap**，writer在压缩前计数，consume以相同上限inflate并精确核report中的journal_plain_bytes；不能因gzip很小越过解压cap。7200s仅操作前/返回检查，MAIN应另用watchdog，不保证抢占native/AD卡住。MATH、TF32false、autocastfalse、冻结模型before/after tensor hash、CUDA-only，无训练/新编码/physics。所有forward/AD/module在操作前登记attempt，数组逐次durable保存，返回才completed；helper/候选中间数组也即时journal。失败不补候选/根，不重试，四行census保持not_started/failed_partial/plan_sealed，任一失败不释放动作。final report失败另尝试有限failure.json；底层磁盘IO失败仍无法保证终态持久化，不隐瞒。

纯算术saved验证也不是免费：run每成功row额外调用一次完整validator（1root bank+222suffix重新解算=224个成功helper内部CPU SVD，四项896），有独立attempt/return；consume再付同等896个CPU SVD。重复人工调用validators按次另计。prepare/verify沿父闭包的saved验证工作未profile，标未测而非0。它们无新神经调用，不计入模型AD/SVD列，也不混称5352是全成本。

I/O工程优化：写前只读内部used_bytes；exclusive O_EXCL/O_NOFOLLOW创建，追加只允许已登记文件并以fstat核inode/旧size；write/短写/异常后在finally用fstat实测增长更新size与used_bytes，不按请求长度假计成功，不自动补写。若无法取到实际size则accounting_failed，后续拒写；已观察到外部inode/size漂移则持久拒收。每次artifact fsync及目录fsync不省略。final写与终态做完整O(files) census，核文件名集合、inode、size、总bytes及cap；consume另核外部report SHA/完整manifest/hash/实际字节和解压journal。普通写不再扫描全目录，避免约4万文件O(n²)。独占目录不是OS隔离：未提前发现的外部写入在终态拒收，不将其变成可释放计划。小型tmp测试只写几十字节mock，不触碰研究资产。

完整native输入输出与J均保留float32原NPY；report和journal含密集浮点JSON，不靠删字段或精度压缩过cap。真实saved dense row的保守32字符/数字界、完整RAM四行消费及真实saved build尺寸回归在同名tests。密集实际row/全部事件六倍投影：final-report保守数值长度界884,023,112 bytes，解压journal保守界975,222,676 bytes；普通区实际密集压缩产物/NPY六倍投影450,598,490 bytes，普通+final投影1,334,621,602 bytes<2GiB。普通区投影不是任意新浮点内容的压缩率定理，只有writer硬cap/floor是强保证；实际CUDA体积尚未测，不承诺无条件完成。saved build canonical目前680,215 bytes（最终文档SHA变动长度固定，不复制旧大报告）。所有测试仅数组/接口mock、微小I/O或只读saved；测试通过不是模型/物理成功。

## MAIN命令

在`/home/gbk/code/LeWorldModel/le-wm-main`。本worker不prepare、不run CUDA/physics。

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_mixedmode_prefix_20261028_v1.py
MOD=experiments.spectral_reachability.agent_mixedmode_prefix_20261028_v1
.venv/bin/python -B -m "$MOD" prepare --output "$PWD/results/agent_mixedmode_prefix_20261028_v1_prepare_MAIN01.json"
.venv/bin/python -B -m "$MOD" verify --prepared "$PWD/results/agent_mixedmode_prefix_20261028_v1_prepare_MAIN01.json" --prepare-sha ACTUAL_PREPARE_SHA
.venv/bin/python -B -m "$MOD" run --prepared "$PWD/results/agent_mixedmode_prefix_20261028_v1_prepare_MAIN01.json" --prepare-sha ACTUAL_PREPARE_SHA --output "$PWD/results/agent_mixedmode_prefix_20261028_v1_cuda_MAIN01" --authorization MAIN_MIXEDMODE222_PREFIX_CUDA_ONLY
.venv/bin/python -B -m "$MOD" consume --output "$PWD/results/agent_mixedmode_prefix_20261028_v1_cuda_MAIN01" --report-sha ACTUAL_REPORT_SHA
```

ownconsume状态`all4_mixedmode_commands_verified_not_admitted`，errors=[]、plans恰4，按(root_slot,condition,arm,task_id,blind_sha256)稳定身份，每plan有commands/prefix_commands/suffix_commands与selection_digest、whole_run_report_sha256。必须新physical adapter绑定本schema；不继承旧端点/白名单，不把latent model_Q当实际Q。当前没有physical adapter、fresh authority/direct筛查或P1三重实际检验；无新增Pareto/选择不变同样是应保留的有限校准结果，不允许事后调β扩池。

## 最终交付验证

实际执行上述同名完整tests：**60 passed in 69.31s**。含四项全222候选/5352-forward形状RAM ledger/全部NPY/CLI consume、原42编译结果逐字段精确相等、破坏收据拒收、前中后失败与终态首错保存、独立解压cap、O(1)写入/追加/短写/写后异常/final census及真实saved build/dense容量回归。首次测试的screen参考归约顺序错误已修正为冻结rowwise screen的精确算术，未放宽生产门或误报科学失败。未执行prepare、CUDA、神经模型或physics；微小tmp仅工程mock。三文件冻结后不再编辑，MAIN独占真实运行。
