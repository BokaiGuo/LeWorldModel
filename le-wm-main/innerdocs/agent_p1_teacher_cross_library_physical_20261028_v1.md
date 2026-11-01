# Teacher共同库40分支物理adapter

仅新增本stem source/tests/本文，不改teacher compiler、旧probe/zero source/results或MAIN文档。worker仅RAM合成测试/saved metadata内存检查；MAIN负责审读、prepare和实际物理。无模型/网络/后代agent。

## 输入与40固定分母

绑定teacher source `ca80a146b602406a1ef960fd2fa064afabc9a2bfd02d5396983f328d7f443b4a`，实际CUDA report `d5ab99d55bacd73c8d9cf0fb774819bb46af63100053198858e639d8fa746738`，MAIN严格消费回执 `d82109eb8de582e59a4f0485307c01c9b91f9f84551c218c61cca4da24335213`。prepare读取外部SHA固定回执，不重跑compiler.consume、96predict/8J或任何神经计算；按全40 release验证schema/plan_digest/同库动作，不读model_Q选分支。

顺序：四任务 `(0,off),(0,on),(1,off),(1,on)` →角色Z/S→库 `[zero,uZ,-uZ,uS,-uS]`，固定40项。每支fresh原seed+sample_state placement recipe→原warm10→release prefix5→release suffix5，完整raw10 literal float32同buffer，无二次tanh/clip/除100。负零sign bit保留；零或重复动作照做，无复用轨迹、退款、补根、替换或重试。Z是零干预、S是旧模型选择；旧选择只作parent ancestry，本模块无新选择器。

原zero20/selected40报告、journal及每角色固定zero-probe分支，仅用于核前15步及teacher15实际PNG身份。每anchor在prepare封存完整51 reference frames（warm0..10/root0/prefix1..5各off/on/off）、原15 raw/state/snapshot；所有原PNG文件SHA纳入closure。新执行逐帧核encoded PNG字节SHA/raw pixels SHA、state7、完整snapshot、goal/renderer等于原参考，到15步核teacher PNG以及由实际warm5:10+prefix5重建的原float32 token/history context。未通过不执行新suffix。旧probe20端点不作为任何新库效果，也不根据响应/符号/评分选参考。

复用公开 `place/bind_goal/native_hooks/validate_env`、off/on/off `helper.observe` 与原 `RgbObservabilityGuard`；自有新schema、40循环、raw dispatch、限制、durable账与strict consumer。不改旧module globals，不reset(savedpoststate)。每primitive fresh RGB且precommand复用上一fresh guard frame；on仅成像干预，所有控制with_target=false。

## 逐资源硬预算与存储

全40满额：constructor/close各40、reset80、public800（warm400+prefix200+suffix200）、internal physics8160（8000 public内部+160 reset内部）、set_state160、goal_setter120、goal_render80、goal_bind40、explicit_render2640（每支66PNG）。无新encoder/predict/神经、训练/Qfit；上游编译与历史source成本完整保留而非免费，不称等FLOPs。

墙钟3600秒（包括run初始验证）；输出1GiB、final reserve64MiB、prepare64MiB、journal按未压缩128MiB、report16MiB、disk floor200MiB；启动要求输出cap+floor。Store维护实际O(1)成功/失败部分写字节，append前inode校验，终态完整目录census；fsync、no-overwrite、未知额外文件拒收。不裁剪日志/数组/原精度来过cap。构造前的内部工作若构造器失败只能标unknown，不捏造完整physics账。仅close允许越过wall/fault做非物理cleanup。

## 状态、标签与严格消费

全部40行从unstarted建立，保native_stopped/failed及未开始项，不补造success。native早停不继续派发或pad；即便在最后一步触发也保停止状态。endpoint20对非horizon_completed行为null，但实际已到checkpoint/last_return/native flags仍保留，不能拿更早终态冒充20步。失败首错与secondary close/写出错误分别保留。partial不能作为完整40支持值/成功比较。

完整每步actual raw bits/state7/full snapshot、reward/terminated/truncated/official eval与lossless native info NPZ；每步off/on/off PNG、guard及链式journal。root10/prefix15/terminal20均有checkpoint，记录official native与独立wrapped/all-jointXY/agentXY/objectXY诊断。原native多圈角公式风险原样保留，独立wrapped诊断不覆盖env终止；原success包含agent+object位置且无velocity门槛，native reward的state7距离包含velocity。不能将object-only或有符号一列当完整任务能力/成功。

strict saved consume外部SHA绑定新report/prepare/source和已验收compiler回执，逐文件、canonical hashchain、operation attempt/return、raw动作顺序、前15步原identity、teacher token算术、guard、全PNG/NPZ及checkpoint join。仅saved CPU数学/metadata，不启动env/model；partial仍保40分母及null。实际latentQ/RGB评分保持null，待MAIN另新编码产物，不伪造。

## MAIN入口

冻结前首轮RAM回归实际为46passed/3failed，不能当物理或科学负结果：原substitute mock将实际动作换成零，对必保留的zero库项没有制造替换；guard mock先被既有capture的fresh RGB/canvas guard拒绝，后层RgbObservabilityGuard未接到该帧，不能冒称其abstained；RAM FakeIO缺少生产fsync所用fileno。修复仅本stem测试：对全部库项注入确实不同的float32动作、核前层拒绝前保存的实际224PNG及后层未虚构观察/停止后无派发、补齐FakeIO描述符。生产源码guard、fsync、数学和预算不变。MAIN已启动的旧prepare若完成应原样保留不用，新闭包使用MAIN02。

最终独占测试实际 **49 passed in 49.05s**（下述命令，exit0）。均为RAM fake environment/PNG/NPZ/文件描述符测试，无实际模拟器、GPU、CPU神经或网络；通过不等于真实物理成功或P1科学准入。

实际saved-only `material(...)` 内存构建通过，canonical **13,546,620 bytes**，44,517 bindings、8 anchors、40 jobs，低于64MiB prepare cap；未写prepare，未调用teacher.consume、模型或环境。该尺寸包括完整来源closure、全部51帧/15动作的八anchor身份、原Q/context和全部40 raw10，不裁剪。输出1GiB为硬上界，不是预言实际压缩大小；报告16MiB、未压缩journal128MiB另硬计，MAIN需按输出cap+200MiB floor做启动预检。

cwd `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_agent_p1_teacher_cross_library_physical_20261028_v1.py
.venv/bin/python -m experiments.spectral_reachability.agent_p1_teacher_cross_library_physical_20261028_v1 prepare --output results/agent_p1_teacher_cross_library_physical_20261028_v1_prepare_MAIN02.json --compiler-output results/agent_p1_teacher_cross_library_20261028_v1_cuda_MAIN01 --compiler-report-sha d5ab99d55bacd73c8d9cf0fb774819bb46af63100053198858e639d8fa746738 --compiler-consume results/consume_teacher_cross_MAIN_20261028_v1_MAIN01.json --compiler-consume-sha d82109eb8de582e59a4f0485307c01c9b91f9f84551c218c61cca4da24335213
.venv/bin/python -m experiments.spectral_reachability.agent_p1_teacher_cross_library_physical_20261028_v1 verify --prepared results/agent_p1_teacher_cross_library_physical_20261028_v1_prepare_MAIN02.json --prepare-sha ACTUAL_PREPARE_SHA
.venv/bin/python -m experiments.spectral_reachability.agent_p1_teacher_cross_library_physical_20261028_v1 run --prepared results/agent_p1_teacher_cross_library_physical_20261028_v1_prepare_MAIN02.json --prepare-sha ACTUAL_PREPARE_SHA --output results/agent_p1_teacher_cross_library_physical_20261028_v1_MAIN02 --authorize-physics
.venv/bin/python -m experiments.spectral_reachability.agent_p1_teacher_cross_library_physical_20261028_v1 consume --output results/agent_p1_teacher_cross_library_physical_20261028_v1_MAIN02 --report-sha ACTUAL_PHYSICAL_REPORT_SHA
```

SHA占位仅MAIN实际创建后填写；worker不创建prepare或MAIN产物。run/consume打印完整report/40rows，不是compact摘要；MAIN可用独占wrapper保存返回值并只打印compact摘要，不改冻结CLI。新40库实际物理、完整消费与后续RGB神经评分尚未由worker执行。两known roots、同elapsed不等于同state/velocity/contact/cache；旧mixed selected仍stage-greedy，C2非贪心前提缺，任何结果不自动开P1、独立确认或否定理论。P1=false。
