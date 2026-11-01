# Archive4 / live4：固定输入统计对照实验（非学习）

本NEW实验只有同stem源码/tests/本文；主线程负责审读，不设独立审计agent，不新增auditor。
使用experiment-code固定协议、持久输出和失败保留原则，不执行神经/物理、训练、拟合、选择或网络。
不重复overlay F/Z/M干预，不改任何sealed脚本/结果/共享文档。输出是描述性输入统计，不是机制证明或P1准入。

## 预定来源与分母

archive使用既有offline v2 prepare，外部SHA固定
`580dd9e35431423885041da76bbdbca3a033808dae7db40aabbaff25e1c34adb`。
只从其绑定的原本地HDF5读原ordinal `[0,6228,12456,18684]`，每条RGB0/5/10/15/20和raw action[0:20)。
核对stat版本、offset/length/末episode边界、episode_idx/step_idx及既有选定pixel/action arrayentry SHA。
不读state/reward/goal，不遍历46GB数据、不做全库stats或训练来源推断；不重建PNG、不复制数据。
训练checkpoint、transform、normalizer均不加载；此实验在**原uint8/raw动作尺度**上比较，不拟合或替换public stats。

live使用原physical四分支report，外部SHA固定
`d205850e95f639b649fba9b5c6ee8059f2507431753a538b1ce30eccbd234300`。
固定root0 GN/D2P、root1 GN/D2P顺序，不按成功、移动、Q或任何outcome挑选/重排。
只投影report/journal中的branch/selection身份、执行raw动作和选定PNG文件/pixel SHA，不访问hidden state或native labels字段。
不打开sourcepack/certificate/witness/task/latent文件，不调用任何旧producer或verify/环境函数。
四分支来自**两根**，不是四独立样本；archive四ordinal lineage亦unknown。共享history会如实计重复而不去重分母。

共同描述范围：两组各4序列×RGB相对0/5/10=12帧、各4×raw前10步=40动作。
archive原15/20帧及后10动作额外单列，不掺入共同分母。
archive是episode原起点，live是warmup后plan起点；它们不是相同物理状态、配对任务或匹配control phase。
此范围只为固定描述，不拿TV距离或幅度差推断泛化、训练成员关系或因果归因。

## 最小统计与精确语义

- RGB：原0..255逐通道mean/population std，预定16等宽bins `[0,16,...,256]`的counts/fractions；
  选定相邻帧raw RGB平均绝对差与变化pixel比例，间隔明确为5 public steps，不称逐步速度。
- raw动作：逐坐标mean/std/min/max，二维norm的mean/RMS/max，五步packet flatten10的L2；
  相邻命令L2差明确是`command_change_per_public_step`，没有时间/物理位移依据时physical_speed=null。
  绝不clip/tanh；abs>1只描述计数，不拒绝合法factual专家轨迹。
- context：三张真实history RGB、历史两packet+首outgoing的逐entry精确重复pair数/固定分母；
  整段RGB history/原raw action history也跨四slots报告重复率。只认dtype/shape/bytes一致，不等同state相同或样本独立。
- group对照：固定直方图每通道TV、live−archive均值和raw幅度均值差；无拟合classifier、阈值、显著性检验或择优。
  pooling不制造跨分支相邻时间delta，只合并每条内部9次命令变化（4×9=36），不把39次拼接变化当真实时序。

**两种live RGB角色分开**：部署起点真实context是warmup0/5/10，raw两包历史来自实际journal中的零动作，
不是合成zero padding。plan0必须与warmup10像素相同。plan0/5/10是后续真实观测，用于输入描述/评价；
原planner open-loop，不能称新RGB已经作为feedback/model context消费，也不拿其替换预测label。
archive W10/W15两实际history单列；不假造live第二轮observed模型调用来配齐它们。

## 失败与资源

每组预定四分母；单行缺PNG、错SHA/动作、缺时刻或身份不符，该行标missing_or_invalid，statistics=null，
不补零、不换episode/分支。相应组pooled=null，跨组comparison=null，保留四行并以退出码2交付partial。
顶层来源SHA、schema/固定角色或全库stat错误则直接拒绝，不生成貌似有效的报告。
只读文件上限8MiB、单PNG1MiB、合计小文件64MiB；原HDF5只取固定slice。
输出NEW单个JSON≤512KiB，exclusive创建/fsync目录，每写保持200MiB reserve，不输出图/大fixture/复制protocol。
文件SHA前后重核，HDF5末尾再查stat。输入统计运行本身零神经/物理调用。

## MAIN可复验命令

内层项目根目录`/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_constructive_archive_live_input_summary_20261027_v1.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_archive_live_input_summary_20261027_v1 --archive-prepared results/p1_constructive_offline_fidelity_cuda_20261027_v2_prepare01.json --archive-sha 580dd9e35431423885041da76bbdbca3a033808dae7db40aabbaff25e1c34adb --physical-report results/p1_constructive_plan_physical_20261027_v1_main01/report.json --physical-sha d205850e95f639b649fba9b5c6ee8059f2507431753a538b1ce30eccbd234300 --output results/p1_constructive_archive_live_input_summary_20261027_v1_main01.json
```

tests只用RAM数据/PNG和小文件只读，mock通过不等于真实统计或科学准入。MAIN可先import/build仅在RAM验证，
不必生成report文件。没有可据本实验宣称的training provenance、overlay因果解释或P1通过。

## 本次实现验证进度

36个RAM测试通过，覆盖固定slice/末episode、超动作球raw保留、精确算术/重复分母、live输入绑定、
missing行保留、不跨序列伪造时差、partial抑制跨组比较和200MiB写入floor。

实际archive prepare＋原四分支report已在RAM完成build，4/4 archive、4/4 live输入可描述、errors为空，
未导入torch/gym、未落盘实际report；预计JSON约77KB、27个小文件输入绑定。
仅描述性发现：live四root RGB history只有2种（符合两根双臂共享起点），四段真实warmup raw history只有1种；
archive四段W10 history RGB/raw均4种。不是把重复除掉后新估计独立样本量，也不推断archive训练身份。
common边际histogram TV约RGB[.0282,.0221,.0247]、raw magnitude均值live−archive约.2264；
这些尺度/统计单独不能解释预测或控制表现，更不证明overlay因果。

没有独立审计agent或独立auditor；源码可交MAIN审读，纯saved正式报告由MAIN按上面命令生成。
