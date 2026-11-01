# Same-prefix actual：独立saved审计

仅审计MAIN真实完成的246分支，不import producer/checker、不构造环境、不运行模型，
不修改任何封存源码/prepare/results。使用experiment-code组织实现和测试，但这里没有新实验或图。
实际report固定SHA `d975ef03f1b7fa540b449a403fec2fa83cf5eb9a89d4dcedbaf6555efe32c4fe`；
prepare固定SHA `fe0d4580530cbdd10838b4eb838a202cfff5d91ab6561183cde3292a9fa46018`。

独立重建两个旧calibration source roots×zero/bank_greedy/two_stage×41 probes。
读取原始task/Q/目标/64个model candidates，复算16个step5 Q，稳定选greedy；two_stage继承原64个
完整H2成本稳定赢家的prefix。roles函数仅接收task与model saved arrays，没有physical rows输入。
原H2候选仍用只读core做action/context合同校验，Q另行NumPy复算。没有新Q或outcome选动作。
逐字段重建所有plan与selection SHA，而不是调用旧collect/verify/analyze。

逐分支核全部10个实际command、raw tanh、native successor/reward/termination/goal/pose、guard顺序；
原native eval的角度公式（包括跨圈时可能为负）原样保留，与wrapped angle/object XY诊断分开。
逐0/5/10 checkpoint核两body position/velocity/unwrapped angle/angular velocity/force/torque/sleep，
核这些字段与state7的投影，并在同role的41个probe间强等step5 RGB/state7/全部已记录动力学。
这**不保证solver warm-start/contact cache或完整隐藏模拟器状态一致**，也不是重跑物理验证。

流式读取完整journal，逐行验证index/previous/event SHA，并独立消费34,196条事件。
每一attempt/return、reset seed/placement options、fresh render、checkpoint、guard、step receipt、
branch close与真实report交叉核对。独立累计492 reset/2460 step/2706 explicit render；
738 PNG全部校验文件SHA及解码uint8 RGB SHA，保留初始RGB与raw source exact绑定。
原始placement request/随机reset observation在journal中交叉核验；不重跑几何采样或reset内部物理。
仅0/5/10保存像素，其余fresh RGB只有durable receipt，不能重新验证其所有像素。
source/input/model绑定先hash、读取后再hash；完整741文件集合与64MiB实际大小上限核对。
无独立硬件neural meter：零新neural来自封存producer代码合同及本审计无模型调用，不是假造GPU测量。

独立复算所有246个原分析响应，并与analysis.json整体严格相等；另算object-only XY向量、
目标位置误差变化、wrapped object角位移。报告仅保存六组描述范围和全部响应digest，
不复制大数组/闭包，不挑probe为控制器。native mixed-unit distance不能冒称object task metric。
实际全部fullH2，partial计0；缺行/partial输入严格拒绝，不悄悄丢弃后按fullH2汇总。
旧两根、重复RGB causal padding、无独立confirmation/预声明准入阈值；P1始终false。

测试只读实际fixtures或内存mock/篡改，不创建pytest tmp，不运行神经/physics。
新增report exclusive小文件≤64KiB，仍保留200MiB floor；journal流式读，不复制37MiB输入。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_same_prefix_saved_audit_20261027_v1.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_same_prefix_saved_audit_20261027_v1 --output results/p1_same_prefix_saved_audit_20261027_v1_main01.json
```

MAIN复验请用新output名称，不能覆盖既有报告。审计成功仅表示saved数据/合同一致，
不是全局可控性、优化器、物理新收益或P1科学准入。
