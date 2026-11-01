# 可观测性 guard 已接入真实执行器

2026-10-26 主线程完成主机验证；原动作、selector、候选库、根与目标不变。不是恢复控制或 P1 成功证据。

## 代码与实际产物

- 执行器：`le-wm-main/experiments/spectral_reachability/pusht_visible_command_executor.py`。每primitive先验证新鲜RGB，再授权原命令；真实执行后立即重新观察。规范推杆外观缺失后不再执行后续命令；不填零、不盲重试、不偷偷选择其他候选。动作错误或step异常另记execution_failed。
- runner：`le-wm-main/experiments/spectral_reachability/run_pusht_visible_executor_cuda.py`。
- 产物：`le-wm-main/results/pusht_visible_executor4_v1_20261026_cuda/report.json`。integration_completed、进程退出0、9项gates全true、64次实际CUDA终态编码、0predict calls。参数、缓冲区、frozen/eval、输入及来源封存核验通过。纯RGB presence检查不使用额外神经模块，没有CPU神经回退。

## 64 个原程序全部重放，保留全部分支

| 根 | 无观测缺失、完整执行 | 观测缺失程序 | 原selector所选程序 |
|---|---:|---:|---|
| 0 | 16 | 0 | prefix12：完整10步 |
| 1 | 12 | 4 | prefix13：完整10步 |
| 2 | 13 | 3 | prefix4：完整10步 |
| 3 | 7 | 9 | prefix3：第5步观测缺失，后5步未执行 |

16个观测缺失程序中，13个发生在第10步之前，3个恰在最后一步。前者实际中止后续执行，后者标记终态不可观测但已无剩余动作。总计44条后续盲命令未执行，实际执行原planned primitive commands 596/640（不含共有的根重建warmup）。不能把16都描述为“提前中止”。

48个完整执行程序的终态RGB哈希与物理状态均与原无guard执行精确一致。若5步前缀完成，runner同样核验前缀RGB/状态与原执行一致。每步记录实际命令、RGB哈希和评估用状态，guard决策仅用RGB；真实状态从不进入guard或候选选择。

## 解决了什么、没有解决什么

修正了此前“Box内合法动作即默认可继续RGB控制”的执行合同漏洞。失败程序并未从样本中删除；观察缺失是一种显式abstention/域失配，不自动变成成功。旧结果不重写，也不将新短轨迹冒充原完整轨迹。

guard只支持显式声明的默认RoyalBlue renderer，presence不能证明完整状态可观测、无碰撞或安全；它不能防止第一次出画面，也没有恢复回画面的控制。正常程序亦无新收益。当前没有进行新根泛化或强基线比较，不能以“阻止44条命令”替代物理任务成功。

未来正式协议必须冻结同一逐primitive反馈频率与mask检查开销，所有比较臂一致；不能仅给D2P额外反馈、事后过滤不可见根或把真实坐标当规划输入。要继续证明P1，仍需可执行的非贪心重配置、真实能力改变、独立根任务收益与等预算对照。

本轮 executor/guard 数组与假环境合同16项测试通过；真实执行证据以上述CUDA/物理产物为准。
