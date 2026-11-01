# PushT：动作统计与真实物体响应审计

本轮为原D2P P1的代码/目标合同排查，不是新校准或outcome，不改变任何
部署常数、程序选择、根集合或阶段门。没有神经计算或新物理执行；对既有
CUDA/真实执行证据复算，不能将本轮计成新增CUDA样本。

## 归一化：有数据差异，尚无权重训练溯源

仅顺序读取本地44GB HDF5的action字段，2336736条float32动作；分块65536行，
不读pixels/state，不哈希整个HDF5。逐行剔除含NaN动作后，float64合并矩计算
均值及ddof=1标准差，与当前训练源码统计语义一致，但非Torch float32归约的
逐位复现。实测没有NaN行。

| 统计 | HDF5动作字段 | 公开示例常数 |
| --- | --- | --- |
| mean | [-0.0078125644, 0.0068606872] | [-0.0087, 0.0068] |
| sample std | [0.2084674875, 0.2067486706] | [0.2019, 0.2002] |
| std比值 | [1.0325284, 1.0327106] | [1, 1] |

两者并非四位小数舍入意义下的相同统计。字段最小值[-1.49471,-1.46553]、
最大值[2.04267,1.79206]也说明全档案动作不全在[-1,1]内；统计没有裁剪，
没有按部署动作盒筛数据，也没有因此放宽现有程序的合法动作合同。

这能否定“本地HDF5的精确全量动作统计就是示例常数”，但不能证明示例常数
不是官方训练Lance版本的统计。当前权重config只保存结构，未绑定训练数据
版本/normalizer，缺失的溯源仍缺失。不得直接替换常数、根据控制结果选统计、
或将约3.3%的差异宣布为P1失败的因果根因。

动作字段独立SHA256：
`db62c4b5317b53e1e256140146b512419a79f27f805f411637a18be8bcc18011`。
这不是全档案内容哈希或checkpoint训练数据认证。

产物：`le-wm-main/results/pusht_d2p_archive_action_statistics_v1_20261026.json`。
实现与不可覆盖源码归档：
`le-wm-main/experiments/spectral_reachability/audit_pusht_d2p_action_statistics.py`。

## 真正执行了什么：推杆变化与物体变化必须拆开

对完整既有4根×16候选×3求解器的192条真实分支复算。在解释前核对64个唯一
root/proposal身份、全部solver、上游协议、预执行计划与观察后后缀因果收据。
状态布局取安装环境`_get_obs`：[agent xy, block xy, block angle, agent velocity]。
物体未变采用保存坐标的精确相等，不引入新阈值。角位移另作wrap数值报告。

| 后缀求解器 | 物体姿态精确未变 | 推杆动而物体未变 | 未变但预测RGB改善 | 未变但真实编码RGB改善 | 物理pose改善/损害 |
| --- | ---: | ---: | ---: | ---: | ---: |
| random8 one-shot | 51/64 | 51/64 | 39/64 | 29/64 | 2/11 |
| full192 one-shot | 47/64 | 47/64 | 46/64 | 33/64 | 3/14 |
| full192 GN4 | 40/64 | 40/64 | 40/64 | 28/64 | 4/20 |

这些是全部分支的描述统计，不按真实收益重新选候选，且各求解器预算不等。
三者可有更好的编码RGB距离而物体终点完全不变，因此“优化目标下降”不能
替代任务进展。增加优化确实让更多分支改变物体，但损害数也增加，不能称
GN已经修复。物体终点未变也不证明中途从未运动或从未接触。

同样检查原4根封存selector的校准：4个selected后缀均未改变物体姿态。
其零前缀后物理pose都等于真实起点，所以本批3/4非贪心失败不能靠把zero
参照改成initial参照修复；这里不存在Reacher中需要单独核对的coasting差异。
以上只定位到当前短后缀/任务目标的实现缺口，尚无充分证据排除模型动力学
误差、有限horizon或接触配置影响，更不能宣告原理论被否定。

产物：`le-wm-main/results/pusht_d2p_suffix_object_response4x16_v1_20261026.json`。
实现：`le-wm-main/experiments/spectral_reachability/audit_pusht_d2p_suffix_object_response.py`。

## 实际命令与验证

在`le-wm-main`执行：

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.audit_pusht_d2p_action_statistics \
  --dataset /home/gbk/.stable_worldmodel/datasets/pusht_expert_train.h5 \
  --checkpoint-config /home/gbk/.stable_worldmodel/checkpoints/pusht/lewm/config.json \
  --output results/pusht_d2p_archive_action_statistics_v1_20261026.json

OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/lewm-matplotlib .venv/bin/python -m experiments.spectral_reachability.audit_pusht_d2p_suffix_object_response \
  --report results/pusht_d2p_full_metric_bank4x16_20261026_cuda/report.json \
  --output results/pusht_d2p_suffix_object_response4x16_v1_20261026.json
```

新增16项测试通过，完整D2P相关回归1607项通过（49.60秒），`git diff --check`
通过。新增测试涵盖分块统计一致性、NaN行语义、动作字段定界读取、
退化/非有限输入拒绝、物体/推杆分离和角度wrap。测试与保存数组复算均非
P1效能证据。研究仍缺物体目标一致的程序生成、真实能力改变、独立校准与
等预算强基线闭环；不打开P2/P3/P5。
