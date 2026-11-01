# P3 固定候选 quota：主机 CUDA 探索运行

固定 prepare `results/codex_p3_quota_host_20261030_v2_PREP_01.json` 的 SHA256 为 `2327c5ab26642678e923a2cbdf6510a9985f51f3aa7f4d09e33970c4ab5c898e`。固定历史来源、两个已知根、每根 cold bank 前 15 项、权重 `[162.32405169996156,156.2867146609139]`、三臂顺序和资格配额。此合同在本次运行之前固定，但已知历史结果，非事后声称外部预注册。

`results/codex_p3_quota_host_20261030_v2_RUN_01` 在合作锁内完成主机 CUDA 候选面板，无物理执行。父进程持锁并等待子进程退出；锁仅约束合作入口。report SHA256 `f3f818f2026bc86df90983ad31f52f81a6ddc1c49175b24fbf7e40c08f699565`；journal tail `d3a0bd503f12f5fd70be6bcfd0321583cd91b855d58fc6904b0b023a9d934cfe`。模型前后哈希均 `4b12c01f5859f4b46bea478f3a8d5e2d3f516df456a622a3feec31bde7e33546`；runtime UUID 与设备查询匹配 `GPU-4e8973ff-0278-53f3-2bae-2615d461e441`。运行报告含实际已加载 Python 文件清单与 SHA，但动态/原生代码完整闭包未知。报告时间 9.145 秒为该进程计时口径，不是各臂 wall 等价。

独立 saved 语义 consumer 成功：90 个候选，720 个实际模块/API attempt 和 720 个 return；四类轴各 180。逐候选重建对应配额、完整轨迹和模型成本及逐根赢家；未独立重建模块数组，不认证 source/physical/scientific。该 consumer 的成功输出为 stdout，未在原结果目录补写认证文件。

| 臂 | root 0 胜者 / model-Q | root 1 胜者 / model-Q |
|---|---|---|
| uniform 10/10 | 1 / 0.7874349800 | 1 / 0.6231905744 |
| signal 15/5 | 12 / 0.7485899228 | 1 / 0.6231905744 |
| reverse 5/15 | 1 / 0.7874349800 | 1 / 0.6231905744 |

结果只是已知根、模型目标下的描述性比较；第一根 signal 与其他臂候选不同，第二根碰撞。各臂全部计算 15+15，差异是资格集合，不是节省计算；不表示等 FLOPs、等全成本、独立确认或物理收益。此前 v1 命令在 CUDA 初始化前因错误的 prepare reader 失败，未生成运行目录；v1 prepare 保留。v2 沙箱初次预检因 nvidia-smi 无设备而停在创建目录之前，之后在授权的主机环境执行成功。

离线测试：v2 host、身份、journal、bridge 与 consumer 共 15 passed；RAM 测试不是神经/物理证据。源与 gate 均未改。
