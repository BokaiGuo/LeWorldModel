# 全方向固定库：完整物理重放与 MAIN 独立指标核验

MAIN 完整审读新executor、saved复用路径和测试；113项测试通过（34.19秒）。旧teacher物理完整consumer复用检查得到reused0/unknown4，完整raw10不同，不能继承。为补齐本轮winner真实执行链，另在主机执行固定四项，不挑成功分支，不从未知补0。

prepare SHA256 `7606acbdb575e4727fb81f280c92ef7326a8c16189805ed262430c2abb8a6560`。真实运行退出0，report `le-wm-main/results/agent_allmode_physical_20261028_v1_MAIN01/report.json`，SHA256 `3ca4891a5f0823bede431e85442513a095b1e9bc8d49c87feefbb56663dca3b4`。errors=[]，四项完整horizon，无早停；完整warmup/history/root/goal/PNG身份检查通过。新增4constructor/close、8reset、80public steps（40warmup+40规划动作）、816内部physics steps、264显式render；这些是嵌套轴，不相加为总调用。

原生成功0/4。MAIN 独立公式复算联合位置、角度及7维距离，通过严格consumer全journal/PNG/来源闭包验收，持久回执 `le-wm-main/results/consume_allmode_physical_MAIN_20261028_v1_MAIN01.json`，SHA256 `c26b2dd15b69a1f3532a5e7fd88ad3f39acb7f84071e34101c5711673d8b4acc`。

| 根/条件 | 初始联合XY | 首段5步 | 终态10步 |
| --- | ---: | ---: | ---: |
| 0/off | 179.737797 | 61.395284 | 56.846847 |
| 0/on | 179.737797 | 62.287987 | 69.805308 |
| 1/off | 223.018422 | 74.592578 | 56.015615 |
| 1/on | 223.018422 | 73.026302 | 52.503513 |

原生成功要求joint XY<20且原角度项<π/9。原生reward distance另含7维状态（包括速度）；不能用reward大直接解释成功率，也不能用object-only接近宣布原生成功。保留原始角度公式，wrapped诊断不改标签。

全部首段均已改善联合位置，故这些选择不构成“短期退步、改变能力、随后受益”的P1见证。动作与原轴库选择不变，结果用于完成严格执行链，不是增加独立样本或发现新算法收益。两个known roots×off/on仍仅两个已知根；无新编码/模型调用，原神经计算来自主机CUDA，无训练或CPU神经回退。

容量工程错误与本轮控制未成功分开：前者已修复；本轮重放无工程错，但不能据固定库四项失败证明理论不可行。混合方向候选仍在另版本实现，尚无该轮新结果。P1/C2、独立确认和研究总goal仍未完成。
