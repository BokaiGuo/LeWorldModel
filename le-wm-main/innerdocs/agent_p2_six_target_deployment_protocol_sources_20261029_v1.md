# Six-target deployment protocol：实际来源补记 v1

本补记只更新实际来源状态，不修改已冻结协议、资源公式、动作/目标选择或放行条件。

原协议：`innerdocs/agent_p2_six_target_deployment_protocol_20261029_v1.md`，SHA256 `97d81f6969e0ac52e808893be09dfacf0180ed04ab0955fe38d202563a32744f`，保持原字节。

以下路径相对 `/home/gbk/code/LeWorldModel/le-wm-main/`，worker已只读核对文件SHA：

| 实际来源 | SHA256 |
|---|---|
| results/agent_p2_six_target_controller_cuda_20261029_v1_prepare_MAIN01.json | cccfadc20911dc1be008a047c57bca7400272f6602deb435fb0341a0c8bbc8f9 |
| results/agent_p2_six_target_controller_cuda_20261029_v1_cuda_MAIN01/report.json | b81f12ee3602835bc14fd2e22d7e3644b8376f7ba8b0b265948bfb18429b6934 |

MAIN报告：HOST CUDA六槽均returned，72 predict API / 324 samples，errors=[]，beforefooter时间76.546s。这些为MAIN回传的producer状态，不冒充worker独立完整消费；beforefooter不等于完整job末尾落盘墙时。

完整consume仍由MAIN执行；本补记不填写receipt路径或SHA，不推测成功。原协议“six controller实际来源尚未提供”的时间状态现部分更新为prepare/report已到、完整receipt未到，其余前置不变。

以上normalized mean及其原双replay不替代实际raw inverse/re-normalize后的付费部署token重放。六槽新paid authority、domain及原生first-step guard仍须分别满足；不读取结果选槽、不裁剪、不拼旧四槽物理row、不签训练normalizer身份或P1/P2科学准入。

本次仅新增此补记与SHA核对；未运行prepare、consumer/fullstrict、NN、physics或子代。待MAIN提供完整实际receipt后再按授权接下一实现slice。
