# P1 midpoint 探索：主机 CUDA 与保存验证

主线程全文审读宿主和模型内部 midpoint 方法，独立联合 19 项测试通过。真实 prepare `le-wm-main/results/agent_p1_midpoint_host_cuda_20261029_v1_prepare_MAIN01.json` SHA256 `2e778d4abc80ef081d82578b780695fbd2e39cd7e26be770317ed8ff30707d8a`，完整来源重建 verify 为 production_ready=true。固定四槽、两个已知根，不训练、不新增物理或编码，不改变科学门。

主机 RTX 5060 Ti CUDA 报告 SHA256 `f7afee2f482e40e4d1d09adda44d659a12433c23f8bba261cc2e842415b5b140`，主线程完整 saved consume `le-wm-main/results/agent_p1_midpoint_host_cuda_20261029_v1_consume_MAIN01.json` SHA256 `b1df0f9a22d87e8ed11ca2e176f23f8709391f7f52016a701528314bf0168a8d`。四槽均 sealed、errors=[]，104 predict 和 32 scalar reverse 均完成；峰值 allocated/reserved 为 140757504/144703488 字节。报告采样 wall 55.129 秒不含最终报告持久化，不能称完整端到端时间。

四槽选中迭代分别为 8/8/8/8，均等于该次原编译器的原 winner；中点均满足本探索规则的严格 Q 顺序。这说明在这四个已知槽中该方法并未改变选择，不能据此认为找到新的物理控制改善，也无法证明模型预测对真实物理排序可靠。中点方案系事后提出，不是前瞻新根；与旧物理运行的动作和上下文逐位等同性仍须核验，不能未经比较便宣称重复或物理结果相同。P1 admission 和 physical release 仍为 false，尚未执行本批新物理。

后续 MAIN 直接核对上述 midpoint receipt 与旧物理 strict receipt `le-wm-main/results/agent_observed_suffix_physical_20261028_v1_consume_MAIN01.json`（SHA256 `268074c84ca2aa0b3b1ae3354894021e0e4e06590fda6cb4cde89e4bb0cc0b6a`）：四槽 selected commands、compiler plan 的 raw float32 bits、observed-context SHA、model/task binding 和 prefix bits 均相同。旧物理四槽均 horizon_completed，initial_exact 和 prefix_reference_exact 均 true，native success 均 false。该对照直接读取已验证 receipt，未重跑旧 full strict，也没有开展新物理；它支持复用相同动作条件下的已有观察，不是一次新独立试验或一般性理论结论。为避免重复，本次不重新执行相同四条轨迹。
