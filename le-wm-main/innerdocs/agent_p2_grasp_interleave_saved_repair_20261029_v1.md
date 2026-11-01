# GRASP v2 saved-only 初始化修复：仅实际 MAIN01

不修改v2 producer、prepare、report或原两个side。绑定实际 report SHA `491367c06c8665c76c4aa3051e4f22db2554d0e44ded068ddb7b3d00d0448f87`，原host.json SHA `b1fe5475838e5f1dd21b17fa036684bf45804190cb7d2379f0ed8412538ed311`，original.json SHA `a2c3875fa9092afd7461531ce318a192b612ee25c9869d174a508b779ed2386a`；原/host保存输入完全相同。原v2实际初始化为 `torch.linspace(0,1,4,device=cuda)[1:-1]`，但没有持久化fraction；旧saved代码用 `np.asarray([1/3,2/3],float32)`，其位模式 `3eaaaaab,3f2aaaab`。逐位复算：第0槽0/192差异，第1槽136/192差异，最大绝对差2.384185791015625e-7。以固定诊断位模式 `3eaaaaab,3f2aaaaa` 做逐个f32操作得到的384个 `v` 与保存值**全部bitwise相等**；变更任意bit仍拒绝，没有容差。

这仅证明后者是与实际保存值一致的一个重建，并不能确定未保存的CUDA linspace真实位模式或作为通用CUDA规律。安装头文件只声明CUDA接口，不包含该kernel实现。原正式GRASP producer另保存fraction供其consumer核验，提示了此次probe遗漏；本件不以结果替换原producer值。精确report pin拒绝其他实验。余下保持原v2 full saved校验、输入/trace/账本/每packet/损失和梯度对比；不重新运行GPU、不重算神经或独立梯度、不声称模型新结果或科学门开放。产物仅保存当前诊断结论及新源/测/文SHA。

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p2_grasp_interleave_saved_repair_20261029_v1.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p2_grasp_interleave_saved_repair_20261029_v1 consume --output results/agent_p2_grasp_interleave_cuda_20261029_v2_cuda_MAIN01 --report-sha 491367c06c8665c76c4aa3051e4f22db2554d0e44ded068ddb7b3d00d0448f87 --receipt results/agent_p2_grasp_interleave_saved_repair_20261029_v1_consume_MAIN02.json
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p2_grasp_interleave_saved_repair_20261029_v1 verify --output results/agent_p2_grasp_interleave_cuda_20261029_v2_cuda_MAIN01 --report-sha 491367c06c8665c76c4aa3051e4f22db2554d0e44ded068ddb7b3d00d0448f87 --receipt results/agent_p2_grasp_interleave_saved_repair_20261029_v1_consume_MAIN02.json --receipt-sha ACTUAL_RECEIPT_SHA
```
