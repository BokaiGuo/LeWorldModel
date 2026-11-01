# Same-prefix反馈后缀：独立saved对照

主线程新增纯saved分析，不导入任何producer、模型、环境，不运行神经/物理，
不拟合目标、不重新选择动作。必须外部SHA明确绑定旧RGB、新RGB、suffix报告、
独立prefix审计。固定四分支，缺项或不一致直接拒绝，不筛负例。

独立复算两种已执行后缀的原identity192 Q，以及observed-context模型的三候选Q；
检查原有stable model-only winner。实际prefix latent须float32 bytes相等，
独立prefix审计须覆盖所比较的两个物理report且不是自比；原后缀未变的控制分支
实际terminal latent也须bytes相等。报告明确P1=false、非独立确认、无唯一根因判定。

本工具不重审全部CUDA内核、物理积分、producer的完整源码闭包或模型权重；
这些依赖上游已封存报告和独立审计。对现存NPY/JSON的算术复算不能代替新根确认。
所有输入hash读取后再验，单文件<2MiB、gzip展开≤4MiB；输出独占小文件≤128KiB、
200MiB floor，不写进任何旧封存目录。结果在完整检查通过后才写入。

CPU执行仅限NumPy/saved诊断；没有CPU神经回退。测试全为内存数组反例。

独立review发现早期分析器没有直接核对winner.commands与plan尾5条命令；
这是分析器的漏检，不是实际执行错配证据。主线程补齐完整winner→suffix→
full-plan→physical dispatch绑定，并加入完整analyze的实际saved内存篡改反例。
17tests通过；main01保留，main02为补强后完整复算，得到2changed/2strict inversions。
main02路径 `results/p1_constructive_feedback_contrast_20261027_v1_main02.json`，
SHA `f6f061e117fbf37a465044d69d57cac17a6b57ba2df751dc66a578f514557149`。
测试中两项只读实际保存链（不存在时skip），不产生临时文件；其余为内存算术。
