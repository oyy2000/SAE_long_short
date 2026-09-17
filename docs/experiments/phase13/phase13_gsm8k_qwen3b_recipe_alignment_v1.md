# GSM8K 3B：对齐历史 1.5B 训练配方的复跑

## 目的与范围

用户于 2026-09-16 确认保留当前 1,024 道训练题及教师 trace，对齐历史 1.5B 学生的训练和评测配方。此分支用于判断 3B 首轮 SFT 下降是否与训练配方有关，不等于复现旧 878 题实验，也不构成新的独立测试集确认。

两轮教师均为 **Qwen/Qwen2.5-7B-Instruct**，revision `a09a35458c702b33eeacc393d103063234e8bc28`。学生仍为 **Qwen/Qwen2.5-3B-Instruct**，revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`。新分支不重新生成教师答案，不覆盖原 adapter 或预测。

## 配方对齐

| 设置 | 原 3B pilot | 本次复跑 / 历史 1.5B 配方 |
| --- | --- | --- |
| Epochs | 3 | 1 |
| 每 GPU batch / 梯度累积 | 1 / 8 | 4 / 1 |
| 学习率 | 5e-5 | 2e-5 |
| 学习率调度 | cosine | linear |
| Warmup ratio | 0.1 | 0.03 |
| LoRA rank / alpha / dropout | 8 / 16 / 0 | 4 / 16 / 0.05 |
| LoRA 模块 | 七类线性投影 | all-linear |
| 训练最大序列长度 | 8,192 | 2,048 |
| 精度 / gradient checkpointing | BF16 / 开启 | BF16 / 开启 |
| 学生 seeds | 17；额外 seeds 受旧 gate 约束 | 17、42、73，全部预先登记 |
| 评测 batch / 生成上限 | 8 / 1,024 | 32 / 512 |
| 评测解码 | greedy | greedy |
| Repetition penalty | 显式 1.0 | 显式 1.1，匹配旧 1.5B 的实际默认值 |

历史配置未显式填写的 linear scheduler、max_grad_norm=1.0、weight_decay=0.0，在准备作业中对照原冻结训练包装器及同一 pinned 运行环境的 TrainingArguments 默认值确认，并在新配置中显式填写。

当前 1,024 题、学生提示词、判分器、预编码 completion labels 及 native EOS 监督均保持不变；旧实验使用文本格式的 completion-only collator。因此本次对齐的是优化配方及解码预算，不声称所有训练实现细节相同。保留显式 labels 时 `completion_only_loss=null` 表示不额外启用文本 collator，实际 prompt labels 仍全部屏蔽；并非训练 prompt loss。

## 第一轮预先登记的比较

- B1：7B 无干预候选中的最短正确 trace。
- B7 0.1：SAE 强度 0.1 后选出的最短正确 trace。
- B7 0.3：SAE 强度 0.3 后选出的最短正确 trace。
- 每组使用同样 1,024 题和三个 seed，共九个 adapter；每次 256 个优化步、每题曝光一次。
- 未微调 3B Base 按相同 512-token cap 重新评测。此前 **86.84% / 310.73 tokens** 来自 1,024-token cap，不能直接作为本轮 cap 下的 Base。
- 每个模型评测 GSM8K `test[50:1319]` 全部 1,269 题，四分片；预期十个模型、40 个分片、12,690 条预测。
- 不按本轮测试成绩重新选择强度。报告各 seed、长度、cap-hit rate，以及两种 SAE 相对 B1 / Base 的题目与 seed 交叉 bootstrap；四个预先登记比较一起调整。结论保持 exploratory、GSM8K-only。

原十四条件 pilot 保留。本轮先回答配方及 SAE 0.1/0.3 的敏感性问题，尚未重新运行 B0/B2–B6，也不扩展到更强 SAE。

## 执行及证据

配置：[phase13_gsm8k_qwen3b_recipe_alignment_v1.json](../../../configs/phase13_gsm8k_qwen3b_recipe_alignment_v1.json)。入口：[13_64_run_gsm8k_recipe_alignment.py](../../../scripts/13_64_run_gsm8k_recipe_alignment.py)。复用现有训练、completion 曝光审计、GPU admission、预测核验和 hash-verified adapter 发布。

结果目录：`results/phase13_baseline_expansion_v1/exploratory/gsm8k_qwen3b_recipe_alignment_v1`。Checkpoint 目录：`checkpoints/phase13_baseline_expansion_v1/exploratory/gsm8k_qwen3b_recipe_alignment_v1`。

CPU 准备 850139 已完成，26 项回归检查通过，三组文本和 token 数据与父实验逐字节一致。最长序列分别为 787、786、632 tokens，无需截断。

完整 B1 / seed 17 训练 850141 已完成：256 步、1 epoch、1,024 题各曝光一次，训练用时 62.13 秒。显存 allocated 峰值 13,026 MiB，reserved 峰值 37,920 MiB；保留 8,192 MiB 安全余量后本轮不安排 L40S，继续使用 H200。评测仍须逐作业通过分配设备的显存 admission。

进一步核验旧冻结 evaluator 后，发现它继承 1.5B `generation_config.json` 的 repetition penalty 1.1；新的 `execution_v2` 将该值显式设为 1.1。旧评测 850142 及释放作业 850143 已取消，部分产物归档，训练不重跑。850140 已正常退出。修订后评测 850148 完成 shard 0 的 318 条预测，耗时约两分钟；实际使用指定 scratch。

测量释放 850158 已完成；其冻结控制代码区分实际 allocated 峰值与 allocator reserved 峰值，并将评测修订配置绑定到最终分析。新增三项控制回归检查通过。剩余八次训练、39 个评测分片和最终分析已提交为 850159–850206；分析为 850206，恢复控制器为 850150。恢复只覆盖本分支：保留失败产物，对节点、超时等可恢复失败重试；确定性配置、数据或依赖错误留待诊断。被替换的中间释放作业 850149 已取消。

2026-09-16 03:42 EDT：新矩阵尚未进入 GPU 执行，四个就绪作业等待 `QOSGrpGRES`（H200 共享 QOS 配额已占用），其余等待成功依赖。其他 `kd_pilot` 作业保持不动。这是调度等待；本轮已确认完成的只有首个训练与首个评测分片。

所有作业使用 `/share/jekml/youyang7/tmp/youyang7-phase13-frozen-<jobid>`，TMPDIR/TMP/TEMP、缓存、Trainer output_dir 一致。首个真实 Trainer output_dir 已核验。51 个修订提交的实际 batch 脚本与冻结启动器一致，均登记 execution_v2 评测配置；证据位于结果目录 `verification/initial_batch_audit.json` 和 `verification/matrix_batch_audit.json`。

提交和准备完成均不代表训练评测完成。最终状态以本分支 `EXPERIMENT_COMPLETE.json`、完整分片/adapter 标记及逐题审计为准。


## 完整结果与复核（2026-09-16）

独立 `gsm8k_qwen3b_recipe_alignment_v1` 已完成九个 adapter、含 Base 的十个评测条件、40 个分片和 12,690 条预测；850206 分析与各登记训练/评测作业均 COMPLETED/0:0。独立复核 236 个标记、6,925 项哈希与逐条件准确率/长度/cap-hit，零缺失或重复。每个学生保留同一组 1,024 题，1 epoch、256 优化步；本轮为旧配方对齐 SFT，并非 KD。

| 条件 | seed17 | seed42 | seed73 | 平均准确率 | 平均输出 tokens |
| --- | --- | --- | --- | --- | --- |
| B1 无干预最短正确 | 83.29% | 82.74% | 83.22% | 83.08% | 277.68 |
| SAE 0.1 | 84.63% | 84.87% | 83.92% | 84.48% | 271.34 |
| SAE 0.3 | 82.74% | 83.14% | 82.51% | 82.79% | 250.86 |

SAE 0.1 相对 B1 三个 seed 均正增益，平均 +1.39 pp、输出缩短 2.29%；题目与训练 seed 交叉 bootstrap 95% CI [−0.11,+2.86] pp，四比较 Holm 调整 p=0.204，尚不能确认优于 B1。SAE 0.3 相对 B1 平均 −0.29 pp、输出缩短 9.66%，CI [−1.89,+1.39] pp。相同 cap/解码下 Base 为 81.56%；SAE 0.1 相对 Base +2.92 pp、调整 p=0.0128，但论文关心的相对 B1 优势仍未获统计支持。

评测为已观察过的 GSM8K test[50:1319] 共 1,269 题、greedy cap512、repetition penalty1.1，属于 exploratory 配方敏感性，不能与 cap1024 的原 pilot 或 KD 开发集准确率直接混比，也不构成新独立确认。见 [结果图](../../../results/phase13_baseline_expansion_v1/exploratory/gsm8k_qwen3b_recipe_alignment_v1/analysis/student_accuracy.png)、[完整比较](../../../results/phase13_baseline_expansion_v1/exploratory/gsm8k_qwen3b_recipe_alignment_v1/analysis/comparisons.json)、[独立复核](../../../results/phase13_baseline_expansion_v1/exploratory/gsm8k_qwen3b_recipe_alignment_v1/verification/RESULTS_VERIFIED.json)。
