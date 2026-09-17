# 小规模 token 概率蒸馏试验

## 问题与边界

已有教师文本 SFT 属于序列级蒸馏。本分支新增教师 token 概率监督，先判断在相同数据与优化预算下是否优于纯 SFT。初阶段仅使用 B1 shortest-correct；不改变已完成的 GSM pilot，不据此宣称 SAE 蒸馏有效。

- 教师：冻结 Qwen2.5-7B-Instruct；学生：相同 revision 的 Qwen2.5-3B-Instruct。
- 从已审核的 1,024 道共同训练题中，以固定哈希顺序抽取 256 道。两臂使用完全相同的问题、文本、token、题序、LoRA、优化步和曝光次数。
- 学生 seeds 17、42；每个 seed 各训练 SFT 和 KD，共四次小试。复用原 3 epoch、batch 1、accumulation 8、rank-8 LoRA、learning rate 5e-5 配方，每次 96 个优化步。
- 开发集固定为已有 300 题，明确标注曾被原 pilot 观察；greedy、1024-token cap。不得读取 locked-test 预测用于调整本小试。

## KD 定义

默认目标为 `0.5 * CE + 0.5 * T² * KL(p_teacher(T) || p_student(T))`，`T=2`。CE 和 KL 都只作用于经过 causal shift 的 completion 位置，排除 prompt/padding，保留原 EOS 监督。教师在相同输入前缀上进行 teacher-forced forward，始终 eval/no-grad，无 SAE steering。

7B/3B 输出维度分别为 152064/151936。必须验证 tokenizer 词表、token-ID、normalizer 和 merges 一致，再在共同的 151665 个有效 token 上重归一化计算 KL；额外填充输出维度不具有对应 token，排除并记录教师被排除的概率质量。CE 仍按学生完整输出维度计算，与原 causal CE 一致。这是明确登记的词表对齐适配。

教师在训练时在线前向，不将完整 logits 写盘。按 64 个位置分块计算 FP32 KL；BF16 模型、SDPA、LoRA 与原训练保持一致。教师载入保存/恢复随机数状态，避免改变学生初始化之后的抽样随机流。复用 `training.py`、completion collator、曝光审计、greedy 评测、评分器及发布哈希工具。

方法依据：[Hinton 等的温度蒸馏](https://arxiv.org/abs/1503.02531)、[Kim 与 Rush 的序列级蒸馏](https://aclanthology.org/D16-1139/)。具体混合系数、温度和放大门槛是本小试预注册选择，不是文献保证有效的参数。

## 执行与准入

CPU 数值/掩码/梯度回归 → 数据与词表核验 → 最长 8 条中的 4-step H200 KD smoke → 实测显存/耗时门槛 → 四个完整小试 → 完整开发预测和配对分析。

临时目录、数据缓存和中间 adapter 使用 `/share/jekml/youyang7/tmp` 的作业独立目录。复用管理员 scratch 配额、路径和工作量准入；既有模型缓存不复制。最终 adapter 与训练、逐题预测、汇总指标和图独立保存在 KD 分支；冻结源码和实际 Slurm 脚本均须核验。首次 KD memory admission 为 60000 MiB，包含 8192 MiB 余量；smoke 实测超过该界限即停止，不盲目扩展。

## 放大门槛

全部四次训练与开发预测完整且审计通过后，同时要求：

1. 两个 seed 的 KD 准确率都高于各自 SFT。
2. 两个 seed 平均准确率增益至少 1 个百分点。
3. 按问题配对、跨两个固定学生 seed 取均值的 95% bootstrap 区间下界大于零。
4. 每个 seed 的 KD 平均输出长度不超过 SFT 的 1.10 倍，各评测 cap-hit 不超过 1%。

置信区间只反映固定已训练模型上的题目不确定性，不估计学生 seed 总体变异。门槛未通过则记录阴性或证据不足，停止扩大，不事后改门槛。通过后才能另行冻结 1,024 题配对复制；SAE 和其他 baseline 扩展置于后续证据门槛。当前不提交完整规模矩阵。

## 实现修订与作废尝试

活动执行为 `token_kd_small_v2`，配置 `configs/phase13_token_kd_small_v2.json`。v1 的自定义损失未使用 Transformers 4.48.3 提供的整组梯度累积有效 token 数，而原 Trainer 对支持 loss kwargs 的模型不再额外除以累积步数。因此 v1 不能用于匹配 SFT/KD 的效果推断。在查看任何开发准确率之前发现该问题，停止剩余评测与分析（850151–850155），保留全部原始产物，并记录 `verification/IMPLEMENTATION_INVALID.json`。

v2 使每个微批次 CE/KL token 总和除以整个累积组有效 token 数。新增不等长度微批次与合并批次的损失/梯度等价测试，以及与固定 Transformers 原生 causal loss 的直接对齐测试。题目、顺序、训练超参数、种子、评测和放大门槛均不变，旧冻结快照未修改。850209→850210→850211→850212 为修正版的测试、数据准备、GPU smoke 与小试启动链。

## v2 完整结果（2026-09-16）

四组训练/评测与分析已完成，1,200 条预测无重复或缺失；独立重算指标及 6,040 项哈希核验通过。

| 学生 seed | SFT 准确率 | KD 准确率 | KD − SFT |
| --- | --- | --- | --- |
| 17 | 269/300（89.67%） | 275/300（91.67%） | +2.00 pp |
| 42 | 269/300（89.67%） | 269/300（89.67%） | 0.00 pp |

平均增益 +1.00 pp，配对问题 bootstrap 95% CI [−1.17,+3.33] pp。KD 平均输出分别 286.65/286.51 tokens，SFT 分别 275.53/272.15；所有评测 cap-hit=0。长度门槛通过，但第二个 seed 未正向提升且置信区间跨零，因此预登记放大门槛未通过，不提交 1,024 题扩展。这是证据不足以确认稳定收益，不是证明 KD 无效。

[比较图](../../../results/phase13_baseline_expansion_v1/exploratory/token_kd_small_v2/analysis/sft_vs_kd.png)与[独立复核](../../../results/phase13_baseline_expansion_v1/exploratory/token_kd_small_v2/verification/RESULTS_VERIFIED.json)保存在独立探索分支。开发集此前已被观察，结果不构成独立确认，也不支持 SAE 有效性结论。
