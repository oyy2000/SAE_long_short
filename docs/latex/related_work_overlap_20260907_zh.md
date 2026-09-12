# SAE 推理蒸馏：相关工作与重叠判断

检索日期：2026-09-07。比较对象为本目录的 `sae_reasoning_distillation_abstract.tex` 和用户提供的研究思路图。采用定向检索及原始论文的方法段核对；这不是穷尽性查重，未发现完整同构工作不能证明首创。以下“差异空间”和投稿风险是研究判断，不是论文作者的结论。

## 判断

组成方法的主要环节已有明确先例：短轨迹蒸馏、长短激活对比、SAE 推理干预，以及干预教师后用输出训练学生。当前未找到同时覆盖“同一教师同题正确自然轨迹配对 → SAE 特征发现与排除混淆 → 教师长度干预 → 更小学生的受控蒸馏收益”的单篇论文，但只把这些模块串起来，新颖性仍可能不足。

较有价值的目标是检验：在题目、答案正确性、输出长度分布和训练预算受到控制时，特定特征干预生成的数据是否仍然更有利于学生学习，并解释哪些可操控内容带来了差异。这个结果需要实验建立，不能由特征与长度的相关性推出。

## 最相关的工作

| 论文 | 已有内容 | 与本项目的边界 |
| --- | --- | --- |
| [Small Models Struggle to Learn from Strong Reasoners](https://arxiv.org/abs/2502.12143)，2025 | 小模型未必受益于长 CoT，提出混合蒸馏。 | 不能把“短轨迹可能教得更好”作为首次观察。其原版长短对比使用 QwQ-32B-Preview 与 Qwen2.5-32B-Instruct 两个教师；我们的同教师配对可减少这项混淆，但本身不证明新机制。 |
| [Concise Reasoning, Big Gains / LiteCoT](https://arxiv.org/abs/2505.19716)，2025 | 按题目难度改写长轨迹，再用简洁轨迹蒸馏学生。 | “生成更短训练数据 → SFT 学生”已有先例。区别应落在特征干预及其增益来源，而非仅输出更短。 |
| [Activation Steering for Chain-of-Thought Compression / ASC](https://aclanthology.org/2026.findings-acl.1828/)，2025 预印本、2026 Findings ACL 版本 | 从同题长短 CoT 构造激活干预，直接压缩生成；正式版以长度归一化对比能量及 KL trust region 学习单个方向。 | 这是“潜在差异 → 更短生成”最直接的先例。它采用稠密方向，简洁参考来自标准答案或强模型；其正文还明确讨论以后把方向蒸馏进权重。不能称现有方法只做输出筛选。 |
| [Feature Extraction and Steering for Enhanced Chain-of-Thought Reasoning in Language Models](https://aclanthology.org/2025.emnlp-main.552/)，EMNLP 2025 | 从普通 CoT 提取 SAE 特征并干预生成，也提出 SAE-free 方法。 | “SAE 特征发现 → 干预推理”已有先例。该文主目标是增强被干预模型的推理能力，区别于受控的短轨迹教学效用研究。 |
| [Controllable LLM Reasoning via Sparse Autoencoder-Based Steering](https://aclanthology.org/2026.acl-long.974/)，ACL 2026 | 基于关键词 logit 和实际控制效果两阶段筛选 SAE 特征，控制回溯、交叉验证等策略。 | 不宜把“细粒度 SAE 推理控制”作为新颖性主张。它的主要终点是策略控制和错误纠正。 |
| [Subliminal Learning Is Steering Vector Distillation](https://arxiv.org/abs/2606.00995)，2026 | 直接使用被干预教师的输出训练学生，研究方向及行为的迁移。实验包括 CONCISE 语义方向和随机 SAE decoder 方向。 | “steer teacher → generate → train student”本身也不新。注意两个实验类别不可混写：CONCISE 是 prompt 对比得到的稠密方向，SAE 方向是随机选择；该文不是从长短正确数学轨迹筛选 SAE 简洁特征来做小模型蒸馏。 |

方法段定位：Small Models 的原版 §2.2；ASC 正式版 §3、§5.1–5.3；SAE-Steering §3；Subliminal Learning §5 与 Appendix I。ASC 正式版与最早预印本的方向构造有变化，复现时应明确版本，不能把旧版均值差实现当成正式版 CES。

## 两项直接影响论证的研究

[Do Sparse Autoencoders Identify Reasoning Features in Language Models?](https://arxiv.org/abs/2601.05679)（2026）对对比筛选的 SAE 特征做 token 注入及语义反例检验，发现所研究特征大量依赖表面语言线索，干预的推理收益有限。它不证明所有 SAE 都无法捕捉推理，但要求本项目区分“长短相关特征”“可操控生成特征”和“推理机制”。

[Compress-Distill: Reasoning Trace Compression for Efficient Knowledge Distillation](https://arxiv.org/abs/2606.05988)（2026）发现压缩能节约训练与推理成本，但其主实验中原始轨迹在所有所测规模与两个教师下保持最高准确率。因此，短轨迹优势必须限定在具体数据、学生和训练设置中，不能扩展为普遍规律。

此外，[SEAL: Steerable Reasoning Calibration of Large Language Models for Free](https://arxiv.org/abs/2504.07986)（2025）已通过潜在空间方向调节执行、反思与转换步骤；[Unlocking General Long Chain-of-Thought Reasoning Capabilities via Representation Engineering](https://arxiv.org/abs/2503.11314)（2025）也是长短表征对比干预的相关先例。这里没有将它们列为已经完成相同学生蒸馏实验的工作。

## 建议收紧的贡献与对照

1. 将同教师、同题、均正确的自然采样配对作为识别设计，说明它降低了教师和题目混淆；继续审计词汇、位置、完成格式及按长度平均造成的偏差。
2. 必须比较 SAE 与简单稠密长短方向、ASC、concise prompt、自然最短正确选择、随机 SAE 方向。只有 SAE 的增量收益明确，才能论证采用这一分解工具的价值。
3. 把教师正确率与学生学习收益分开；在同题支持、长度匹配及等样本/等监督 token 的独立比较中检验收益。正确答案不能保证中间步骤全部正确，必要时对关键步骤做额外审计。
4. 把核心因果链写成可逐项否决的检验：目标读回是否成功、输出内容/长度是否改变、改变后的数据是否改善学生。某个阶段成功不能代替后续阶段。
5. 论文主张继续限定 GSM8K；本次文献比较不构成扩展 OOD 或新 GPU 矩阵的执行指令。

可以作为论文研究问题的定位：**Can interventions on length-associated teacher features improve student learning beyond what is explained by shorter supervision alone?** 这是待验证的差异化方向，不是已建立的创新结论。

## 对现有 abstract 的影响

“Length-based selection filters completed traces”作为对一种方法的描述成立，但相关工作已直接控制生成，不能由这句暗示所有前人方法都不操控教师内部状态。摘要中的 SAE-guided 框架也不应冠以首次发现、首次控制或首次蒸馏方向。当前 abstract 保持原样；本文档提供后续修改定位的依据。
